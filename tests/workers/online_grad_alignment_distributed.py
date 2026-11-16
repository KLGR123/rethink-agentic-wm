"""Real FSDP1/FSDP2 sharded statistics and fixed-weight probe invariants.

PYTHONPATH=. torchrun --standalone --nproc_per_node=2 tests/workers/online_grad_alignment_distributed.py
"""

import copy
import csv
import math
import os
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import fully_shard
from transformers import Qwen2Config, Qwen2ForCausalLM

from verl.experimental.grad_alignment.probe import observation_weights, parameter_group, run_probe
from verl.utils import tensordict_utils as tu
from verl.utils.torch_functional import entropy_from_logits
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def make_engine(version, orig):
    torch.manual_seed(7)
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=57,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=False,
        attention_dropout=0,
    )
    cfg._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(cfg).cuda()
    reference = copy.deepcopy(model)
    if version == 1:
        for i, layer in enumerate(model.model.layers):
            model.model.layers[i] = FSDP(layer, use_orig_params=orig, device_id=torch.cuda.current_device())
        model = FSDP(model, use_orig_params=orig, device_id=torch.cuda.current_device())
    else:
        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.module = model
    engine.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    engine.engine_config = SimpleNamespace(use_no_sync_for_gradient_accumulation=True, entropy_checkpointing=False)
    engine.ulysses_device_mesh = None
    engine.ulysses_sequence_parallel_size = 1
    engine.use_ulysses_sp = False
    engine.pad_to_length = False
    engine.pass_packed_cu_seqlens = False
    engine._autocast_dtype = torch.float32
    engine.scaler = None
    engine.compute_entropy_from_logits = entropy_from_logits
    return engine, reference


def actor_loss(model_output, data, dp_group=None):
    p = model_output["log_probs"].values()
    # A different target subset from observation CE tests full-batch scaling.
    mask = data["loss_mask"].values()
    return -(p * mask).sum() * data["dp_size"] / data["batch_num_tokens"], {}


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world >= 2, "Run with >=2 GPUs to exercise actual sharded gradients"
    torch.manual_seed(13)
    all_ids = torch.randint(0, 128, (world * 2, 9))
    all_obs = torch.tensor([[0, 0, 0, 1, 1, 0, 2, 2, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0]] * world)

    def nested(x):
        return torch.nested.as_nested_tensor(list(x), layout=torch.jagged)

    ids = all_ids[rank * 2 : rank * 2 + 2]
    wm = {"wm_input_ids": nested(ids), "wm_obs_ids": nested(all_obs[rank * 2 : rank * 2 + 2])}
    directory = [tempfile.mkdtemp(prefix="wm-probe-test-") if rank == 0 else None]
    dist.broadcast_object_list(directory, src=0)
    for version, orig in ((1, False), (1, True), (2, False)):
        engine, reference = make_engine(version, orig)
        data = TensorDict(
            {
                "input_ids": nested(ids),
                "position_ids": nested(torch.arange(9).expand(2, -1)),
                "loss_mask": nested(torch.tensor([[1.0, 1, 0, 1, 0, 0, 1, 0, 0]] * 2)),
            },
            batch_size=[2],
        )
        tu.assign_non_tensor(
            data,
            temperature=1.0,
            use_remove_padding=False,
            use_dynamic_bsz=False,
            micro_batch_size_per_gpu=1,
            calculate_entropy=False,
            use_fused_kernels=False,
        )
        config = dict(
            step=version * 10 + int(orig), num_samples=world * 2, output_dir=directory[0], wm_reduction="turn-mean"
        )
        before = torch.cuda.get_rng_state().clone()
        parameter_before = {n: p.detach().clone() for n, p in engine.module.named_parameters()}
        metrics = run_probe(engine, data, wm, actor_loss, config)
        assert torch.equal(before, torch.cuda.get_rng_state()), "probe consumed training RNG"
        assert not engine.optimizer.state, "probe changed optimizer state"
        for name, p in engine.module.named_parameters():
            assert p.grad is None, "probe leaked gradients"
            torch.testing.assert_close(p, parameter_before[name], rtol=0, atol=0)
        # Independent unsharded gradients for the same complete global data.
        logits = reference(all_ids.cuda()).logits
        logp = logits.log_softmax(-1).gather(-1, torch.roll(all_ids.cuda(), -1, 1).unsqueeze(-1)).squeeze(-1)
        loss_mask = torch.tensor([[1.0, 1, 0, 1, 0, 0, 1, 0, 0]] * (world * 2), device="cuda")
        (-(logp * loss_mask).sum() / loss_mask.sum()).backward()
        actor = {n: p.grad.detach().double().clone() for n, p in reference.named_parameters()}
        reference.zero_grad()
        weights, turns, _ = observation_weights(nested(all_obs), "turn-mean")
        w = weights.to_padded_tensor(0).cuda()
        logits = reference(all_ids.cuda()).logits
        lp = logits[:, :-1].log_softmax(-1).gather(-1, all_ids[:, 1:].cuda().unsqueeze(-1)).squeeze(-1)
        (-(lp * w[:, 1:]).sum() / turns).backward()
        dot = a = b = 0.0
        for n, p in reference.named_parameters():
            left, right = actor[n].flatten(), p.grad.double().flatten()
            dot += (left * right).sum().item()
            a += (left * left).sum().item()
            b += (right * right).sum().item()
        # Every reported layer/type must match the unsharded reference, not
        # just the global cosine (which could hide misattributed layer names).
        grouped = {}
        for name, p in reference.named_parameters():
            left, right = actor[name].flatten(), p.grad.double().flatten()
            values = ((left * right).sum().item(), (left * left).sum().item(), (right * right).sum().item())
            layer, kind = parameter_group(name)
            for key in ((layer, kind), (layer, "ALL"), ("ALL", kind), ("ALL", "ALL")):
                acc = grouped.setdefault(key, [0.0, 0.0, 0.0])
                for i, value in enumerate(values):
                    acc[i] += value
        dist.barrier()
        with open(os.path.join(directory[0], f"step_{config['step']}_grad_alignment.csv")) as stream:
            rows = list(csv.DictReader(stream))
        assert {(row["layer"], row["param_type"]) for row in rows} == set(grouped)
        for row in rows:
            dot_ref, a_ref, b_ref = grouped[row["layer"], row["param_type"]]
            for column, value in (("dot", dot_ref), ("norm_rl", math.sqrt(a_ref)), ("norm_wm", math.sqrt(b_ref))):
                torch.testing.assert_close(float(row[column]), value, rtol=2e-5, atol=1e-7)
        expected = dot / (a * b) ** 0.5
        assert abs(metrics["grad_alignment/cosine"] - expected) < 2e-5, (metrics, expected)

        # Compare the next AdamW update, including populated momentum state.
        def update(engine=engine, data=data):
            engine.optimizer_zero_grad()
            engine.forward_backward_batch(data.clone(recurse=False), loss_function=actor_loss, forward_only=False)
            engine.optimizer.step()
            engine.optimizer_zero_grad()

        def parameters(engine=engine):
            return {n: p.detach().clone() for n, p in engine.module.named_parameters()}

        update()  # Populate AdamW moments before the comparison.
        saved_parameters = parameters()
        saved_optimizer = copy.deepcopy(engine.optimizer.state_dict())
        saved_rng = torch.cuda.get_rng_state().clone()
        update()
        expected_parameters = parameters()
        expected_optimizer = copy.deepcopy(engine.optimizer.state_dict())
        with torch.no_grad():
            for n, p in engine.module.named_parameters():
                p.copy_(saved_parameters[n])
        engine.optimizer.load_state_dict(saved_optimizer)
        torch.cuda.set_rng_state(saved_rng)
        run_probe(engine, data, wm, actor_loss, {**config, "step": config["step"] + 100})
        update()
        for n, p in engine.module.named_parameters():
            torch.testing.assert_close(p, expected_parameters[n], rtol=0, atol=0)
        for key, state in engine.optimizer.state_dict()["state"].items():
            for field, value in state.items():
                torch.testing.assert_close(value, expected_optimizer["state"][key][field], rtol=0, atol=0)
        # An empty target batch skips cleanly without another backward.
        empty_wm = {**wm, "wm_obs_ids": nested(torch.zeros_like(ids))}
        skipped = run_probe(engine, data, empty_wm, actor_loss, {**config, "step": config["step"] + 200})
        assert skipped == {"grad_alignment/skipped": 1.0}
        if rank == 0:
            print(
                f"PASS FSDP{version} orig={orig}: cosine={expected:.8f}; "
                "layer statistics match; next AdamW update exactly unchanged; empty batch skipped",
                flush=True,
            )
        del engine, reference
        torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        print("ALL DISTRIBUTED CHECKS PASSED; results", directory[0], flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
