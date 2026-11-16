"""Two backwards at fixed weights, local gradient shards, tiny reduced outputs.

Runs before the normal actor update. No optimizer/scheduler step, checkpoint
merge, full-gradient all-gather, or gradient dump is involved.
"""

import csv
import json
import math
import os
import random
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor, Shard

from verl.utils import tensordict_utils as tu


def clean_name(name):
    return ".".join(
        x for x in name.split(".") if x not in {"_fsdp_wrapped_module", "_checkpoint_wrapped_module", "_orig_mod"}
    )


def parameter_group(name):
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    if "self_attn." in name:
        kind = "attn"
    elif ".mlp." in name:
        kind = "ffn"
    elif "embed_tokens" in name:
        kind = "embedding"
    elif "lm_head" in name:
        kind = "lm_head"
    elif "norm" in name:
        kind = "norm"
    else:
        kind = "other"
    return (f"layer_{match[1]}" if match else {"embedding": "embed"}.get(kind, kind)), kind


def local_gradients(module):
    """Yield (name, local view, replica correction) without gathering parameters.

    FSDP1 private shard metadata is contained in this adapter. FSDP2 uses DTensor
    placements. Replicated copies contribute only once to global scalar sums.
    """
    world = dist.get_world_size()
    if isinstance(module, FSDP):
        from torch.distributed.fsdp._common_utils import _get_module_fsdp_state

        for prefix, unit in module.named_modules():
            if not isinstance(unit, FSDP):
                continue
            handle = getattr(_get_module_fsdp_state(unit), "_handle", None)
            if handle is None:
                continue
            flat = handle.flat_param
            grad = handle.sharded_grad
            if grad is None:
                raise RuntimeError("Missing FSDP gradient in alignment probe")
            correction = (
                dist.get_world_size(handle.process_group) / world if handle.uses_sharded_strategy else 1 / world
            )
            if handle.uses_sharded_strategy:
                for name, info in zip(flat._fqns, flat._shard_param_infos, strict=True):
                    value = (
                        grad[info.offset_in_shard : info.offset_in_shard + info.numel_in_shard]
                        if info.in_shard
                        else grad[:0]
                    )
                    yield clean_name(f"{prefix}.{name}"), value, correction
            else:
                for name, value in zip(flat._fqns, handle._get_unflat_views(grad), strict=True):
                    yield clean_name(f"{prefix}.{name}"), value, correction
    else:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                raise RuntimeError(f"Missing gradient for {name}")
            grad = param.grad
            if isinstance(grad, DTensor):
                shards = math.prod(
                    grad.device_mesh.size(i)
                    for i, placement in enumerate(grad.placements)
                    if isinstance(placement, Shard)
                )
                yield clean_name(name), grad.to_local(), shards / world
            else:
                yield clean_name(name), grad, 1 / world


def snapshot_gradients(module):
    return {
        name: value.detach().to(device="cpu", dtype=torch.float32, copy=True)
        for name, value, _ in local_gradients(module)
    }


def reduce_alignment(module, actor_grads):
    sums = {}
    for name, value, correction in local_gradients(module):
        left, right = actor_grads.pop(name), value.detach().to(device="cpu", dtype=torch.float32)
        if left.shape != right.shape:
            raise ValueError(f"Gradient shard shape changed: {name}")
        dot = norm_left = norm_right = 0.0
        for a, b in zip(left.reshape(-1).split(1_000_000), right.reshape(-1).split(1_000_000), strict=True):
            a, b = a.double(), b.double()
            dot += torch.dot(a, b).item()
            norm_left += torch.dot(a, a).item()
            norm_right += torch.dot(b, b).item()
        layer, kind = parameter_group(name)
        for group in ((layer, kind), (layer, "ALL"), ("ALL", kind), ("ALL", "ALL")):
            acc = sums.setdefault(group, [0.0, 0.0, 0.0])
            for i, x in enumerate((dot, norm_left, norm_right)):
                acc[i] += correction * x
    if actor_grads:
        raise ValueError("Actor and observation gradient parameter sets differ")
    # Names are structural and identical on all ranks, even for empty local shards.
    groups = sorted(sums)
    stats = torch.tensor([sums[g] for g in groups], dtype=torch.float64, device=torch.cuda.current_device())
    dist.all_reduce(stats)
    rows = []
    for (layer, kind), (dot, a, b) in zip(groups, stats.cpu().tolist(), strict=True):
        valid = all(math.isfinite(x) for x in (dot, a, b)) and a > 0 and b > 0
        rows.append(
            dict(
                layer=layer,
                param_type=kind,
                cosine=dot / math.sqrt(a * b) if valid else float("nan"),
                dot=dot,
                norm_rl=math.sqrt(a),
                norm_wm=math.sqrt(b),
            )
        )
    return rows


def observation_weights(ids, reduction):
    """Target-position weights; token zero cannot have a causal predecessor."""
    weights, turns, tokens = [], 0, 0
    for row in ids.unbind():
        row = row.clone()
        row[0] = 0
        weight = torch.zeros_like(row, dtype=torch.float32)
        for turn_id in row.unique().tolist():
            if turn_id <= 0:
                continue
            selected = row == turn_id
            count = int(selected.sum())
            weight[selected] = 1 / count if reduction == "turn-mean" else 1
            turns += 1
            tokens += count
        weights.append(weight)
    return torch.nested.as_nested_tensor(weights, layout=torch.jagged), turns, tokens


def observation_loss(model_output, data, dp_group=None):
    # Engine log_probs[i] predicts token i+1. First-token weights are zero for
    # EACH sequence, so shifting the packed mask cannot cross sample boundaries.
    weight = torch.roll(data["loss_mask"].values(), shifts=-1, dims=0)
    logp = model_output["log_probs"].values()
    return -(logp * weight).sum() * data["dp_size"] / data["batch_num_tokens"], {}


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def write_result(config, rows, details):
    if dist.get_rank() != 0:
        return
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    step = config["step"]
    for suffix, write in (
        ("grad_alignment.csv", lambda f: _write_csv(f, rows, step)),
        ("metadata.json", lambda f: json.dump({**config, **details}, f, indent=2)),
    ):
        fd, path = tempfile.mkstemp(dir=root, prefix=".alignment-")
        try:
            with os.fdopen(fd, "w") as file:
                write(file)
            os.replace(path, root / f"step_{step}_{suffix}")
        finally:
            if os.path.exists(path):
                os.unlink(path)


def _write_csv(file, rows, step):
    writer = csv.DictWriter(file, fieldnames=["step", "layer", "param_type", "cosine", "dot", "norm_rl", "norm_wm"])
    writer.writeheader()
    writer.writerows({"step": step, **row} for row in rows)


def run_probe(engine, data, wm_inputs, actor_loss, config):
    start = time.monotonic()
    if getattr(engine, "scaler", None) is not None:
        raise ValueError("Alignment probe currently supports bf16/fp32, not loss-scaled fp16")
    weights, turns, tokens = observation_weights(wm_inputs["wm_obs_ids"], config["wm_reduction"])
    counts = torch.tensor([turns, tokens], dtype=torch.long, device=torch.cuda.current_device())
    dist.all_reduce(counts, group=engine.get_data_parallel_group())
    turns, tokens = counts.tolist()
    details = dict(observation_turns=turns, observation_tokens=tokens, torch_version=torch.__version__)
    if not tokens:
        write_result(config, [], {**details, "status": "no_observations"})
        return {"grad_alignment/skipped": 1.0}
    actor_data = data.clone(recurse=False)
    tu.assign_non_tensor(actor_data, global_batch_size=config["num_samples"], return_model_output=False)
    wm_data = data.clone(recurse=False)
    wm_data["input_ids"] = wm_inputs["wm_input_ids"]
    wm_data["position_ids"] = torch.nested.as_nested_tensor(
        [torch.arange(len(row), device=row.device) for row in wm_inputs["wm_input_ids"].unbind()], layout=torch.jagged
    )
    wm_data["loss_mask"] = weights
    tu.assign_non_tensor(wm_data, temperature=1.0, calculate_entropy=False, return_model_output=False)
    # No training state is advanced; even exceptions restore RNG and clear grads.
    with preserve_rng():
        engine.optimizer_zero_grad()
        try:
            engine.forward_backward_batch(actor_data, loss_function=actor_loss, forward_only=False)
            snapshot = snapshot_gradients(engine.module)
            engine.optimizer_zero_grad()
            engine.forward_backward_batch(wm_data, loss_function=observation_loss, forward_only=False)
            rows = reduce_alignment(engine.module, snapshot)
        finally:
            engine.optimizer_zero_grad()
    elapsed = time.monotonic() - start
    write_result(config, rows, {**details, "status": "ok", "seconds": elapsed})
    overall = next(row for row in rows if row["layer"] == row["param_type"] == "ALL")
    return {
        "grad_alignment/cosine": overall["cosine"],
        "grad_alignment/seconds": elapsed,
        "grad_alignment/observation_turns": turns,
        "grad_alignment/observation_tokens": tokens,
    }
