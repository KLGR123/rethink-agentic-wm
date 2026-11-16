"""Environment adapters preserve actor trajectories when observation probing is enabled."""

import asyncio
import importlib
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.experimental.agent_loop.episode_utils import merge_generation_metadata
from verl.utils import tensordict_utils as tu
from verl.utils.tokenizer.continuous_token import ContinuousTokenBuilder
from verl.workers.engine_workers import TrainingWorker


class Tokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(c) for c in text] if tokenize else text

    def decode(self, tokens, **kwargs):
        return "".join(chr(t) for t in tokens)


class Environment:
    def __init__(self, terminal_after):
        self.terminal_after = terminal_after
        self.steps = 0
        self.released = self.closed = False

    def state(self):
        return dict(
            session="test-session",
            instruction="Test task",
            obs=f"STATE_{self.steps}",
            reward=float(self.steps == self.terminal_after),
            done=self.steps == self.terminal_after,
            available_actions={"clickables": ["Buy Now"]},
            admissible_commands=["look"],
            inventory="empty",
        )

    async def reset(self, *args):
        return self.state()

    async def step(self, session, action):
        self.steps += 1
        return self.state()

    async def release(self, session):
        self.released = True

    async def close(self):
        self.closed = True


async def episode(environment, enabled, terminal_after, budget):
    module = importlib.import_module(f"scripts.{environment}.agent_loop")
    cls_name = {"alfworld": "ALFWorldAgentLoop", "sciworld": "ScienceWorldAgentLoop", "webshop": "WebShopAgentLoop"}
    agent = object.__new__(getattr(module, cls_name[environment]))
    agent.config = OmegaConf.create({"actor_rollout_ref": {"actor": {"grad_alignment": {"enable": enabled}}}})
    agent.rollout_config = SimpleNamespace(prompt_length=10000)
    agent.response_length = budget
    agent.max_turns = 3
    agent.loop = asyncio.get_running_loop()
    agent.tokenizer = Tokenizer()
    agent.continuous_token_builder = ContinuousTokenBuilder(agent.tokenizer)
    agent._client = Environment(terminal_after)

    async def generate(**kwargs):
        action = "Action: click[Buy Now]" if environment == "webshop" else "Action: look"
        ids = [ord(c) for c in action]
        return SimpleNamespace(
            token_ids=ids,
            log_probs=[-0.2] * len(ids),
            extra_fields={"min_global_steps": None, "max_global_steps": None},
            num_preempted=0,
        )

    agent.server_manager = SimpleNamespace(generate=generate)
    result = await agent._run_episode({}, global_steps=7, extra_info={"goal_idx": 0})
    assert agent._client.released and agent._client.closed
    return result


@pytest.mark.parametrize("environment", ["alfworld", "sciworld", "webshop"])
@pytest.mark.parametrize("terminal_after,budget", [(1, 10000), (2, 10000), (2, 25)])
def test_observation_probe_preserves_actor_episode(environment, terminal_after, budget):
    plain = asyncio.run(episode(environment, False, terminal_after, budget))
    probed = asyncio.run(episode(environment, True, terminal_after, budget))
    for field in ("prompt_ids", "response_ids", "response_mask", "response_logprobs", "reward_score", "num_turns"):
        assert getattr(plain, field) == getattr(probed, field), field
    assert plain.wm_input_ids is None and plain.wm_obs_ids is None
    assert len(probed.wm_input_ids) == len(probed.wm_obs_ids)
    targets = Tokenizer().decode([t for t, n in zip(probed.wm_input_ids, probed.wm_obs_ids, strict=True) if n])
    assert "STATE_0" not in targets and "<user>" not in targets and "<assistant>" not in targets
    if budget == 10000:
        assert f"STATE_{terminal_after}" in targets
        assert f"STATE_{terminal_after}" not in Tokenizer().decode(probed.prompt_ids + probed.response_ids)
    else:
        assert not targets  # The first observation cannot fit completely.
    assert probed.extra_fields["min_global_steps"] == probed.extra_fields["max_global_steps"] == 7


def test_generation_metadata_keeps_full_version_span():
    acc = {}
    merge_generation_metadata(acc, {}, 4)
    assert not acc
    merge_generation_metadata(acc, {"min_global_steps": None, "max_global_steps": None, "first": True}, 4)
    merge_generation_metadata(acc, {"min_global_steps": 2, "max_global_steps": 6}, 4)
    merge_generation_metadata(acc, {"min_global_steps": 3, "max_global_steps": 5}, 4)
    assert acc == {"min_global_steps": 2, "max_global_steps": 6, "first": True}


@pytest.mark.parametrize("forward_only", [False, True])
def test_engine_defaults_keep_inference_limits_and_explicit_overrides(forward_only):
    worker = SimpleNamespace(
        model_config={"use_remove_padding": True},
        engine_config=SimpleNamespace(
            use_dynamic_bsz=True,
            max_token_len_per_gpu=100,
            infer_max_token_len_per_gpu=200,
            micro_batch_size_per_gpu=2,
            infer_micro_batch_size_per_gpu=4,
            use_fused_kernels=False,
        ),
    )
    data = TensorDict({}, batch_size=[1])
    TrainingWorker._inject_engine_defaults(worker, data, forward_only=forward_only)
    assert tu.get_non_tensor_data(data, "max_token_len_per_gpu", default=None) == (200 if forward_only else 100)
    assert tu.get_non_tensor_data(data, "micro_batch_size_per_gpu", default=None) == (4 if forward_only else 2)
    tu.assign_non_tensor(data, max_token_len_per_gpu=999)
    TrainingWorker._inject_engine_defaults(worker, data, forward_only=forward_only)
    assert tu.get_non_tensor_data(data, "max_token_len_per_gpu", default=None) == 999
