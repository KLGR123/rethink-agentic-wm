"""WebShop multi-turn agent loop for verl GRPO training.

Registers as "webshop_agent" in the verl AgentLoop registry.
Each call to run() drives one complete WebShop episode via the HTTP env server,
returning an AgentLoopOutput with reward_score set to the final WebShop reward,
which bypasses the external RewardManager entirely.
"""

import asyncio
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.experimental.agent_loop.episode_utils import merge_generation_metadata
from verl.experimental.grad_alignment.observations import (
    finalize_observation_targets,
    merge_context_with_observations,
    merge_observation_ids,
)
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput
from scripts.webshop.env_client import EnvClient
from scripts.webshop.prompts import (
    build_obs_user_message,
    build_system_message,
    format_action,
    parse_action,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# An env slot is held for an entire episode (reset -> steps -> release), not
# just for one HTTP call. Without a per-worker cap, every sample on a worker
# opens its episode at once: train_batch_size * n episodes contend for
# WEBSHOP_NUM_ENVS slots, most /reset calls sit in the server's idle queue past
# the client timeout, and the abandoned handlers leak slots until the TTL.
# Keep episodes in flight <= pool size. Must match --num-envs of the server.
_ENV_SERVER_SIZE = int(os.getenv("WEBSHOP_NUM_ENVS", "96"))
_EPISODE_LIMIT_OVERRIDE = int(os.getenv("WEBSHOP_MAX_CONCURRENT_EPISODES", "0"))
_EPISODE_SEM: asyncio.Semaphore | None = None


def _episode_semaphore(limit: int) -> asyncio.Semaphore:
    """Per-process semaphore shared by every episode on this worker.

    AgentLoopBase instances are created per sample, so the limit cannot live on
    the instance. Created on first use so it binds to the actor's running loop.
    """

    global _EPISODE_SEM
    if _EPISODE_SEM is None:
        _EPISODE_SEM = asyncio.Semaphore(limit)
        logger.warning(
            "WebShop agent: capping episodes in flight at %d per worker (%s).",
            limit,
            "WEBSHOP_MAX_CONCURRENT_EPISODES"
            if _EPISODE_LIMIT_OVERRIDE
            else f"WEBSHOP_NUM_ENVS={_ENV_SERVER_SIZE} / workers",
        )
    return _EPISODE_SEM


@register("webshop_agent")
class WebShopAgentLoop(AgentLoopBase):
    """Multi-turn agent loop that drives a WebShop environment episode.

    A single run() call corresponds to one complete episode for one goal.
    The terminal WebShop reward (0–1 float) is returned via reward_score so
    that no external reward function is needed — verl's _postprocess picks it
    up directly and writes rm_scores into the batch.

    Config knobs (read from trainer_config.actor_rollout_ref.rollout):
        response_length — hard cap on total accumulated response tokens
        agent.num_workers — controls parallel episodes across all workers
        multi_turn.enable — must be True (set in the launch script)
    """

    def __init__(self, *args, tools=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        self.max_turns: int = 15
        num_workers = max(1, int(self.rollout_config.agent.num_workers))
        limit = _EPISODE_LIMIT_OVERRIDE or max(1, _ENV_SERVER_SIZE // num_workers)
        self._episode_sem = _episode_semaphore(limit)
        self._client = EnvClient(base_url=os.getenv("WEBSHOP_ENV_URL", "http://127.0.0.1:8000"))

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        # Hold a slot for the whole episode, not just the reset call.
        async with self._episode_sem:
            return await self._run_episode(sampling_params, priority, **kwargs)

    async def _run_episode(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        priority = int(priority)
        extra_info = kwargs.get("extra_info", {})
        goal_idx: int = int(extra_info.get("goal_idx", 0))
        # The dataloader step this sample belongs to. Used as a fallback for the
        # version stamp below when the rollout server has not reported one yet.
        dataloader_steps = int(kwargs.get("global_steps", -1))

        metrics: dict = {}
        env_session: str | None = None
        reward_score: float = 0.0
        turn_count: int = 0
        wm_terminal = None
        track_wm = self.config.actor_rollout_ref.actor.get("grad_alignment", {}).get("enable", False)
        # Merged TokenOutput.extra_fields across turns. The trainer reads
        # min_global_steps/max_global_steps from here for off-policy staleness;
        # dropping them makes the tag None and _compute_metrics raises.
        gen_extra_fields: dict[str, Any] = {}
        last_output: TokenOutput | None = None

        try:
            # 1. Allocate an env slot and get the first observation
            with simple_timer("env_reset", metrics):
                env_data = await self._client.reset(goal_idx)

            env_session = env_data["session"]
            instruction = env_data["instruction"]
            obs = env_data["obs"]
            available_actions = env_data["available_actions"]

            # 2. Build initial messages (system + first user turn)
            messages = [
                build_system_message(instruction),
                build_obs_user_message(obs, available_actions),
            ]

            # 3. Tokenise the initial prompt via the Continuous Token builder
            prompt_ids = await self.ct_build_initial_tokens(messages)

            # Running state for the episode.
            runtime_token_ids = list(prompt_ids)
            response_mask: list[int] = []
            response_logprobs: list[float] | None = []
            # Positive IDs distinguish post-action observation turns. Initial
            # context, generated actions and template delimiters have ID zero.
            observation_ids: list[int] = [0] * len(prompt_ids)
            done = False

            # 4. Action loop
            while not done and turn_count < self.max_turns:
                # Budget check: stop before issuing another generate call if
                # the accumulated response would already exceed the hard cap.
                if len(response_mask) >= self.response_length:
                    break

                # 4a. Generate the next assistant turn
                with simple_timer("generate_sequences", metrics):
                    request_id = uuid4().hex
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=runtime_token_ids,
                        sampling_params=sampling_params,
                        priority=priority,
                    )

                # 4b. Merge generated tokens; response_mask = 1 for these tokens
                last_output = output
                merge_generation_metadata(gen_extra_fields, output.extra_fields, dataloader_steps)
                merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                    runtime_token_ids,
                    output.token_ids,
                    response_mask,
                    response_logprobs,
                    assistant_logprobs=output.log_probs if output.log_probs else None,
                )
                runtime_token_ids = merge_result.token_ids
                # Assistant tokens are never observations. The mask must be extended with
                # the SAME edit the merge applied, boundary tokens included.
                observation_ids = merge_observation_ids(merge_result, observation_ids)
                turn_count += 1

                # 4c. Parse the action from the generated text
                generated_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
                parsed = parse_action(generated_text)

                if parsed is None:
                    # No valid action found. Previously this fell back to
                    # "click[back to search]", which on most pages is not a
                    # clickable: WebShop then no-ops, the observation repeats
                    # and the episode burns all remaining turns for zero
                    # reward. End the episode at 0 instead.
                    logger.warning("No valid action parsed; ending episode: %r", generated_text[-200:])
                    reward_score = 0.0
                    done = True
                    break
                else:
                    action_name, action_arg = parsed
                    action_str = format_action(action_name, action_arg)

                # 4d. Execute the action in the env
                with simple_timer("env_step", metrics):
                    step_data = await self._client.step(env_session, action_str)

                obs = step_data["obs"]
                reward_score = float(step_data["reward"])
                done = bool(step_data["done"])
                available_actions = step_data["available_actions"]

                if done and not track_wm:
                    break

                # Merge once. Terminal observations belong only to the diagnostic
                # sequence; commit nonterminal observations to the actor context.
                next_messages = messages + [build_obs_user_message(obs, available_actions)]
                merge, next_mask, next_logprobs, next_ids = await merge_context_with_observations(
                    self,
                    messages,
                    next_messages,
                    runtime_token_ids,
                    response_mask,
                    observation_ids,
                    response_logprobs,
                )
                if done:
                    wm_terminal = (merge.token_ids, next_ids)
                    break
                messages = next_messages
                runtime_token_ids = merge.token_ids
                response_mask, response_logprobs, observation_ids = next_mask, next_logprobs, next_ids

        finally:
            if env_session is not None:
                await self._client.release(env_session)
            await self._client.close()

        # Guarantee the version stamps exist even if no generation completed
        # (e.g. an immediate parse failure): the V1 trainer casts these tags to
        # int and raises on None.
        fallback = dataloader_steps if dataloader_steps >= 0 else 0
        gen_extra_fields.setdefault("min_global_steps", fallback)
        gen_extra_fields.setdefault("max_global_steps", fallback)

        # 5. Split final token sequence into prompt and response
        total_response_len = len(response_mask)
        prompt_ids_final = runtime_token_ids[: len(runtime_token_ids) - total_response_len]
        response_ids_final = runtime_token_ids[len(runtime_token_ids) - total_response_len :]

        # Apply response_length hard cap
        response_ids_final = response_ids_final[: self.response_length]
        response_mask = response_mask[: self.response_length]
        if response_logprobs:
            response_logprobs = response_logprobs[: self.response_length]

        wm_tokens = wm_ids = None
        if track_wm:
            wm_tokens, wm_ids = finalize_observation_targets(
                runtime_token_ids,
                observation_ids,
                len(prompt_ids_final) + self.response_length,
                wm_terminal,
            )

        return AgentLoopOutput(
            prompt_ids=list(prompt_ids_final),
            response_ids=list(response_ids_final),
            response_mask=list(response_mask),
            response_logprobs=list(response_logprobs) if response_logprobs else None,
            wm_input_ids=wm_tokens,
            wm_obs_ids=wm_ids,
            reward_score=reward_score,
            num_turns=turn_count * 2 + 1,  # system + user assistant pairs
            metrics=AgentLoopMetrics(
                generate_sequences=metrics.get("generate_sequences", 0.0),
                num_preempted=(
                    int(last_output.num_preempted)
                    if last_output is not None and last_output.num_preempted is not None
                    else -1
                ),
            ),
            extra_fields={
                **gen_extra_fields,
                "turn_scores": [],
                "tool_rewards": [],
                "num_env_turns": turn_count,
                "goal_idx": goal_idx,
                # Read verbatim by the V1 AgentLoopWorkerTQ when an agent loop returns
                # several outputs, and by the DAPO group filter when
                # algorithm.filter_groups.metric names a key other than "reward".
                # "acc" mirrors the reward managers' convention and doubles as the
                # val-core variable.
                "reward_extra_info": {"acc": reward_score, "score": reward_score},
            },
        )
