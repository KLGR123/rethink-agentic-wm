"""WebShop multi-turn agent loop for verl GRPO training.

Registers as "webshop_agent" in the verl AgentLoop registry.
Each call to run() drives one complete WebShop episode via the HTTP env server,
returning an AgentLoopOutput with reward_score set to the final WebShop reward,
which bypasses the external RewardManager entirely.
"""

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from verl_webshop.env_client import EnvClient
from verl_webshop.prompts import (
    build_system_message,
    build_obs_user_message,
    parse_action,
    format_action,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


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
        self._client = EnvClient(base_url="http://127.0.0.1:8000")

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        priority = int(priority)
        extra_info = kwargs.get("extra_info", {})
        goal_idx: int = int(extra_info.get("goal_idx", 0))

        metrics: dict = {}
        env_session: str | None = None
        reward_score: float = 0.0
        turn_count: int = 0

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
                merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                    runtime_token_ids,
                    output.token_ids,
                    response_mask,
                    response_logprobs,
                    assistant_logprobs=output.log_probs if output.log_probs else None,
                )
                runtime_token_ids = merge_result.token_ids
                turn_count += 1

                # 4c. Parse the action from the generated text
                generated_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
                parsed = parse_action(generated_text)

                if parsed is None:
                    # No valid action found — take a null step so the episode
                    # progresses and gets 0 reward rather than hanging
                    print("No Valid Action Found in Generated Text:", generated_text)
                    action_str = "click[back to search]"
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

                if done:
                    break

                # 4e. Append observation as a user turn (response_mask = 0)
                prev_messages = list(messages)
                messages.append(build_obs_user_message(obs, available_actions))

                merge_result, response_mask, response_logprobs = await self.ct_merge_context_msg(
                    prev_messages,
                    messages,
                    runtime_token_ids,
                    response_mask,
                    response_logprobs,
                )
                runtime_token_ids = merge_result.token_ids

        finally:
            if env_session is not None:
                await self._client.release(env_session)
            await self._client.close()

        # 5. Split final token sequence into prompt and response
        total_response_len = len(response_mask)
        prompt_ids_final = runtime_token_ids[: len(runtime_token_ids) - total_response_len]
        response_ids_final = runtime_token_ids[len(runtime_token_ids) - total_response_len :]

        # Apply response_length hard cap
        response_ids_final = response_ids_final[: self.response_length]
        response_mask = response_mask[: self.response_length]
        if response_logprobs:
            response_logprobs = response_logprobs[: self.response_length]

        return AgentLoopOutput(
            prompt_ids=list(prompt_ids_final),
            response_ids=list(response_ids_final),
            response_mask=list(response_mask),
            response_logprobs=list(response_logprobs) if response_logprobs else None,
            reward_score=reward_score,
            num_turns=turn_count * 2 + 1, # system + user assistant pairs
            metrics=AgentLoopMetrics(
                generate_sequences=metrics.get("generate_sequences", 0.0),
                num_preempted=int(output.num_preempted) if "output" in dir() and output.num_preempted is not None else -1,
            ),
            extra_fields={
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
