"""Token bookkeeping for action-conditioned observation targets (no torch dependency)."""


def content_span(builder, previous_messages, updated_messages):
    """Locate the new user content by differencing full and empty-content renders.

    Only the appended observation is changed; role delimiters and assistant header
    are the common prefix/suffix. Tokens crossing the content boundary belong to
    the content. Generated assistant tokens are never re-tokenized.
    """
    if len(updated_messages) != len(previous_messages) + 1 or updated_messages[-1]["role"] != "user":
        raise ValueError("Observation tracking expects one appended user message")
    full = builder.tokenize_context_incremental_messages(previous_messages, updated_messages)
    empty_messages = updated_messages[:-1] + [{**updated_messages[-1], "content": ""}]
    empty = builder.tokenize_context_incremental_messages(previous_messages, empty_messages)
    start = 0
    while start < min(len(full), len(empty)) and full[start] == empty[start]:
        start += 1
    suffix = 0
    while suffix < min(len(full), len(empty)) - start and full[-suffix - 1] == empty[-suffix - 1]:
        suffix += 1
    end = len(full) - suffix
    return full, [int(start <= i < end) for i in range(len(full))]


def merge_observation_ids(merge, previous_ids, content_mask=None):
    """Carry per-turn IDs through exactly the same edits as runtime token IDs."""
    kept = previous_ids[: -merge.removed_prefix_token_count] if merge.removed_prefix_token_count else list(previous_ids)
    if merge.kind == "assistant":
        appended = [0] * merge.appended_token_count
    else:
        if content_mask is None or len(content_mask) != merge.appended_token_count:
            raise ValueError("Observation content span does not match appended tokens")
        turn_id = max(previous_ids, default=0) + 1
        appended = [turn_id if value else 0 for value in content_mask]
    result = kept + [0] * len(merge.inserted_token_ids) + appended
    if len(result) != len(merge.token_ids):
        raise ValueError("Observation IDs are not aligned with runtime tokens")
    return result


def finalize_observation_targets(runtime_tokens, observation_ids, budget, terminal=None):
    """Clip diagnostic context, excluding every target in an incomplete turn.

    A terminal observation may extend the sequence only while the executed action
    itself fits the budget. The actor's runtime sequence is never changed.
    """
    if len(runtime_tokens) != len(observation_ids):
        raise ValueError("Observation IDs do not match runtime tokens")
    tokens, ids = runtime_tokens, observation_ids
    if terminal is not None and len(runtime_tokens) <= budget:
        tokens, ids = terminal
    if len(tokens) != len(ids):
        raise ValueError("Terminal observation IDs do not match its tokens")
    incomplete = set(ids[budget:]) - {0}
    return list(tokens[:budget]), [0 if value in incomplete else value for value in ids[:budget]]


async def merge_context_with_observations(
    agent,
    previous_messages,
    updated_messages,
    runtime_token_ids,
    response_mask,
    observation_ids,
    response_logprobs=None,
    tools=None,
):
    merge, response_mask, response_logprobs = await agent.ct_merge_context_msg(
        previous_messages, updated_messages, runtime_token_ids, response_mask, response_logprobs, tools=tools
    )
    if not agent.config.actor_rollout_ref.actor.get("grad_alignment", {}).get("enable", False):
        return merge, response_mask, response_logprobs, [0] * len(merge.token_ids)
    _, mask = await agent.loop.run_in_executor(
        None, lambda: content_span(agent.continuous_token_builder, previous_messages, updated_messages)
    )
    ids = merge_observation_ids(merge, observation_ids, mask)
    return merge, response_mask, response_logprobs, ids
