"""Target semantics and length-normalization tests for the online experiment."""

import torch

from verl.experimental.grad_alignment.observations import (
    content_span,
    finalize_observation_targets,
    merge_observation_ids,
)
from verl.experimental.grad_alignment.probe import observation_weights, parameter_group
from verl.utils.tokenizer.continuous_token import ContinuousTokenBuilder, MergeResult


class Tokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode()) if tokenize else text


def test_content_only_excludes_user_delimiters_and_assistant_header():
    builder = ContinuousTokenBuilder(Tokenizer())
    previous = [{"role": "system", "content": "task"}, {"role": "user", "content": "initial"}]
    updated = previous + [{"role": "user", "content": "new state"}]
    tokens, mask = content_span(builder, previous, updated)
    assert bytes(t for t, m in zip(tokens, mask, strict=True) if m).decode() == "new state"
    merge = MergeResult(token_ids=[1, 2] + tokens, appended_token_count=len(tokens), kind="context")
    ids = merge_observation_ids(merge, [0, 0], mask)
    assert ids[:2] == [0, 0]
    assert set(ids) == {0, 1}


def test_turn_ids_survive_boundary_edits():
    merge = MergeResult(
        token_ids=[1, 2, 99, 3, 4, 5],
        appended_token_count=3,
        kind="context",
        removed_prefix_token_count=1,
        inserted_token_ids=[99],
    )
    assert merge_observation_ids(merge, [0, 1, 0], [0, 1, 0]) == [0, 1, 0, 0, 2, 0]


def test_turn_mean_and_token_mean_have_explicit_weights():
    ids = torch.nested.as_nested_tensor(
        [torch.tensor([0, 0, 1, 1, 2, 2, 2, 2]), torch.tensor([0, 1, 0])], layout=torch.jagged
    )
    weights, turns, tokens = observation_weights(ids, "turn-mean")
    assert (turns, tokens) == (3, 7)
    torch.testing.assert_close(weights.values(), torch.tensor([0, 0, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0, 1, 0]))
    weights, _, _ = observation_weights(ids, "token-mean")
    assert weights.sum() == 7


def test_first_token_and_empty_rows_do_not_become_cross_sequence_targets():
    ids = torch.nested.as_nested_tensor([torch.tensor([1, 0, 0]), torch.tensor([0, 0])], layout=torch.jagged)
    weights, turns, tokens = observation_weights(ids, "turn-mean")
    assert not weights.values().any()
    assert (turns, tokens) == (0, 0)


def test_layer_grouping():
    assert parameter_group("model.layers.12.self_attn.q_proj.weight") == ("layer_12", "attn")
    assert parameter_group("model.layers.12.mlp.down_proj.weight") == ("layer_12", "ffn")


def test_terminal_context_and_truncated_turns():
    runtime = list(range(6))
    obs_ids = [0, 0, 1, 1, 0, 0]
    terminal = (list(range(10)), obs_ids + [0, 2, 2, 0])
    # A complete terminal turn is included without modifying the actor input.
    tokens, ids = finalize_observation_targets(runtime, obs_ids, 10, terminal)
    assert ids == terminal[1]
    assert runtime == list(range(6))
    # A cut terminal turn contributes no partial target.
    _, ids = finalize_observation_targets(runtime, obs_ids, 8, terminal)
    assert ids == [0, 0, 1, 1, 0, 0, 0, 0]
    # Never use terminal context after a truncated action.
    tokens, ids = finalize_observation_targets(runtime, obs_ids, 3, terminal)
    assert tokens == [0, 1, 2]
    assert ids == [0, 0, 0]
