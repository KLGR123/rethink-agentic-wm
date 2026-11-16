#!/usr/bin/env python3
"""Prepare ALFWorld train/val parquet files for verl GRPO training.

Run this in the verl training env: the game list and instruction text come from
the game.tw-pddl JSON files directly, so textworld/alfworld/aiohttp are not
needed here.

The system prompt and observation template are injected by ALFWorldAgentLoop
at rollout time, not baked into the parquet, so changing the prompt does not
require re-running data prep.

goal_idx (the index into the sorted game list) must match what the env server
uses, otherwise goal_idx will refer to different games at train time than at
data-prep time. Both this script and the server share game_files.py, so the
order is identical as long as the underlying json_2.1.1 tree does not change.
"""

import argparse
import os
import sys

# Reuse the server's collector so the index space cannot drift.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.alfworld.game_files import collect_game_files, default_data_path, read_instruction  # noqa: E402


def load_goals(alfworld_data: str, split: str) -> list[dict]:
    games = collect_game_files(alfworld_data, split)
    print(f"Found {len(games)} solvable games in split={split}. Collecting instructions ...")
    goals = []
    for idx, game_file in enumerate(games):
        instruction = read_instruction(game_file)
        goals.append({"goal_idx": idx, "instruction": instruction, "game_file": game_file})
        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(games)}")
    print("Done.")
    return goals


def build_row(goal_idx: int, instruction: str, split: str, row_index: int) -> dict:
    return {
        "data_source": "alfworld",
        "prompt": [{"role": "user", "content": instruction}],
        "agent_name": "alfworld_agent",
        "ability": "text_agent",
        "extra_info": {
            "goal_idx": goal_idx,
            "split": split,
            "index": row_index,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ALFWorld parquet data for verl")
    parser.add_argument("--alfworld-data", default=default_data_path())
    parser.add_argument("--output-dir", default="/fs_mol/liujiarun/rethink-agentic-wm/datasets/alfworld")
    parser.add_argument("--train-split", default="train",
                        choices=["train", "valid_seen", "valid_unseen"],
                        help="Which json_2.1.1 split to draw the training set from")
    parser.add_argument("--val-split", default="valid_seen",
                        choices=["train", "valid_seen", "valid_unseen"],
                        help="Which json_2.1.1 split to draw the validation set from")
    parser.add_argument("--train-range", type=int, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="Goal index range for train split (default: whole split)")
    parser.add_argument("--val-range", type=int, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="Goal index range for val split (default: whole split)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_goals = load_goals(args.alfworld_data, args.train_split)
    val_goals = load_goals(args.alfworld_data, args.val_split)

    train_range = args.train_range or [0, len(train_goals)]
    val_range = args.val_range or [0, len(val_goals)]

    # (hf output name, ALFWorld split the env server must load, goals, range)
    splits = {
        "train": (args.train_split, train_goals, train_range),
        "val": (args.val_split, val_goals, val_range),
    }

    try:
        import datasets as hf_datasets
    except ImportError:
        print("huggingface datasets not found — install it in this env.")
        sys.exit(1)

    for out_name, (alf_split, goals, (start, end)) in splits.items():
        end = min(end, len(goals))
        start = max(0, min(start, end))
        # extra_info.split is sent to the env server on /reset, so it must be
        # the ALFWorld split name, not the hf output name; goal_idx is the
        # position within that split, which is why the range offset is kept.
        rows = [
            build_row(g["goal_idx"], g["instruction"], alf_split, i)
            for i, g in enumerate(goals[start:end], start=start)
        ]
        ds = hf_datasets.Dataset.from_list(rows)
        out_path = os.path.join(args.output_dir, f"{out_name}.parquet")
        ds.to_parquet(out_path)
        print(
            f"Wrote {len(rows)} rows to {out_path} "
            f"(alfworld split={alf_split}, goal_idx {start}..{end})"
        )


if __name__ == "__main__":
    main()
