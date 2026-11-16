#!/usr/bin/env python3
"""Prepare ScienceWorld train/val parquet files for verl GRPO training.

Run this in the verl training env: the scenario list comes from tasks.py
(stdlib only), so scienceworld/py4j/Java are not needed here.

The system prompt and observation template are injected by
ScienceWorldAgentLoop at rollout time, not baked into the parquet, so changing
the prompt does not require re-running data prep.

goal_idx is a position in the split's scenario list built by tasks.py
(all tasks in task_id order, each task's variation index inside the split).
The env server shares that module, so the index space cannot drift; the server
additionally checks tasks.py's variation counts against the live simulator at
startup and refuses to boot on a mismatch.

Splits: ScienceWorld partitions each task's variations 50/25/25 into
train/dev/test (PythonInterface.getSets). Defaults: train -> train,
val -> dev. Use --val-split test for held-out evaluation.
"""

import argparse
import os
import sys

# Reuse the server's catalog so the index space cannot drift.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.sciworld.tasks import SPLITS, collect_goals, split_sizes  # noqa: E402


def build_row(goal: dict, split: str, row_index: int) -> dict:
    return {
        "data_source": "sciworld",
        "prompt": [{"role": "user", "content": f"{goal['task']} ({goal['task_name']}, variation {goal['variation_idx']})"}],
        "agent_name": "sciworld_agent",
        "ability": "text_agent",
        "extra_info": {
            "goal_idx": goal["goal_idx"],
            "split": split,
            "task_id": goal["task_id"],
            "task_name": goal["task_name"],
            "variation_idx": goal["variation_idx"],
            "index": row_index,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ScienceWorld parquet data for verl")
    parser.add_argument("--output-dir", default="/fs_mol/liujiarun/rethink-agentic-wm/datasets/sciworld")
    parser.add_argument("--train-split", default="train", choices=list(SPLITS),
                        help="Which ScienceWorld variation split to draw the training set from")
    parser.add_argument("--val-split", default="dev", choices=list(SPLITS),
                        help="Which ScienceWorld variation split to draw the validation set from")
    parser.add_argument("--train-range", type=int, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="Goal index range for train split (default: whole split)")
    parser.add_argument("--val-range", type=int, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="Goal index range for val split (default: whole split)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Split sizes: {split_sizes()}")
    if args.train_split == args.val_split:
        print(
            f"WARNING: train and val both use split={args.train_split}. "
            "Training-set rows will appear in validation unless the ranges are disjoint."
        )

    try:
        import datasets as hf_datasets
    except ImportError:
        print("huggingface datasets not found — install it in this env.")
        sys.exit(1)

    # (hf output name, ScienceWorld split the env server must load, range)
    splits = {
        "train": (args.train_split, args.train_range),
        "val": (args.val_split, args.val_range),
    }

    for out_name, (sw_split, rng) in splits.items():
        goals = collect_goals(sw_split)
        start, end = rng or [0, len(goals)]
        end = min(end, len(goals))
        start = max(0, min(start, end))
        # extra_info.split is sent to the env server on /reset, so it must be
        # the ScienceWorld split name, not the hf output name; goal_idx is the
        # position within that split, which is why the range offset is kept.
        rows = [
            build_row(g, sw_split, i)
            for i, g in enumerate(goals[start:end], start=start)
        ]
        ds = hf_datasets.Dataset.from_list(rows)
        out_path = os.path.join(args.output_dir, f"{out_name}.parquet")
        ds.to_parquet(out_path)
        print(
            f"Wrote {len(rows)} rows to {out_path} "
            f"(sciworld split={sw_split}, goal_idx {start}..{end})"
        )


if __name__ == "__main__":
    main()
