#!/usr/bin/env python3
"""Prepare WebShop train val parquet files for verl GRPO training.

Run this with the conda env (Python 3.10 + pyserini) since
it needs to instantiate WebAgentTextEnv to extract goal instructions.

The system prompt and observation template are injected by WebShopAgentLoop
at rollout time, not baked into the parquet, so changing the prompt does not
require re-running data prep.
"""

import argparse
import os
import sys


def load_goals(webshop_path: str, file_path: str = "") -> list[dict]:
    """Instantiate a minimal WebAgentTextEnv and extract all goals + instructions.

    Goals are shuffled once with random.seed(233) inside SimServer.__init__, so the
    index mapping is deterministic as long as file_path matches what server.py
    uses — otherwise goal_idx will refer to different goals at train time than at
    data-prep time.
    """
    
    if webshop_path not in sys.path:
        sys.path.insert(0, webshop_path)
    from web_agent_site.envs import WebAgentTextEnv  # noqa: PLC0415

    print("Constructing WebAgentTextEnv to extract goals ...")
    env_kwargs: dict = dict(observation_mode="text", num_products=None, human_goals=True, file_path=file_path)
    env = WebAgentTextEnv(**env_kwargs)

    goals = []
    total = len(env.server.goals)
    print(f"Found {total} goals. Collecting instructions ...")

    for idx in range(total):
        obs, _ = env.reset(session=idx)
        instruction = env.instruction_text
        goals.append({"goal_idx": idx, "instruction": instruction})
        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{total}")

    env.close()
    print("Done.")
    return goals


def build_row(goal_idx: int, instruction: str, split: str, row_index: int) -> dict:
    return {
        "data_source": "webshop",
        "prompt": [{"role": "user", "content": instruction}],
        "agent_name": "webshop_agent",
        "ability": "web_agent",
        "extra_info": {
            "goal_idx": goal_idx,
            "split": split,
            "index": row_index,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WebShop parquet data for verl")
    parser.add_argument("--webshop-path", default="/fs_mol/liujiarun/WebShop")
    parser.add_argument("--output-dir", default="/fs_mol/liujiarun/rethink-agentic-wm/datasets/webshop")
    parser.add_argument("--file-path", default="/fs_mol/liujiarun/WebShop/data/items_shuffle.json")
    parser.add_argument("--train-range", type=int, nargs=2, default=[1500, 6910],
                        metavar=("START", "END"),
                        help="Goal index range for train split (default: 1500 6910)")
    parser.add_argument("--val-range", type=int, nargs=2, default=[500, 1500],
                        metavar=("START", "END"),
                        help="Goal index range for val split (default: 500 1500)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_goals = load_goals(args.webshop_path, args.file_path)
    splits = {
        "train": (args.train_range[0], args.train_range[1]),
        "val": (args.val_range[0],   args.val_range[1]),
    }

    try:
        import datasets as hf_datasets
    except ImportError:
        print("huggingface datasets not found — install it in this env.")
        sys.exit(1)

    for split_name, (start, end) in splits.items():
        rows = [
            build_row(g["goal_idx"], g["instruction"], split_name, i)
            for i, g in enumerate(all_goals[start:end])
        ]
        ds = hf_datasets.Dataset.from_list(rows)
        out_path = os.path.join(args.output_dir, f"{split_name}.parquet")
        ds.to_parquet(out_path)
        print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
