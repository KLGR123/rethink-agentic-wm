"""Game-file discovery shared by the ALFWorld env server and data prep.

Both must agree on the goal_idx -> game file mapping, so the list construction
lives here and is imported by both sides instead of being duplicated.

This module only uses the standard library: data prep runs in the verl training
env, which has no textworld/alfworld installed.
"""

import json
import os
import re

SPLITS = {
    "train": "train",
    "valid_seen": "valid_seen",
    "valid_unseen": "valid_unseen",
}

_TASK_RE = re.compile(r'"rhs"\s*:\s*"Your task is to:\s*([^"]+)"')

# Shared copy on the /fs_mol mount, so every training node sees the same tree
# without running alfworld-download. goal_idx depends on this exact tree, so
# $ALFWORLD_DATA should not point at a different one unless you re-prep data.
DEFAULT_DATA_PATH = "/fs_mol/liujiarun/alfworld/data"


def default_data_path() -> str:
    """$ALFWORLD_DATA if set, else the shared /fs_mol copy."""
    return os.path.expandvars(os.getenv("ALFWORLD_DATA", DEFAULT_DATA_PATH))


def collect_game_files(alfworld_data: str, split: str) -> list[str]:
    """Return the sorted, filtered game files of one split.

    Mirrors AlfredTWEnv.collect_game_files (skip movable receptacles and games
    that are not pre-checked solvable) but sorts the result, because os.walk
    order is filesystem-dependent and goal_idx must be stable across the data
    prep run and every server start.
    """
    if split not in SPLITS:
        raise KeyError(f"Unknown split: {split}")
    root = os.path.join(alfworld_data, "json_2.1.1", SPLITS[split])
    games: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "traj_data.json" not in filenames:
            continue
        if "movable" in dirpath or "Sliced" in dirpath:
            continue
        game_file = os.path.join(dirpath, "game.tw-pddl")
        if not os.path.isfile(game_file):
            continue
        try:
            with open(game_file, "r") as f:
                gamedata = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not gamedata.get("solvable", False):
            continue
        games.append(game_file)
    games.sort()
    return games


def read_instruction(game_file: str) -> str:
    """Extract 'Your task is to: …' from the compiled game's grammar."""
    with open(game_file, "r") as f:
        gamedata = json.load(f)
    match = _TASK_RE.search(gamedata.get("grammar", ""))
    if match:
        return match.group(1).strip()
    return "complete the household task"
