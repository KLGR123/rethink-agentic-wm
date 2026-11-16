"""Task / variation catalog shared by the ScienceWorld env server and data prep.

Both sides must agree on the goal_idx -> (task_name, variation_idx) mapping, so
the table lives here and is imported by both instead of being duplicated.

This module only uses the standard library: data prep runs in the verl training
env, which has no scienceworld / py4j / Java installed.

ScienceWorld indexes scenarios by (task_name, variation_idx), not by a flat
index. `variation_counts` records `get_max_variations()` for every task; the
server validates this table against the live simulator at startup, and
data prep prints it into the parquet rows so a mismatch is visible.

Splits are therefore virtual: `goal_idx` is a position in the concatenation of
all tasks of one split, ordered by task_id (the order in tasks.json), then by
variation. The env-server `split` field accepts "train", "dev" and "test"
(ScienceWorld's own 50/25/25 variation split, computed in Scala by PythonInterface.getSets).
"""

from typing import Dict, List, Tuple

# (task_id, topic, task label, task_name, max variations)
# task_name is the ScienceWorld identifier passed to env.load(); task_id is the
# paper's numeric id. Order matters: it defines the goal_idx space.
TASKS = [
    ("1-1", "Matter", "Changes of State (Boiling)", "boil", 30),
    ("1-2", "Matter", "Changes of State (Melting)", "melt", 30),
    ("1-3", "Matter", "Changes of State (Freezing)", "freeze", 30),
    ("1-4", "Matter", "Changes of State (Any)", "change-the-state-of-matter-of", 30),
    ("2-1", "Measurement", "Use Thermometer", "use-thermometer", 540),
    ("2-2", "Measurement", "Measuring Boiling Point (known)", "measure-melting-point-known-substance", 436),
    ("2-3", "Measurement", "Measuring Boiling Point (unknown)", "measure-melting-point-unknown-substance", 300),
    ("3-1", "Electricity", "Create a circuit", "power-component", 20),
    ("3-2", "Electricity", "Renewable vs Non-renewable Energy", "power-component-renewable-vs-nonrenewable-energy", 20),
    ("3-3", "Electricity", "Test Conductivity (known)", "test-conductivity", 900),
    ("3-4", "Electricity", "Test Conductivity (unknown)", "test-conductivity-of-unknown-substances", 600),
    ("4-1", "Classification", "Find a living thing", "find-living-thing", 300),
    ("4-2", "Classification", "Find a non-living thing", "find-non-living-thing", 300),
    ("4-3", "Classification", "Find a plant", "find-plant", 300),
    ("4-4", "Classification", "Find an animal", "find-animal", 300),
    ("5-1", "Biology", "Grow a plant", "grow-plant", 126),
    ("5-2", "Biology", "Grow a fruit", "grow-fruit", 126),
    ("6-1", "Chemistry", "Mixing (generic)", "chemistry-mix", 32),
    ("6-2", "Chemistry", "Mixing paints (secondary colours)", "chemistry-mix-paint-secondary-color", 36),
    ("6-3", "Chemistry", "Mixing paints (tertiary colours)", "chemistry-mix-paint-tertiary-color", 36),
    ("7-1", "Biology", "Identify longest-lived animal", "lifespan-longest-lived", 125),
    ("7-2", "Biology", "Identify shortest-lived animal", "lifespan-shortest-lived", 125),
    ("7-3", "Biology", "Identify longest-then-shortest-lived animal", "lifespan-longest-lived-then-shortest-lived", 125),
    ("8-1", "Biology", "Identify life stages (plant)", "identify-life-stages-1", 14),
    ("8-2", "Biology", "Identify life stages (animal)", "identify-life-stages-2", 10),
    ("9-1", "Forces", "Inclined Planes (determine angle)", "inclined-plane-determine-angle", 168),
    ("9-2", "Forces", "Friction (known surfaces)", "inclined-plane-friction-named-surfaces", 1386),
    ("9-3", "Forces", "Friction (unknown surfaces)", "inclined-plane-friction-unnamed-surfaces", 162),
    ("10-1", "Biology", "Mendelian Genetics (known plants)", "mendelian-genetics-known-plant", 120),
    ("10-2", "Biology", "Mendelian Genetics (unknown plants)", "mendelian-genetics-unknown-plant", 480),
]

SPLITS = ("train", "dev", "test")

TASK_NAMES: Dict[str, int] = {t[3]: t[4] for t in TASKS}


def variation_ids(task_name: str, split: str) -> List[int]:
    """Variation indices of one task in one split.

    Mirrors PythonInterface.getSets, which partitions range(max_variations) into
    four equal groups and assigns groups 0+1 to train, 2 to dev and 3 to test
    (extra groups, when max_variations is not divisible by 4, go to test).
    Kept in stdlib form so data prep does not need the JVM.
    """
    if split not in SPLITS:
        raise KeyError(f"Unknown split: {split}. Expected one of {SPLITS}")
    max_var = TASK_NAMES[task_name]
    if max_var <= 0:
        return []
    if max_var == 1:
        return [0] if split in ("train", "test") else []
    if max_var == 2:
        return {"train": [0], "dev": [], "test": [1]}[split]
    if max_var == 3:
        return {"train": [0, 1], "dev": [2], "test": []}[split]
    if max_var == 4:
        return {"train": [0, 1], "dev": [2], "test": [3]}[split]

    group_size = max_var // 4
    groups = [list(range(i, min(i + group_size, max_var))) for i in range(0, max_var, group_size)]
    if split == "train":
        out = groups[0] + groups[1]
    elif split == "dev":
        out = groups[2]
    else:
        out = [v for g in groups[3:] for v in g]
    return out


def collect_goals(split: str) -> List[dict]:
    """Return the split's scenarios in goal_idx order.

    Each entry: {"goal_idx", "task_id", "task", "task_name", "variation_idx"}.
    """
    goals: List[dict] = []
    for task_id, _topic, label, task_name, _max_var in TASKS:
        for var_idx in variation_ids(task_name, split):
            goals.append(
                {
                    "goal_idx": len(goals),
                    "task_id": task_id,
                    "task": label,
                    "task_name": task_name,
                    "variation_idx": var_idx,
                }
            )
    return goals


def split_sizes() -> Dict[str, int]:
    return {split: len(collect_goals(split)) for split in SPLITS}
