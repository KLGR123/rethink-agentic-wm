"""Small, stateless helpers shared by the text-environment loops."""

from typing import Any


def merge_generation_metadata(acc: dict[str, Any], new: dict[str, Any], fallback_steps: int = -1) -> None:
    """Keep the episode's model-version span; use the data step before first sync."""
    if not new:
        return

    def _coerce(v):
        return fallback_steps if v is None else int(v)

    if not acc:
        acc.update(new)
        acc["min_global_steps"] = _coerce(new.get("min_global_steps"))
        acc["max_global_steps"] = _coerce(new.get("max_global_steps"))
        return

    min_steps = _coerce(new.get("min_global_steps"))
    max_steps = _coerce(new.get("max_global_steps"))
    cur_min = acc.get("min_global_steps")
    acc["min_global_steps"] = min_steps if cur_min is None else min(cur_min, min_steps)
    cur_max = acc.get("max_global_steps")
    acc["max_global_steps"] = max_steps if cur_max is None else max(cur_max, max_steps)
