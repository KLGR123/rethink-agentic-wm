"""Prompt templates and action parsing for the ALFWorld agent loop."""

import re

SYSTEM_PROMPT = """\
You are an autonomous household agent in a text-world simulation. Your goal: {instruction}.

Each turn, reason briefly about the current observation, then emit exactly one action.
Valid actions are natural-language commands drawn from the available actions list, e.g.
  go to <receptacle>
  take <object> from <receptacle>
  put <object> in/on <receptacle>
  open/close <receptacle>
  use <object>
  heat/cool/clean <object> with <appliance>
  look at <object> under/in <light>

Format your action as:
Action: <one of the available actions>"""

OBS_TEMPLATE = """\
{observation}

Available actions: {available_actions}"""

# "Action:" prefix is optional (small models routinely drop it). The command
# itself is free text, so we take everything up to the end of the line.
_ACTION_RE = re.compile(r"Action\s*:\s*(.+)", re.IGNORECASE)


def build_system_message(instruction: str) -> dict:
    return {"role": "system", "content": SYSTEM_PROMPT.format(instruction=instruction)}


def build_obs_user_message(observation: str, admissible_commands: list) -> dict:
    # Drop the duplicated task line TextWorld prints at reset so it is not
    # mistaken for a fresh instruction mid-episode.
    obs = observation
    marker = "Your task is to:"
    if marker in obs:
        obs = obs.split(marker, 1)[0].rstrip(" \n-=")
    cmds = ", ".join(admissible_commands) if admissible_commands else "(none)"
    content = OBS_TEMPLATE.format(observation=obs, available_actions=cmds)
    return {"role": "user", "content": content}


def parse_action(text: str) -> str | None:
    """Return the command from the model output, or None.

    Prefer an explicit "Action: <cmd>" line; otherwise fall back to the last
    non-empty line of the output.
    """
    m = _ACTION_RE.search(text)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else None


def format_action(command: str) -> str:
    return command
