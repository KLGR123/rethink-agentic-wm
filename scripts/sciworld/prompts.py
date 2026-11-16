"""Prompt templates and action parsing for the ScienceWorld agent loop."""

import re

SYSTEM_PROMPT = """\
You are an autonomous science agent in a text-world simulation. Your goal: {instruction}

Each turn, reason briefly about the current observation, then emit exactly one action.
Valid actions are natural-language commands that act on the objects in the current room, e.g.
  look around / look at <object> / inventory
  go to <room> (through a door) / open <door or container>
  take <object> from <container> / move <object> to <location>
  focus on <substance>
  heat <object> with <appliance> / cool <object> with <appliance>
  mix <substance> and <substance>
  connect <wire> to <terminal> / disconnect <wire> from <terminal>
  read <instrument> / measure <substance> with <instrument>
  use <instrument> on <substance>

The available actions list uses generic action templates (e.g. "use X on Y"):
to do something, substitute the object names you see in the observation.

Format your action as:
Action: <one of the available actions>"""

OBS_TEMPLATE = """\
{observation}

Inventory: {inventory}

Available actions: {available_actions}"""

# "Action:" prefix is optional (small models routinely drop it). The command
# itself is free text, so we take everything up to the end of the line.
_ACTION_RE = re.compile(r"Action\s*:\s*(.+)", re.IGNORECASE)


def build_system_message(instruction: str) -> dict:
    return {"role": "system", "content": SYSTEM_PROMPT.format(instruction=instruction)}


def build_obs_user_message(
    observation: str, inventory: str = "", admissible_commands: list | None = None
) -> dict:
    """Render one observation turn from the env's own data, unmodified.

    The admissible list is passed through verbatim: ScienceWorld already decides
    what is legal this turn, and rewriting it (sampling, template induction)
    risks hiding actions the simulator would accept. The full list costs ~400
    tokens per turn, which the 16384 response budget covers for a full episode.
    """
    cmds = list(admissible_commands or [])
    available = ", ".join(cmds) if cmds else "(none)"
    content = OBS_TEMPLATE.format(
        observation=observation.strip(),
        inventory=inventory.strip() or "(empty)",
        available_actions=available,
    )
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
