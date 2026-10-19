"""Prompt templates and action parsing for the WebShop agent loop."""

import re

SYSTEM_PROMPT = """\
You are an autonomous shopping agent. Your goal: {instruction}.

Each turn, reason briefly about the current page, then emit exactly one action.
Valid actions:
  search[keywords] — only available on the search page
  click[button] — button must exactly match one of the available options

Format your action as:
Action: search[keywords]
  or
Action: click[button name]"""

OBS_TEMPLATE = """\
{observation}

Available actions: {available_actions}"""

_ACTION_RE = re.compile(r"Action:\s*(search|click)\[(.+?)\]", re.IGNORECASE)


def build_system_message(instruction: str) -> dict:
    return {"role": "system", "content": SYSTEM_PROMPT.format(instruction=instruction)}


def build_obs_user_message(observation: str, available_actions: dict) -> dict:
    clickables = available_actions.get("clickables", [])
    has_search = available_actions.get("has_search_bar", False)
    parts = []
    if has_search:
        parts.append("search[...]")
    parts.extend(f"click[{c}]" for c in clickables)
    actions_str = ", ".join(parts) if parts else "(none)"
    content = OBS_TEMPLATE.format(observation=observation, available_actions=actions_str)
    return {"role": "user", "content": content}


def parse_action(text: str) -> tuple[str, str] | None:
    """Return the last (action_name, action_arg) from the model output, or None."""
    matches = _ACTION_RE.findall(text)
    if not matches:
        return None
    name, arg = matches[-1]
    return name.lower(), arg.strip()


def format_action(name: str, arg: str) -> str:
    return f"{name}[{arg}]"
