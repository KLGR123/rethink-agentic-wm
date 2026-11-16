"""Prompt templates and action parsing for the ScienceWorld agent loop."""

import re

SYSTEM_PROMPT = """\
You are an autonomous science agent in a text-world simulation. Your goal: {instruction}

Keep the task goal in mind. Before each action, briefly plan the next useful step using the
observation and previous results. Do not simply cycle through visible objects or command verbs.
If the task names a different room, navigate there before manipulating unrelated objects.
Use the simulator's command templates below. Replace OBJ placeholders with the exact
visible object names; replace room placeholders with room names. Templates describe syntax,
not a guarantee that every object combination is applicable. Do not invent command verbs.
Navigation: use open <door name> for a closed door, then go to <adjacent room name>.
A room that is not adjacent cannot be reached in one move. The Navigation actions field below
contains exact currently available navigation/open commands. For example, if your observation
shows a closed door to the kitchen, use Action: open door to kitchen, then on the next turn
Action: go to kitchen. Do not try to pick up or move the agent (yourself).
For observation use look at <object> or look in <container>; 'inspect' is not a command.
For a thermometer use use thermometer on <object>, not read thermometer.
For mixing, place ingredients in one container, then mix <container>.
For an ambiguous request, reply with the requested number (e.g. Action: 0).
Use the actual names in the current observation; the examples are syntax, not task solutions. An error is feedback: change the action instead of repeating the same
unsuccessful command in an unchanged state. Focus only on the object requested by the task;
focusing on a wrong object can end the episode.

Action templates (provided by the simulator):
{action_templates}

Respond with one command. You may optionally precede it with a short plan (at most two sentences):
Thought: <optional next step and why it helps the goal>
Action: <command>"""

# "Action:" prefix is optional (small models routinely drop it). The command
# itself is free text, so we take everything up to the end of the line.
_ACTION_RE = re.compile(r"Action\s*:\s*(.+)", re.IGNORECASE)


def build_system_message(instruction: str, action_templates: list[str]) -> dict:
    return {"role": "system", "content": SYSTEM_PROMPT.format(
        instruction=instruction, action_templates="\n".join(action_templates))}


def build_obs_user_message(observation: str, inventory: str = "", *,
                           look: str = "", objects: list[str] | None = None,
                           admissible_commands: list[str] | None = None,
                           previous: dict | None = None) -> dict:
    """State deltas: preserve the simulator reply; repeat auxiliary state only on change.

    Previous state remains in the conversation. Navigation commands are a small exact
    subset of the simulator's list; other commands use the official templates and names.
    This never substitutes an action or hides the simulator's response.
    """
    previous = previous or {}
    parts = [observation.strip()]
    if look.strip() and look.strip() != observation.strip() and look != previous.get("look"):
        parts.append("Current room: " + look.strip())
    if inventory != previous.get("inventory"):
        parts.append("Inventory: " + (inventory.strip() or "(empty)"))
    if objects != previous.get("objects"):
        parts.append("Visible object names: " + "; ".join(dict.fromkeys(objects or [])))
    navigation = [a for a in admissible_commands or [] if a.startswith(("go ", "open "))]
    old_navigation = [a for a in previous.get("admissible_commands", []) if a.startswith(("go ", "open "))]
    if not previous or navigation != old_navigation:
        parts.append("Navigation/open actions: " + "; ".join(navigation))
    return {"role": "user", "content": "\n\n".join(parts)}


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


def nearby_commands(observation: str, failed_action: str, admissible_commands: list[str]) -> list[str]:
    """Retrieve exact simulator strings by lexical overlap, never by task/answer."""
    if "No known action matches" not in observation:
        return []
    stop = {"a", "an", "the", "to", "in", "on", "then", "and"}
    def words(text):
        return set(re.findall(r"[a-z0-9]+", text.lower())) - stop
    query = words(failed_action)
    if not query:
        return []
    verb = failed_action.lower().split()[0]
    ranked = []
    for command in set(admissible_commands):
        tokens = words(command)
        overlap = query & tokens
        if not overlap:
            continue
        score = len(overlap) / len(query | tokens)
        score += 0.25 * (command.lower().split()[0] == verb)
        ranked.append((-score, len(command), command))
    return [command for _, _, command in sorted(ranked)[:6]]
