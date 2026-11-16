"""Hint candidates only: never restrict or rewrite submitted actions."""

def accessible_commands(env):
    visible_ids = set(map(str, env.get_possible_object_referent_LUT()))
    candidates = env.get_valid_action_object_combinations_with_templates()
    # The raw candidate API recursively enumerates closed-container contents.
    # Use only IDs from the official visible/accessible referent API. This is
    # conservative: even retained commands need not be semantically applicable.
    return [item["action"] for item in candidates
            if "obj_ids" in item
            and all(str(obj_id) in visible_ids for obj_id in item["obj_ids"])]
