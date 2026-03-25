"""Reward function for RL navigation episodes.

Simplified reward structure — no per-flag correct_map or correct_npc rewards.
The model learns optimal paths purely from exploration + flag completion signal.
"""

from dataclasses import dataclass, field


@dataclass
class EpisodeTracker:
    """Tracks per-episode state for reward computation."""
    steps: int = 0
    visited_tiles: set = field(default_factory=set)
    coverage_milestones: set = field(default_factory=set)  # {0.25, 0.50, 0.75}
    total_map_tiles: dict = field(default_factory=dict)    # map_id -> tile count estimate
    talked_npcs: set = field(default_factory=set)          # (map_id, x, y) of NPCs already talked to
    visited_maps: set = field(default_factory=set)         # map_ids already entered
    reward_total: float = 0.0


# Global best steps per flag (persists across episodes)
_best_steps: dict[str, int] = {}


def efficiency_bonus(flag_id: str, steps_taken: int) -> float:
    """Bonus for beating personal best step count."""
    if flag_id not in _best_steps:
        _best_steps[flag_id] = steps_taken
        return 0.0

    previous_best = _best_steps[flag_id]

    if steps_taken < previous_best:
        improvement = (previous_best - steps_taken) / previous_best
        bonus = 50.0 * improvement
        _best_steps[flag_id] = steps_taken
        return bonus

    if steps_taken > previous_best * 1.5:
        return -20.0

    return 0.0


def compute_reward(
    before: dict,
    after: dict,
    flag_check: callable,
    flag_id: str,
    tracker: EpisodeTracker,
    mastery: float,
    max_steps: int,
) -> tuple[float, bool]:
    """Compute step reward and whether episode is done.

    Args:
        before: game state before action
        after: game state after action
        flag_check: function(state) -> bool for flag completion
        flag_id: current flag identifier
        tracker: episode tracking state
        mastery: 0.0 (never seen) to 1.0 (fully mastered)
        max_steps: episode timeout limit

    Returns:
        (reward, done) tuple
    """
    tracker.steps += 1

    # --- Terminal conditions ---

    # Flag completed
    if flag_check(after):
        bonus = efficiency_bonus(flag_id, tracker.steps)
        return 100.0 + bonus, True

    # Player blacked out (all party fainted)
    if _player_blacked_out(after):
        return -100.0, True

    # Episode timeout — end episode but no extra penalty (step costs are enough)
    if tracker.steps >= max_steps:
        return 0.0, True

    # --- Step rewards ---
    reward = 0.0

    # Step cost (always)
    reward -= 0.01

    # Wall bash — position didn't change despite moving
    if _position_unchanged(before, after):
        reward -= 0.1

    # New tile visited (decays with mastery)
    tile = (after.get("map_id", 0), after.get("player_x", 0), after.get("player_y", 0))
    if tile not in tracker.visited_tiles:
        tracker.visited_tiles.add(tile)
        reward += 1.0 * (1.0 - mastery)

    # New map entered (first visit only)
    new_map_id = after.get("map_id", 0)
    if _entered_new_map(before, after) and new_map_id not in tracker.visited_maps:
        tracker.visited_maps.add(new_map_id)
        reward += 25.0

        # Building entered (indoor maps — bank >= 4)
        if _entered_building(before, after):
            reward += 25.0

    # NPC talked to (dialogue opened) — only reward first interaction per location
    if _started_dialogue(before, after):
        npc_tile = (before.get("map_id", 0), before.get("player_x", 0), before.get("player_y", 0))
        if npc_tile not in tracker.talked_npcs:
            tracker.talked_npcs.add(npc_tile)
            reward += 3.0

    # Item picked up (party or inventory changed)
    if _picked_up_item(before, after):
        reward += 5.0

    # Map coverage milestones (25%, 50%, 75%)
    current_map = after.get("map_id", 0)
    tiles_on_map = sum(1 for t in tracker.visited_tiles if t[0] == current_map)
    # Estimate map size — grows as we discover tiles
    if current_map not in tracker.total_map_tiles:
        tracker.total_map_tiles[current_map] = max(tiles_on_map, 20)
    else:
        tracker.total_map_tiles[current_map] = max(
            tracker.total_map_tiles[current_map], tiles_on_map
        )
    estimated_size = tracker.total_map_tiles[current_map]
    coverage = tiles_on_map / max(estimated_size, 1)

    for milestone in [0.25, 0.50, 0.75]:
        key = (current_map, milestone)
        if coverage >= milestone and key not in tracker.coverage_milestones:
            tracker.coverage_milestones.add(key)
            reward += 5.0

    tracker.reward_total += reward
    return reward, False


def _position_unchanged(before: dict, after: dict) -> bool:
    return (
        before.get("player_x") == after.get("player_x")
        and before.get("player_y") == after.get("player_y")
        and before.get("map_id") == after.get("map_id")
    )


def _entered_new_map(before: dict, after: dict) -> bool:
    return before.get("map_id") != after.get("map_id")


def _entered_building(before: dict, after: dict) -> bool:
    """Indoor maps have bank >= 4 in FireRed's map system."""
    if before.get("map_id") == after.get("map_id"):
        return False
    before_bank = before.get("map_bank", 0)
    after_bank = after.get("map_bank", 0)
    # Went from outdoor (bank 1-3) to indoor (bank >= 4)
    return before_bank < 4 and after_bank >= 4


def _started_dialogue(before: dict, after: dict) -> bool:
    return not before.get("in_dialogue", False) and after.get("in_dialogue", False)


def _picked_up_item(before: dict, after: dict) -> bool:
    # Detect party count increase (got pokemon) or key item acquisition
    if after.get("party_count", 0) > before.get("party_count", 0):
        return True
    # Check key items that appeared
    key_items = [
        "has_pokedex", "has_oaks_parcel", "has_ss_ticket", "has_bicycle",
        "has_bike_voucher", "has_silph_scope", "has_poke_flute",
    ]
    for item in key_items:
        if after.get(item, False) and not before.get(item, False):
            return True
    return False


def _player_blacked_out(state: dict) -> bool:
    """All party Pokemon fainted."""
    party = state.get("party", [])
    if not party:
        return False
    return all(m.get("hp", 0) <= 0 for m in party)


def get_best_steps() -> dict[str, int]:
    """Get the current best steps record for all flags."""
    return dict(_best_steps)


def set_best_steps(data: dict[str, int]) -> None:
    """Restore best steps from saved data."""
    global _best_steps
    _best_steps = dict(data)
