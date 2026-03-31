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
    subgoal_rewards: set = field(default_factory=set)
    rewarded_tiles: int = 0
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
        bonus = 25.0 * improvement
        _best_steps[flag_id] = steps_taken
        return bonus

    if steps_taken > previous_best * 1.5:
        return -10.0

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
        reward = _completion_reward(flag_id) + bonus
        tracker.reward_total += reward
        return reward, True

    # Player blacked out (all party fainted)
    if _player_blacked_out(after):
        reward = -75.0
        tracker.reward_total += reward
        return reward, True

    # Timeout should be mildly bad so successful shorter runs dominate.
    if tracker.steps >= max_steps:
        reward = -15.0
        tracker.reward_total += reward
        return reward, True

    # --- Step rewards ---
    reward = 0.0

    # Step cost (always), kept small so transition tiles are not overwhelmed by revisit noise.
    reward -= 0.005

    # New tile visited (decays with mastery)
    tile = (after.get("map_id", 0), after.get("player_x", 0), after.get("player_y", 0))
    if tile not in tracker.visited_tiles:
        tracker.visited_tiles.add(tile)
        if tracker.rewarded_tiles < 40:
            reward += _new_tile_reward(flag_id, mastery)
            tracker.rewarded_tiles += 1

    # New map entered (first visit only)
    new_map_id = after.get("map_id", 0)
    if _entered_new_map(before, after) and new_map_id not in tracker.visited_maps:
        tracker.visited_maps.add(new_map_id)
        reward += 5.0

        # Building entered (indoor maps — bank >= 4)
        if _entered_building(before, after):
            reward += 3.0

    # NPC talked to (dialogue opened) — only reward first interaction per location
    if _started_dialogue(before, after):
        npc_tile = (before.get("map_id", 0), before.get("player_x", 0), before.get("player_y", 0))
        if npc_tile not in tracker.talked_npcs:
            tracker.talked_npcs.add(npc_tile)
            reward += 1.0

    # Item picked up (party or inventory changed)
    if _picked_up_item(before, after):
        reward += 10.0

    reward += _subgoal_reward(flag_id, before, after, tracker)

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


def _subgoal_reward(flag_id: str, before: dict, after: dict, tracker: EpisodeTracker) -> float:
    """Small one-time shaping rewards for bottleneck transitions.

    These should only help the agent discover the next meaningful phase of the
    task, not replace the terminal reward.
    """
    reward = 0.0

    if flag_id == "leave_house":
        if (
            before.get("map_name") != "PLAYERS_HOUSE_1F"
            and after.get("map_name") == "PLAYERS_HOUSE_1F"
            and "leave_house:reach_1f" not in tracker.subgoal_rewards
        ):
            tracker.subgoal_rewards.add("leave_house:reach_1f")
            # Treat the first 2F -> 1F transition as the staircase milestone.
            reward += 60.0

    return reward


def _completion_reward(flag_id: str) -> float:
    if flag_id == "leave_house":
        return 300.0
    return 150.0


def _new_tile_reward(flag_id: str, mastery: float) -> float:
    if flag_id == "leave_house":
        return 0.02 * (1.0 - mastery)
    return 0.10 * (1.0 - mastery)


def get_best_steps() -> dict[str, int]:
    """Get the current best steps record for all flags."""
    return dict(_best_steps)


def set_best_steps(data: dict[str, int]) -> None:
    """Restore best steps from saved data."""
    global _best_steps
    _best_steps = dict(data)
