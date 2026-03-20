"""Milestone tracking, rolling summaries, and progress persistence — Ollama version.

Fully self-contained replacement for progress.py. No anthropic dependency.
"""

import json
import os
from datetime import datetime

from memory import MAP_NAMES
from ollama_client import _ollama_chat, MODEL
from game_knowledge import get_relevant_knowledge

PROGRESS_FILE = "logs/progress.json"

# Gym progression for FireRed
GYM_PROGRESSION = [
    {"badge": 1, "leader": "Brock", "city": "Pewter City", "type": "Rock"},
    {"badge": 2, "leader": "Misty", "city": "Cerulean City", "type": "Water"},
    {"badge": 3, "leader": "Lt. Surge", "city": "Vermilion City", "type": "Electric"},
    {"badge": 4, "leader": "Erika", "city": "Celadon City", "type": "Grass"},
    {"badge": 5, "leader": "Koga", "city": "Fuchsia City", "type": "Poison"},
    {"badge": 6, "leader": "Sabrina", "city": "Saffron City", "type": "Psychic"},
    {"badge": 7, "leader": "Blaine", "city": "Cinnabar Island", "type": "Fire"},
    {"badge": 8, "leader": "Giovanni", "city": "Viridian City", "type": "Ground"},
]

DEFAULT_PROGRESS = {
    "badges": 0,
    "maps_visited": [],
    "current_objective": "Find Professor Oak's lab in Pallet Town and get your first Pokemon",
    "tier1_objective": "Get your first Pokemon and defeat Brock at Pewter City Gym for Badge #1",
    "tier2_objective": "Get a starter Pokemon from Professor Oak's lab",
    "tier2_last_action": 0,
    "party_pokemon": [],
    "key_events": [],
    "rolling_summary": "Game just started. No progress yet.",
    "total_actions": 0,
    "last_updated": None,
}


def load_progress() -> dict:
    """Load progress from disk, or return defaults."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            saved = json.load(f)
        for key, default in DEFAULT_PROGRESS.items():
            if key not in saved:
                saved[key] = default
        if not saved.get("tier1_objective"):
            saved["tier1_objective"] = get_tier1_objective(saved.get("badges", 0))
        return saved
    return DEFAULT_PROGRESS.copy()


def save_progress(progress: dict) -> None:
    """Save progress to disk."""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    progress["last_updated"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def update_progress(progress: dict, game_state: dict, action_count: int) -> dict:
    """Update progress based on current game state."""
    progress["total_actions"] = action_count

    map_id = game_state.get("map_id", 0)
    if map_id and map_id not in progress["maps_visited"]:
        progress["maps_visited"].append(map_id)

    return progress


def get_summary_line(progress: dict) -> str:
    """Multi-line summary for injection into prompts, showing all three objective tiers."""
    return (
        f"{progress['badges']} badges | "
        f"{len(progress['maps_visited'])} maps visited | "
        f"Action #{progress['total_actions']}\n"
        f"GOAL: {progress.get('tier1_objective', '')}\n"
        f"STRATEGY: {progress.get('tier2_objective', '')}\n"
        f"CURRENT TASK: {progress.get('current_objective', '')}"
    )


def get_tier1_objective(badges: int) -> str:
    """Determine the high-tier objective based on badge count."""
    if badges >= 8:
        return "Defeat the Elite Four and become the Pokemon Champion"
    gym = GYM_PROGRESSION[badges]
    return f"Defeat {gym['leader']} at {gym['city']} Gym for Badge #{gym['badge']} ({gym['type']}-type)"


def check_tier1_update(progress: dict, game_state: dict) -> bool:
    """Check if tier 1 needs updating when badge count changes."""
    current_badges = game_state.get("badges", 0)
    stored_badges = progress.get("badges", 0)

    if not progress.get("tier1_objective"):
        progress["tier1_objective"] = get_tier1_objective(current_badges)
        print(f"  [tier1] Restored goal: {progress['tier1_objective']}")
        if current_badges == stored_badges:
            return True

    if current_badges != stored_badges:
        progress["badges"] = current_badges
        progress["tier1_objective"] = get_tier1_objective(current_badges)
        print(f"  [tier1] Badge earned! New goal: {progress['tier1_objective']}")
        return True
    return False


def _format_party_with_moves(game_state: dict) -> str:
    """Format party string including moves for strategy prompts."""
    party = game_state.get("party", [])
    party_parts = []
    for p in party:
        mon_str = f"{p['name']} Lv{p['level']} {p['hp']}/{p['max_hp']}HP"
        if p.get("moves"):
            mon_str += f" [{', '.join(p['moves'])}]"
        party_parts.append(mon_str)
    return ", ".join(party_parts) if party_parts else "No Pokemon yet"



def _format_maps_visited(progress: dict) -> str:
    """Format visited-map context for objective planning."""
    map_ids = progress.get("maps_visited", [])
    if not map_ids:
        return "Visited areas: none yet."

    names = []
    for map_id in map_ids[-20:]:
        bank = map_id // 256
        num = map_id % 256
        names.append(MAP_NAMES.get((bank, num), f"MAP_{bank}_{num}"))

    unique_names = []
    seen = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    return "Visited areas: " + ", ".join(unique_names)


def _format_battle_moves(game_state: dict) -> str:
    """Format active battle moves with PP for battle prompts."""
    moves = game_state.get("battle_moves", [])
    if not moves:
        return "Available moves: unknown"
    parts = []
    for move in moves:
        status = f"{move['pp']} PP" if move.get("pp", 0) > 0 else "0 PP (unusable)"
        parts.append(f"{move['name']} [{status}]")
    return "Available moves: " + ", ".join(parts)


def _infer_progress_blockers(game_state: dict) -> str:
    """Summarize progression-relevant facts and likely unmet prerequisites."""
    notes = []
    if not game_state.get("party"):
        notes.append("Party is empty; the player does not have a Pokemon yet.")
    notes.append(f"Badges: {game_state.get('badges', 0)}/8.")

    # Report owned key items and HMs — let the LLM reason about what's needed
    key_items = [
        ("Oak's Parcel", game_state.get("has_oaks_parcel", False)),
        ("Pokedex", game_state.get("has_pokedex", False)),
        ("S.S. Ticket", game_state.get("has_ss_ticket", False)),
        ("Bike Voucher", game_state.get("has_bike_voucher", False)),
        ("Bicycle", game_state.get("has_bicycle", False)),
        ("Silph Scope", game_state.get("has_silph_scope", False)),
        ("Lift Key", game_state.get("has_lift_key", False)),
        ("Tea", game_state.get("has_tea", False)),
        ("Card Key", game_state.get("has_card_key", False)),
        ("Poke Flute", game_state.get("has_poke_flute", False)),
        ("Gold Teeth", game_state.get("has_gold_teeth", False)),
        ("Secret Key", game_state.get("has_secret_key", False)),
    ]
    hms = [
        ("Cut", game_state.get("has_hm01_cut", False)),
        ("Fly", game_state.get("has_hm02_fly", False)),
        ("Surf", game_state.get("has_hm03_surf", False)),
        ("Strength", game_state.get("has_hm04_strength", False)),
        ("Flash", game_state.get("has_hm05_flash", False)),
    ]
    owned_items = [name for name, owned in key_items if owned]
    owned_hm_list = [name for name, owned in hms if owned]
    notes.append(f"Key items: {', '.join(owned_items) if owned_items else 'none'}.")
    notes.append(f"HMs: {', '.join(owned_hm_list) if owned_hm_list else 'none'}.")

    return "Game state:\n- " + "\n- ".join(notes)


def rethink_tier2(game_state: dict, tier1_objective: str, in_battle: bool = False, previous_strategy: str = "") -> str:
    """Use local LLM to determine the mid-level strategy."""
    party_str = _format_party_with_moves(game_state)
    progress_blockers = _infer_progress_blockers(game_state)
    badges = game_state.get("badges", 0)
    map_name = game_state.get("map_name", "?")

    if in_battle:
        enemy = game_state.get("enemy_species", "?")
        enemy_level = game_state.get("enemy_level", "?")
        enemy_hp = game_state.get("enemy_hp", "?")
        battle_moves = _format_battle_moves(game_state)
        battle_knowledge = get_relevant_knowledge(badges, map_name, context="battle")
        prompt = (
            f"You are a Pokemon FireRed battle advisor. "
            f"Player is in battle on {map_name}.\n"
            f"Enemy: Lv{enemy_level} (species#{enemy}) HP:{enemy_hp}\n"
            f"Party: {party_str}\n"
            f"{battle_moves}\n"
            f"{battle_knowledge}\n"
            f"Consider: moves with 0 PP are unusable, type effectiveness, "
            f"whether to fight or flee, and if HP is low enough to need healing.\n"
            f"Reply with ONE concise battle strategy sentence. No explanation."
        )
    else:
        knowledge = get_relevant_knowledge(badges, map_name, context="strategy", game_state=game_state)
        knowledge_block = f"\n{knowledge}\n" if knowledge else ""
        prompt = (
            f"You are a Pokemon FireRed strategy advisor. "
            f"Current goal: {tier1_objective}\n"
            f"Player location: {map_name}\n"
            f"Badges: {badges}\n"
            f"Party: {party_str}\n"
            f"{progress_blockers}\n"
            f"{knowledge_block}\n"
            f"Based on the game knowledge above, what should the player focus on RIGHT NOW?\n"
            f"Priorities: GYM if strong enough, TRAIN if underleveled, "
            f"ITEM if a required story item is needed and nearby, "
            f"TRAVEL if not in the right city yet. "
            f"If the party is empty, getting the first Pokemon is highest priority.\n"
            f"Reply with ONE concise strategy sentence (what to do and why). No explanation."
        )

    return _ollama_chat([{"role": "user", "content": prompt}], model=MODEL, max_tokens=80)


def check_tier2_update(progress: dict, game_state: dict, action_count: int, in_battle: bool = False) -> bool:
    """Check if tier 2 needs updating."""
    last_update = progress.get("tier2_last_action", 0)

    if action_count > 0 and (action_count - last_update) >= 50:
        try:
            tier1 = progress.get("tier1_objective", get_tier1_objective(progress.get("badges", 0)))
            previous_strategy = progress.get("tier2_objective", "") if in_battle else ""
            new_strategy = rethink_tier2(game_state, tier1, in_battle=in_battle, previous_strategy=previous_strategy)
            progress["tier2_objective"] = new_strategy
            progress["tier2_last_action"] = action_count
            print(f"  [tier2] New strategy: {new_strategy}")
            return True
        except Exception as e:
            print(f"  [tier2] Failed: {e}")
    return False


def rethink_objective(game_state: dict, tier2_objective: str = "", in_battle: bool = False) -> str:
    """Use local LLM to determine the immediate next step."""
    party_str = _format_party_with_moves(game_state)
    progress_blockers = _infer_progress_blockers(game_state)
    visited_areas = _format_maps_visited(game_state.get("_progress_context", {}))
    strategy_context = (
        f"\nBattle strategy: {tier2_objective}" if tier2_objective and in_battle
        else f"\nCurrent strategy: {tier2_objective}" if tier2_objective else ""
    )

    if in_battle:
        battle_moves = _format_battle_moves(game_state)
        prompt = (
            f"You are guiding a Pokemon FireRed battle. "
            f"Party: {party_str}.\n"
            f"{battle_moves}"
            f"{strategy_context}\n"
            f"Moves with 0 PP are unusable. "
            f"What should the player do RIGHT NOW in this battle turn? "
            f"Reply with ONLY one specific, actionable sentence."
        )
    else:
        map_name = game_state.get('map_name', '?')
        badges = game_state.get('badges', 0)
        knowledge = get_relevant_knowledge(badges, map_name, context="tactical", game_state=game_state)
        knowledge_block = f"\n{knowledge}" if knowledge else ""
        prompt = (
            f"You are guiding a Pokemon FireRed playthrough. "
            f"The player is currently in {map_name} "
            f"with {badges} badges "
            f"and this party: {party_str}.\n"
            f"{progress_blockers}"
            f"\n{visited_areas}"
            f"{knowledge_block}"
            f"{strategy_context}\n"
            f"Based on the nearby map connections above, describe the next local step. "
            f"Do NOT chase items for later in the game. "
            f"If the party is empty, focus on obtaining the first Pokemon. "
            f"Reply with ONLY one specific, actionable sentence."
        )

    return _ollama_chat([{"role": "user", "content": prompt}], model=MODEL, max_tokens=50)


def generate_rolling_summary(recent_actions: list[dict], old_summary: str) -> str:
    """Write a 2-sentence summary of recent progress using local LLM."""
    actions_text = "\n".join(
        f"  {a['action']} - {a['reason']}" for a in recent_actions[-50:]
    )

    return _ollama_chat(
        [{"role": "user", "content": (
            f"Previous summary: {old_summary}\n\n"
            f"Recent actions:\n{actions_text}\n\n"
            "Write a 2-sentence summary of what happened during these actions. "
            "Focus on meaningful progress (new areas, battles, items, story events). "
            "Be concise."
        )}],
        model=MODEL,
        max_tokens=150,
    )
