"""Milestone tracking, rolling summaries, and progress persistence."""

import json
import os
from datetime import datetime

import anthropic

from memory import MAP_NAMES

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

REQUIRED_ITEM_ORDER_MAIN_STORY = [
    "Oak's Parcel",
    "Pokedex",
    "S.S. Ticket",
    "HM01 Cut",
    "Lift Key",
    "Silph Scope",
    "Poké Flute",
    "Tea",
    "Card Key",
    "HM03 Surf",
    "Gold Teeth",
    "HM04 Strength",
    "Secret Key",
]

ITEM_LOCATION_HINTS = {
    "Oak's Parcel": "Viridian City Poké Mart",
    "Pokedex": "Professor Oak's Lab in Pallet Town (after delivering Oak's Parcel)",
    "S.S. Ticket": "Bill's House in Cerulean Cape / Vermilion City route progression",
    "HM01 Cut": "S.S. Anne in Vermilion City",
    "Lift Key": "Team Rocket Hideout in Celadon City",
    "Silph Scope": "Team Rocket Hideout in Celadon City",
    "Poké Flute": "Mr. Fuji in Lavender Town after Pokémon Tower",
    "Tea": "Celadon Mansion in Celadon City",
    "Card Key": "Silph Co. in Saffron City",
    "HM03 Surf": "Safari Zone Secret House in Fuchsia City",
    "Gold Teeth": "Safari Zone in Fuchsia City",
    "HM04 Strength": "Fuchsia City Safari Warden (after returning Gold Teeth)",
    "Secret Key": "Pokémon Mansion on Cinnabar Island",
}

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

client = anthropic.Anthropic()


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


def _format_progress_facts(game_state: dict) -> str:
    """Format authoritative progression booleans for planning prompts."""
    key_facts = [
        ("Pokedex", game_state.get("has_pokedex", False)),
        ("Oak's Parcel", game_state.get("has_oaks_parcel", False)),
        ("S.S. Ticket", game_state.get("has_ss_ticket", False)),
        ("Silph Scope", game_state.get("has_silph_scope", False)),
        ("Poke Flute", game_state.get("has_poke_flute", False)),
        ("Secret Key", game_state.get("has_secret_key", False)),
        ("Card Key", game_state.get("has_card_key", False)),
        ("Lift Key", game_state.get("has_lift_key", False)),
        ("Tea", game_state.get("has_tea", False)),
        ("Bicycle", game_state.get("has_bicycle", False)),
        ("Bike Voucher", game_state.get("has_bike_voucher", False)),
        ("Gold Teeth", game_state.get("has_gold_teeth", False)),
        ("Tri-Pass", game_state.get("has_tri_pass", False)),
        ("Rainbow Pass", game_state.get("has_rainbow_pass", False)),
    ]
    hm_facts = [
        ("Cut", game_state.get("has_hm01_cut", False)),
        ("Fly", game_state.get("has_hm02_fly", False)),
        ("Surf", game_state.get("has_hm03_surf", False)),
        ("Strength", game_state.get("has_hm04_strength", False)),
        ("Flash", game_state.get("has_hm05_flash", False)),
        ("Rock Smash", game_state.get("has_hm06_rock_smash", False)),
        ("Waterfall", game_state.get("has_hm07_waterfall", False)),
    ]
    key_line = ", ".join(f"{name}={'yes' if owned else 'no'}" for name, owned in key_facts)
    hm_line = ", ".join(f"{name}={'yes' if owned else 'no'}" for name, owned in hm_facts)
    missing_key_items = [name for name, owned in key_facts if not owned]
    missing_hms = [name for name, owned in hm_facts if not owned]
    missing_key_line = ", ".join(missing_key_items) if missing_key_items else "none"
    missing_hm_line = ", ".join(missing_hms) if missing_hms else "none"
    return (
        f"Progress facts:\n"
        f"- Key items owned/missing: {key_line}\n"
        f"- Missing key items: {missing_key_line}\n"
        f"- HMs owned/missing: {hm_line}\n"
        f"- Missing HMs: {missing_hm_line}"
    )


def _format_required_item_order(game_state: dict) -> str:
    """Format the main-story required item order with current ownership status."""
    item_status = {
        "Oak's Parcel": game_state.get("has_oaks_parcel", False),
        "Pokedex": game_state.get("has_pokedex", False),
        "S.S. Ticket": game_state.get("has_ss_ticket", False),
        "HM01 Cut": game_state.get("has_hm01_cut", False),
        "Lift Key": game_state.get("has_lift_key", False),
        "Silph Scope": game_state.get("has_silph_scope", False),
        "Poké Flute": game_state.get("has_poke_flute", False),
        "Tea": game_state.get("has_tea", False),
        "Card Key": game_state.get("has_card_key", False),
        "HM03 Surf": game_state.get("has_hm03_surf", False),
        "Gold Teeth": game_state.get("has_gold_teeth", False),
        "HM04 Strength": game_state.get("has_hm04_strength", False),
        "Secret Key": game_state.get("has_secret_key", False),
    }
    ordered = ", ".join(
        f"{item}={'yes' if item_status.get(item, False) else 'no'}"
        for item in REQUIRED_ITEM_ORDER_MAIN_STORY
    )
    return (
        "Main-story required item order (earlier missing items usually gate later progress):\n"
        f"- {ordered}"
    )


def _format_next_required_item(game_state: dict) -> str:
    """Highlight the earliest missing required story item."""
    item_status = [
        ("Oak's Parcel", game_state.get("has_oaks_parcel", False)),
        ("Pokedex", game_state.get("has_pokedex", False)),
        ("S.S. Ticket", game_state.get("has_ss_ticket", False)),
        ("HM01 Cut", game_state.get("has_hm01_cut", False)),
        ("Lift Key", game_state.get("has_lift_key", False)),
        ("Silph Scope", game_state.get("has_silph_scope", False)),
        ("Poké Flute", game_state.get("has_poke_flute", False)),
        ("Tea", game_state.get("has_tea", False)),
        ("Card Key", game_state.get("has_card_key", False)),
        ("HM03 Surf", game_state.get("has_hm03_surf", False)),
        ("Gold Teeth", game_state.get("has_gold_teeth", False)),
        ("HM04 Strength", game_state.get("has_hm04_strength", False)),
        ("Secret Key", game_state.get("has_secret_key", False)),
    ]
    next_missing = next((name for name, owned in item_status if not owned), None)
    if next_missing is None:
        return "Next required story item: none missing from the configured main-story chain."
    location_hint = ITEM_LOCATION_HINTS.get(next_missing, "unknown location")
    return (
        "Next required story item:\n"
        f"- The earliest missing item in the main-story chain is {next_missing}.\n"
        f"- Canonical FireRed location hint for that item: {location_hint}."
    )


def _format_item_location_hints() -> str:
    """Format canonical FireRed item-to-location hints for the planner."""
    ordered = ", ".join(
        f"{item} -> {ITEM_LOCATION_HINTS.get(item, 'unknown')}"
        for item in REQUIRED_ITEM_ORDER_MAIN_STORY
    )
    return (
        "Canonical FireRed location hints for required story items:\n"
        f"- {ordered}"
    )


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
    next_required_item = next(
        (
            name
            for name, owned in [
                ("Oak's Parcel", game_state.get("has_oaks_parcel", False)),
                ("Pokedex", game_state.get("has_pokedex", False)),
                ("S.S. Ticket", game_state.get("has_ss_ticket", False)),
                ("HM01 Cut", game_state.get("has_hm01_cut", False)),
                ("Lift Key", game_state.get("has_lift_key", False)),
                ("Silph Scope", game_state.get("has_silph_scope", False)),
                ("Poké Flute", game_state.get("has_poke_flute", False)),
                ("Tea", game_state.get("has_tea", False)),
                ("Card Key", game_state.get("has_card_key", False)),
                ("HM03 Surf", game_state.get("has_hm03_surf", False)),
                ("Gold Teeth", game_state.get("has_gold_teeth", False)),
                ("HM04 Strength", game_state.get("has_hm04_strength", False)),
                ("Secret Key", game_state.get("has_secret_key", False)),
            ]
            if not owned
        ),
        None,
    )
    if next_required_item:
        notes.append(f"The earliest missing required story item is {next_required_item}.")
    if game_state.get("has_oaks_parcel", False):
        notes.append("Oak's Parcel is currently in inventory.")
    else:
        notes.append("Oak's Parcel is not currently in inventory.")

    owned_hms = game_state.get("owned_hms", [])
    if owned_hms:
        notes.append("Owned HMs: " + ", ".join(owned_hms) + ".")
    else:
        notes.append("No HMs owned yet.")

    return "Progress interpretation notes:\n- " + "\n- ".join(notes)


def rethink_tier2(game_state: dict, tier1_objective: str, in_battle: bool = False, previous_strategy: str = "") -> str:
    """Use Sonnet to determine the mid-level strategy."""
    party_str = _format_party_with_moves(game_state)
    progress_facts = _format_progress_facts(game_state)
    required_item_order = _format_required_item_order(game_state)
    next_required_item = _format_next_required_item(game_state)
    item_location_hints = _format_item_location_hints()
    progress_blockers = _infer_progress_blockers(game_state)
    badges = game_state.get("badges", 0)
    map_name = game_state.get("map_name", "?")

    if in_battle:
        enemy = game_state.get("enemy_species", "?")
        enemy_level = game_state.get("enemy_level", "?")
        enemy_hp = game_state.get("enemy_hp", "?")
        battle_moves = _format_battle_moves(game_state)
        prompt = (
            f"You are a Pokemon FireRed battle advisor. "
            f"Player is in battle on {map_name}.\n"
            f"Enemy: Lv{enemy_level} (species#{enemy}) HP:{enemy_hp}\n"
            f"Party: {party_str}\n"
            f"{battle_moves}\n"
            f"{progress_facts}\n"
            f"What battle strategy should the player use? Consider:\n"
            f"- Moves with 0 PP are unusable and should not be recommended.\n"
            f"- Which moves are super effective or strong against this enemy?\n"
            f"- Should the player fight to train, or flee to conserve HP?\n"
            f"- Is HP low enough to need healing items?\n\n"
            f"Reply with ONE concise battle strategy sentence."
        )
    else:
        prompt = (
            f"You are a Pokemon FireRed strategy advisor. "
            f"Current goal: {tier1_objective}\n"
            f"Player location: {map_name}\n"
            f"Badges: {badges}\n"
            f"Party: {party_str}\n"
            f"{progress_facts}\n"
            f"{required_item_order}\n"
            f"{next_required_item}\n"
            f"{item_location_hints}\n"
            f"{progress_blockers}\n\n"
            f"Based on your knowledge of Pokemon FireRed, what should the player focus on RIGHT NOW "
            f"to make progress toward the goal? Consider:\n"
            f"- These progression facts are authoritative. Do NOT assume the player owns missing key items or HMs.\n"
            f"- Infer the next quest from BOTH the important items/HMs the player already has and the important ones they still do not have.\n"
            f"- Missing progression items often indicate which required story quest has not been completed yet.\n"
            f"- Respect the ordered main-story item chain above; if an earlier required item is still missing, it usually takes priority over later story rewards.\n"
            f"- Strongly prioritize the earliest missing required story item named above when choosing the next strategy.\n"
            f"- Do not recommend a later reward item or story step before the earliest missing required item is addressed.\n"
            f"- Never state an item source unless it is directly implied by the earliest missing-item progression step.\n"
            f"- Use the canonical FireRed item-location hints above when deciding where the player should go for a missing required item.\n"
            f"- Visible room objects are weak evidence for global story-item location; do not assume a nearby item ball or object is the missing required story item unless the progression step directly implies it.\n"
            f"- Use local objects mainly to choose how to execute an already-known local task, not to infer the overall story quest source.\n"
            f"- Prefer naming the next destination or quest step over giving cardinal directions across multiple maps.\n"
            f"- Only mention a direction like north/south/east/west if it is the immediate local move from the current map and you are confident it is correct.\n"
            f"- If the party is empty, acquiring the first Pokemon is the highest priority before any later story or gym objective.\n"
            f"- Use your knowledge of Pokemon FireRed story progression to infer any unmet prerequisites from the current items, badges, HMs, location, and party.\n"
            f"- If a required story or item prerequisite is still missing, prioritize clearing that prerequisite before recommending gym progression.\n"
            f"- Only recommend actions that are consistent with the authoritative facts above.\n"
            f"- Do not rely on generic 'early-game' assumptions; base the strategy only on the explicit facts and required-item ordering shown above.\n"
            f"- Is the party strong enough to challenge the next gym?\n"
            f"- Does the player need to catch more Pokemon for type coverage?\n"
            f"- Are there required HMs or key items needed to progress?\n"
            f"- Does the player need to train or grind levels first?\n"
            f"- Are there story events that must be completed first?\n\n"
            f"Reply with ONE concise strategy sentence (what to do and why)."
        )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=80,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


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
    """Use Sonnet to determine the immediate next step, informed by tier 2 strategy."""
    party_str = _format_party_with_moves(game_state)
    progress_facts = _format_progress_facts(game_state)
    required_item_order = _format_required_item_order(game_state)
    next_required_item = _format_next_required_item(game_state)
    item_location_hints = _format_item_location_hints()
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
            f"{battle_moves}\n"
            f"{progress_facts}"
            f"{strategy_context} "
            f"Moves with 0 PP are unusable. "
            f"What should the player do RIGHT NOW in this battle turn? "
            f"Reply with ONLY one specific, actionable sentence."
        )
    else:
        prompt = (
            f"You are guiding a Pokemon FireRed playthrough. "
            f"The player is currently in {game_state.get('map_name', '?')} "
            f"with {game_state.get('badges', 0)} badges "
            f"and this party: {party_str}.\n"
            f"{progress_facts}\n"
            f"{required_item_order}\n"
            f"{next_required_item}\n"
            f"{item_location_hints}\n"
            f"{progress_blockers}"
            f"\n{visited_areas}"
            f"{strategy_context} "
            f"Treat the progression facts above as authoritative. Do not assume missing key items or HMs are already owned. "
            f"Infer the immediate next quest step from BOTH the important progression items/HMs the player has and the ones they are still missing. "
            f"Use missing progression items as evidence for which required story quest is currently incomplete. "
            f"Respect the ordered main-story item chain above; if an earlier required item is still missing, it usually blocks later story rewards. "
            f"Strongly prioritize the earliest missing required story item named above when choosing the immediate task. "
            f"Do not recommend pursuing a later reward item or story step before that earliest missing required item is addressed. "
            f"Never state an item source unless it is directly implied by the earliest missing-item progression step. "
            f"Use the canonical FireRed item-location hints above when deciding the next destination for a missing required item. "
            f"Visible room objects are weak evidence for global story-item location; do not assume a nearby item ball or object is the missing required story item unless the progression step directly implies it. "
            f"Use local objects mainly to execute an already-known local task, not to infer the overall quest source. "
            f"Do not rely on generic 'early-game' assumptions; base the immediate task only on the explicit facts and required-item ordering shown above. "
            f"Describe only the next local step from the current map; do not bundle multiple later map transitions into one task sentence. "
            f"Only mention a cardinal direction if it is the immediate local move from the current map and you are confident it is correct. "
            f"If the party is empty, the immediate task should focus on obtaining the first Pokemon before any later progression. "
            f"Use your knowledge of Pokemon FireRed to infer the next necessary local step from the current location, story state, items, badges, HMs, and party. "
            f"Only recommend an immediate action that is consistent with those facts. "
            f"What is the immediate next step the player should take right now? "
            f"Reply with ONLY one specific, actionable sentence."
        )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


def generate_rolling_summary(recent_actions: list[dict], old_summary: str) -> str:
    """Ask Claude to write a 2-sentence summary of recent progress."""
    actions_text = "\n".join(
        f"  {a['action']} - {a['reason']}" for a in recent_actions[-50:]
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": (
                f"Previous summary: {old_summary}\n\n"
                f"Recent actions:\n{actions_text}\n\n"
                "Write a 2-sentence summary of what happened during these actions. "
                "Focus on meaningful progress (new areas, battles, items, story events). "
                "Be concise."
            ),
        }],
    )

    return response.content[0].text.strip()
