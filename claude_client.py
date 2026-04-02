"""Anthropic API wrapper with prompt caching and model routing."""

import base64
import io
import json
import random
import re
import anthropic
from PIL import Image

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an AI playing Pokemon FireRed (GBA). Your goal: earn all 8 badges and defeat the Elite Four.

You receive STRUCTURED DATA about the game state. The header tells you what mode you're in.

BATTLE CONTROLS:
- You see the battle action menu (Fight/Bag/Pokemon/Run) and your moves with PP.
- Use Left/Right/Up/Down to move the cursor, A to confirm selection.
- To attack: select Fight, then pick a move with PP remaining, press A.
- Press B to go back from move select to action menu.
- You CANNOT exit battle with B. To flee from WILD battles, move cursor to Run and press A.
- You CANNOT flee from TRAINER battles — you must fight.
- Low HP in a wild battle? Consider Run or use a healing item from Bag.

BAG/POKEMON/SUMMARY: Press B to go back.

Respond ONLY with valid JSON:
{"action": "<button>", "reason": "<why, under 40 words>", "display": "<casual 25-word viewer summary>"}"""

NAV_SYSTEM_PROMPT = """You are an AI playing Pokemon FireRed (GBA). Your goal: earn all 8 badges and defeat the Elite Four.

You receive a list of NEARBY EXITS and NPCs. Pick a TARGET to navigate to. The game will pathfind there automatically.

RULES:
- Choose a target that advances your current objective.
- Exits labeled "unknown" haven't been explored yet — explore them if they might lead somewhere useful.
- NPCs labeled "unknown" haven't been talked to yet — talk to them if your objective involves NPCs.
- If you're in the wrong area, pick the exit that leads toward your destination.
- If you need to heal, go to a Pokemon Center.
- Read NPC dialogue summaries to decide if you need to talk to them again.

TARGET TYPES:
- "door:X,Y" — walk to a door/stairs tile and enter it
- "npc:ID" — walk to an NPC and talk to them
- "edge:DIRECTION" — walk to the map edge to leave (north/south/east/west)

Respond ONLY with valid JSON:
{"target": "<target_id>", "reason": "<why, under 40 words>", "display": "<casual 25-word viewer summary>"}"""

client = anthropic.Anthropic()

_NICKNAME_FALLBACKS = [
    "BUD",
    "MOSS",
    "EMBER",
    "GEKKO",
    "SPROUT",
    "FANG",
    "ZAP",
    "NOVA",
]


def generate_pokemon_nickname(species_name: str = "", theme: str = "") -> str:
    """Generate a short Pokemon-related nickname for the naming screen.

    Returns an uppercase ASCII nickname (3-8 chars) or a local fallback on failure.
    """
    species_hint = species_name.strip() or "starter Pokemon"
    theme_hint = theme.strip() or "Pokemon-related"
    prompt = (
        "You are naming a Pokemon in Pokemon FireRed.\n"
        f"Pokemon/context: {species_hint}\n"
        f"Theme hint: {theme_hint}\n"
        "Return ONE short nickname only.\n"
        "Rules:\n"
        "- 3 to 8 characters\n"
        "- ASCII letters only\n"
        "- Pokemon-flavored or creature-flavored\n"
        "- No spaces, punctuation, numbers, or explanation\n"
        "- Prefer something cute or cool, not a human name\n"
    )
    try:
        response = client.messages.create(
            model=HAIKU,
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip().splitlines()[0]
        cleaned = re.sub(r"[^A-Za-z]", "", raw).upper()
        if 3 <= len(cleaned) <= 8:
            return cleaned
    except Exception:
        pass

    return random.choice(_NICKNAME_FALLBACKS)


def get_navigation_target(exploration_summary: str,
                          progress_summary: str,
                          failed_targets: list[str] | None = None) -> dict | None:
    """Ask Claude to pick a navigation target from the exploration summary.

    Returns {"target": str, "reason": str, "display": str} or None on failure.
    Target format: "door:X,Y", "npc:ID", or "edge:DIRECTION".
    """
    # Remove failed targets from the summary so the LLM can't pick them
    filtered_summary = exploration_summary
    if failed_targets:
        lines = filtered_summary.split("\n")
        filtered_lines = []
        for line in lines:
            skip = False
            for ft in failed_targets:
                # Match "door:X,Y" against "door at (X,Y)" lines
                if ft.startswith("door:"):
                    coords = ft.split(":", 1)[1]  # "5,51"
                    x, y = coords.split(",")
                    if f"door at ({x},{y})" in line:
                        skip = True
                        break
                # Match "edge:direction" against "direction edge" lines
                elif ft.startswith("edge:"):
                    direction = ft.split(":", 1)[1]  # "north"
                    if f"{direction} edge" in line:
                        skip = True
                        break
                # Match "npc:ID" against "NPC #ID" lines
                elif ft.startswith("npc:"):
                    npc_id = ft.split(":", 1)[1]  # "4"
                    if f"NPC #{npc_id}" in line:
                        skip = True
                        break
            if not skip:
                filtered_lines.append(line)
        filtered_summary = "\n".join(filtered_lines)

    prompt = (
        f"CURRENT PROGRESS: {progress_summary}\n\n"
        f"{filtered_summary}\n\n"
        "Pick the best target to navigate to."
    )

    try:
        response = client.messages.create(
            model=HAIKU,
            max_tokens=150,
            system=[{
                "type": "text",
                "text": NAV_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": '{"target": "'},
            ],
        )
    except Exception:
        return None

    raw_text = '{"target": "' + response.content[0].text.strip()
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(raw_text[start:end])
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed, dict) or "target" not in parsed:
        return None

    target = str(parsed["target"]).strip()
    # Validate target format
    if not (target.startswith("door:") or target.startswith("npc:") or target.startswith("edge:")):
        return None

    reason = str(parsed.get("reason", "")).strip()[:120]
    display = str(parsed.get("display", reason)).strip()[:80]

    usage_info = {
        "model": HAIKU,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_read": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
    }

    return {"target": target, "reason": reason, "display": display, "usage": usage_info}


def image_to_base64(img: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def _strip_coordinates(text: str) -> str:
    """Remove coordinate/position references from reason text."""
    # Remove patterns like (3,7), (3, 7), position (3,7), coordinates (3,7)
    text = re.sub(r'\(?\d+\s*,\s*\d+\)?', '', text)
    # Remove "at x=3, y=7" style
    text = re.sub(r'[xy]\s*[=:]\s*\d+', '', text)
    # Remove "player_x", "player_y" references
    text = re.sub(r'player_[xy]\s*[=:]\s*\d+', '', text)
    # Remove map names that leak location
    text = re.sub(r'PLAYERS?_HOUSE_[12]F', '', text)
    # Remove minimap references (row, column, col, tiles with quotes)
    text = re.sub(r'\brows?\s*\d+[-–]?\d*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcol(?:umn)?s?\s*\d+[-–]?\d*', '', text, flags=re.IGNORECASE)
    text = re.sub(r"['\"\(][.#@*]['\"\)]", '', text)  # quoted/parens tile chars: '.', "#", (.)
    text = re.sub(r'\bminimap\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\btiles?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bunexplored\b', 'new', text, flags=re.IGNORECASE)
    text = re.sub(r'\bvisit count\b', '', text, flags=re.IGNORECASE)
    # Clean up extra spaces
    text = re.sub(r'  +', ' ', text).strip()
    return text


def build_messages(game_state: dict,
                   recent_actions: list[dict], progress_summary: str,
                   exploration_summary: str | None = None) -> list[dict]:
    """Build the messages array for the API call (battle/bag/menu states)."""
    parts = []

    parts.append(f"CURRENT PROGRESS: {progress_summary}")

    # Section 2: Current state — context depends on game_state
    gs = game_state
    gstate = gs.get("game_state", "overworld")

    party_info = ""
    party = gs.get("party", [])
    if party:
        party_strs = [f"{p['name']} Lv{p['level']} {p['hp']}/{p['max_hp']}HP" for p in party]
        party_info = f"\nParty: {', '.join(party_strs)}"

    # Dialogue text from memory
    dialogue_text = gs.get("dialogue_text", "")
    dialogue_suspect = gs.get("dialogue_text_suspect", False)
    dialogue_line = ""
    if dialogue_text:
        dialogue_line = f"\nDIALOGUE TEXT: \"{dialogue_text}\""
    elif dialogue_suspect:
        dialogue_line = "\nDIALOGUE TEXT: unavailable or unreliable; do not assume repeated A will help."

    if gstate == "battle":
        # Battle context
        enemy_name = gs.get("enemy_species", "?")
        is_trainer = gs.get("is_trainer_battle", False)
        battle_info = f"{'TRAINER BATTLE' if is_trainer else 'WILD BATTLE'}: Lv{gs.get('enemy_level', '?')} {gs.get('enemy_species', '?')} HP:{gs.get('enemy_hp', '?')}"
        if is_trainer:
            battle_info += " (cannot flee!)"

        moves = gs.get("battle_moves", [])
        action_labels = ["Fight", "Bag", "Pokemon", "Run"]
        action_cursor = gs.get("battle_action_cursor", 0)
        move_cursor = gs.get("battle_move_cursor", 0)
        menu_state = gs.get("battle_menu_state", 1)
        # Vanilla FireRed values: 1=action menu, 2=move select, 4=executing/animations

        if menu_state == 2:
            # Move select screen
            battle_info += "\n=== MOVE SELECT ==="
            cursor_move = None
            if moves:
                move_slots = ["(empty)"] * 4
                for m in moves:
                    pp_str = f"{m['pp']} PP"
                    if m["pp"] <= 0:
                        pp_str = "0 PP (unusable)"
                    move_slots[m["slot"]] = f"{m['name']} [{pp_str}]"
                    if m["slot"] == move_cursor:
                        cursor_move = move_slots[m["slot"]]
                # Show as 2x2 grid with position labels
                battle_info += f"\n  Top-Left: {move_slots[0]}  |  Top-Right: {move_slots[1]}"
                battle_info += f"\n  Bot-Left: {move_slots[2]}  |  Bot-Right: {move_slots[3]}"
            pos_names = {0: "Top-Left", 1: "Top-Right", 2: "Bot-Left", 3: "Bot-Right"}
            battle_info += f"\nCursor is on: {cursor_move or '???'} ({pos_names.get(move_cursor, '?')}). Press A to use it."
            battle_info += "\nMoves with 0 PP cannot be used. Left/Right = move horizontal. Up/Down = move vertical. B = back to action menu."

        elif menu_state == 1:
            # Action menu
            action_name = action_labels[action_cursor] if action_cursor < 4 else "?"
            battle_info += "\n=== ACTION MENU ==="
            battle_info += f"\nCursor: [{action_name}]"
            battle_info += "\n  Fight (top-left)   | Bag (top-right)"
            battle_info += "\n  Pokemon (bot-left) | Run (bot-right)"
            if moves:
                move_lines = []
                for m in moves:
                    status = f"{m['pp']} PP" if m["pp"] > 0 else "0 PP (unusable)"
                    move_lines.append(f"{m['name']} [{status}]")
                battle_info += f"\nAvailable moves: {', '.join(move_lines)}"
            battle_info += "\nLeft/Right = horizontal, Up/Down = vertical. A = select. B does nothing here."

        else:
            # Executing/animations (menu_state == 4 or other)
            battle_info += "\nBattle animation in progress. Press A to advance text/animation."

        parts.append(
            f"=== BATTLE ===\n"
            f"{battle_info}\n"
            f"{party_info}{dialogue_line}"
        )

    elif gstate == "bag":
        bag = gs.get("bag_items")
        if bag:
            pocket_keys = ["Items", "KeyItems", "PokeBalls", "TMs", "Berries"]
            display_names = ["Items", "Key Items", "Poke Balls", "TMs & HMs", "Berries"]
            current_idx = bag["current_pocket"]

            # Pocket tabs
            tabs = []
            for i, dn in enumerate(display_names):
                tabs.append(f"[{dn}]" if i == current_idx else dn)

            # Items in current pocket
            pocket_key = pocket_keys[current_idx] if current_idx < 5 else pocket_keys[0]
            items = bag["pockets"].get(pocket_key, [])
            cursor = bag["cursor"]
            scroll = bag["scroll"]

            item_lines = []
            for idx, item in enumerate(items):
                marker = " >> " if idx == scroll + cursor else "    "
                qty_str = f" x{item['quantity']}" if item["quantity"] > 1 else ""
                item_lines.append(f"{marker}{item['name']}{qty_str}")

            bag_display = f"=== BAG ===\nPockets: {' | '.join(tabs)}\n"
            if item_lines:
                bag_display += "\n".join(item_lines)
            else:
                bag_display += "    (empty)"
            bag_display += "\nLeft/Right = switch pocket. Up/Down = scroll. A = use item. B = close bag."

            parts.append(bag_display + f"{party_info}{dialogue_line}")
        else:
            parts.append(
                f"=== BAG ===\n"
                f"Bag is open. Left/Right = switch pocket. Up/Down = scroll. A = use. B = close."
                f"{party_info}{dialogue_line}"
            )

    elif gstate == "pokemon":
        parts.append(
            f"=== POKEMON MENU (in battle) ===\n"
            f"You have the Pokemon menu open. Use Up/Down to select, A to choose, B to go back."
            f"{party_info}{dialogue_line}"
        )

    elif gstate == "summary":
        parts.append(
            f"=== POKEMON SUMMARY ===\n"
            f"Viewing a Pokemon's summary. Press B to go back."
            f"{party_info}{dialogue_line}"
        )

    elif gstate == "transition":
        parts.append(
            f"=== TRANSITION ===\n"
            f"Game is transitioning. Press A or wait."
            f"{party_info}"
        )

    else:
        # Overworld / unknown
        parts.append(
            f"=== CURRENT STATE ===\n"
            f"Position: ({gs.get('player_x', '?')}, {gs.get('player_y', '?')})\n"
            f"Map: {gs.get('map_name', 'UNKNOWN')}\n"
            f"Badges: {gs.get('badges', 0)} | Dialogue: {gs.get('in_dialogue', False)}"
            f"{party_info}{dialogue_line}"
        )

    state_text = "\n\n".join(parts)

    return [
        {
            "role": "user",
            "content": state_text,
        }
    ]


def get_action(game_state: dict,
               recent_actions: list[dict], progress_summary: str) -> tuple[dict, dict]:
    """Call Claude for battle/bag/menu actions. Returns (parsed_action, usage_info)."""
    # Use Sonnet for battles (better tactical decisions), Haiku for menus
    gstate = game_state.get("game_state", "")
    model = SONNET if gstate == "battle" else HAIKU

    messages = build_messages(game_state, recent_actions, progress_summary)

    # Add assistant prefill for Haiku — force JSON output
    if "haiku" in model:
        messages.append({"role": "assistant", "content": '{"action": "'})

    response = client.messages.create(
        model=model,
        max_tokens=400,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=messages,
    )

    raw_text = response.content[0].text.strip()

    # If we used prefill, prepend the prefilled portion
    if "haiku" in model:
        raw_text = '{"action": "' + raw_text

    # Parse JSON response, with fallback
    parsed = None
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                parsed = json.loads(raw_text[start:end])
            except json.JSONDecodeError:
                pass

    VALID_BUTTONS = {"A", "B", "Up", "Down", "Left", "Right", "Start", "Select"}
    BUTTON_NORMALIZE = {b.lower(): b for b in VALID_BUTTONS}

    # Normalize and validate parsed action
    if parsed and "action" in parsed:
        raw_action = parsed["action"]
        if isinstance(raw_action, str):
            # Take first word only (LLM sometimes returns "Right Right" etc.)
            first_word = raw_action.strip().split()[0] if raw_action.strip() else ""
            normalized = BUTTON_NORMALIZE.get(first_word.lower())
        else:
            normalized = None
        if normalized:
            parsed["action"] = normalized
        else:
            parsed = None  # force fallback

    if not parsed or "action" not in parsed:
        # Last resort: look for a button name in the text (whole word match)
        for btn in ["Start", "Select", "Up", "Down", "Left", "Right"]:
            if re.search(r'\b' + btn + r'\b', raw_text):
                parsed = {"action": btn, "reason": raw_text[:80]}
                break
        # Check A and B last with stricter matching to avoid matching "action"
        if not parsed:
            for btn in ["A", "B"]:
                if re.search(r'(?<![a-zA-Z])' + btn + r'(?![a-zA-Z])', raw_text):
                    parsed = {"action": btn, "reason": raw_text[:80]}
                    break
        if not parsed:
            parsed = {"action": "A", "reason": "Failed to parse, defaulting to A"}

    usage_info = {
        "model": model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_read": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
    }

    return parsed, usage_info
