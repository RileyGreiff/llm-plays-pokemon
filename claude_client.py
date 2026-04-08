"""Anthropic API wrapper with prompt caching and model routing."""

import base64
import io
import json
import random
import re
import time
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
- "door:X,Y" — walk to a door tile and enter it
- "stairs:X,Y" — walk to a stairs tile and use it
- "npc:ID" — walk to an NPC and talk to them
- "edge:DIRECTION" — walk to the map edge to leave (north/south/east/west)

Respond ONLY with valid JSON:
{"target": "<target_id>", "reason": "<why, under 40 words>", "display": "<casual 25-word viewer summary>"}"""

SYSTEM_PROMPT = """Play Pokemon FireRed and reply with JSON only.

Valid buttons: A, B, Up, Down, Left, Right, Start, Select.

Battle:
- Action menu is Fight / Bag / Pokemon / Run.
- Move menu chooses a move with PP > 0.
- B only backs out of move select.
- Trainer battles cannot flee.
- Wild battles may flee by selecting Run.

Menus:
- In bag, pokemon, and summary menus, B backs out.

Return exactly:
{"action": "<button>", "reason": "<why, under 40 words>", "display": "<casual 25-word viewer summary>"}"""

NAV_SYSTEM_PROMPT = """Pick one navigation target for Pokemon FireRed.

Prefer targets that advance the current objective.
Unknown exits and unknown NPCs are worth exploring when relevant.
If you are in the wrong area, choose the exit that best moves toward the destination.

Valid target formats:
- "door:X,Y"
- "stairs:X,Y"
- "npc:ID"
- "edge:DIRECTION"

Return exactly:
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
    Target format: "door:X,Y", "stairs:X,Y", "npc:ID", or "edge:DIRECTION".
    """
    # Remove failed targets from the summary so the LLM can't pick them
    filtered_summary = exploration_summary
    if failed_targets:
        lines = filtered_summary.split("\n")
        filtered_lines = []
        for line in lines:
            skip = False
            for ft in failed_targets:
                # Match "door:X,Y" or "stairs:X,Y" against exit lines
                if ft.startswith("door:") or ft.startswith("stairs:"):
                    coords = ft.split(":", 1)[1]  # "5,51"
                    x, y = coords.split(",")
                    if f"door at ({x},{y})" in line or f"stairs at ({x},{y})" in line:
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
        f"PROGRESS: {_compact_progress_summary(progress_summary, battle=False)}\n\n"
        f"{filtered_summary}\n\n"
        "Pick the best target."
    )

    try:
        print("  [claude:navigation] Requesting navigation target")
        started = time.monotonic()
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
        elapsed = time.monotonic() - started
        print(f"  [claude:navigation] Response received in {elapsed:.2f}s")
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
    if not (
        target.startswith("door:")
        or target.startswith("stairs:")
        or target.startswith("npc:")
        or target.startswith("edge:")
    ):
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


def _compact_progress_summary(progress_summary: str, *, battle: bool = False) -> str:
    """Compress the verbose progress summary into a shorter prompt line."""
    lines = [line.strip() for line in progress_summary.splitlines() if line.strip()]
    header = lines[0] if lines else ""
    goal = next((line[5:].strip() for line in lines if line.startswith("GOAL:")), "")
    strategy = next((line[9:].strip() for line in lines if line.startswith("STRATEGY:")), "")
    task = next((line[13:].strip() for line in lines if line.startswith("CURRENT TASK:")), "")

    parts = []
    if header and not battle:
        parts.append(header)
    if goal:
        parts.append(f"goal={goal}")
    if strategy and not battle:
        parts.append(f"strategy={strategy}")
    if task:
        parts.append(f"task={task}")
    return " | ".join(parts)


def _cursor_slot_name(slot: int) -> str:
    return {0: "TL", 1: "TR", 2: "BL", 3: "BR"}.get(slot, "?")


def build_messages(game_state: dict,
                   recent_actions: list[dict], progress_summary: str,
                   exploration_summary: str | None = None) -> list[dict]:
    """Build the messages array for the API call (battle/bag/menu states)."""
    parts = []

    gs = game_state
    gstate = gs.get("game_state", "overworld")
    in_battle = gs.get("in_battle", False)
    parts.append(f"PROGRESS: {_compact_progress_summary(progress_summary, battle=in_battle)}")

    in_battle = gs.get("in_battle", False)

    party_info = ""
    party = gs.get("party", [])
    if party:
        party_strs = [f"{p['name']} L{p['level']} {p['hp']}/{p['max_hp']}" for p in party]
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
        is_trainer = gs.get("is_trainer_battle", False)
        battle_info = (
            f"{'TRAINER' if is_trainer else 'WILD'} BATTLE | "
            f"enemy=L{gs.get('enemy_level', '?')} {gs.get('enemy_species', '?')} hp={gs.get('enemy_hp', '?')}"
        )
        if is_trainer:
            battle_info += " | no flee"

        moves = gs.get("battle_moves", [])
        action_labels = ["Fight", "Bag", "Pokemon", "Run"]
        action_cursor = gs.get("battle_action_cursor", 0)
        move_cursor = gs.get("battle_move_cursor", 0)
        menu_state = gs.get("battle_menu_state", 1)
        if menu_state == 2:
            battle_info += "\nMOVE MENU"
            if moves:
                move_slots = ["(empty)"] * 4
                for m in moves:
                    pp_str = f"{m['pp']}" if m["pp"] > 0 else "0!"
                    move_slots[m["slot"]] = f"{m['name']}[{pp_str}]"
                battle_info += f"\nTL={move_slots[0]} | TR={move_slots[1]}"
                battle_info += f"\nBL={move_slots[2]} | BR={move_slots[3]}"
            battle_info += f"\ncursor={_cursor_slot_name(move_cursor)}"

        elif menu_state == 1:
            action_name = action_labels[action_cursor] if action_cursor < 4 else "?"
            battle_info += "\nACTION MENU"
            battle_info += f"\ncursor={action_name}"
            battle_info += "\nTL=Fight | TR=Bag"
            battle_info += "\nBL=Pokemon | BR=Run"
            if moves:
                move_lines = [f"{m['name']}[{m['pp'] if m['pp'] > 0 else '0!'}]" for m in moves]
                battle_info += f"\nMoves: {', '.join(move_lines)}"

        else:
            battle_info += "\nANIMATION/TEXT"

        parts.append(
            f"=== BATTLE ===\n"
            f"{battle_info}\n"
            f"{party_info}{dialogue_line if menu_state not in (1, 2) else ''}"
        )

    elif gstate == "bag":
        bag = gs.get("bag_items")
        in_battle_bag = gs.get("in_battle", False)
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

            bag_title = "=== BATTLE BAG ===" if in_battle_bag else "=== BAG ==="
            bag_display = f"{bag_title}\nPockets: {' | '.join(tabs)}\n"
            if item_lines:
                bag_display += "\n".join(item_lines)
            else:
                bag_display += "    (empty)"
            if in_battle_bag:
                trainer_note = " Trainer battle: Run is unavailable." if gs.get("is_trainer_battle", False) else ""
                bag_display += "\nBattle bag: Up/Down = choose item. A = use item. B = return to battle menu." + trainer_note
            else:
                bag_display += "\nLeft/Right = switch pocket. Up/Down = scroll. A = use item. B = close bag."

            parts.append(bag_display + f"{party_info}{dialogue_line}")
        else:
            bag_title = "=== BATTLE BAG ===" if in_battle_bag else "=== BAG ==="
            parts.append(
                f"{bag_title}\n"
                f"{'Bag is open during battle. A = use item. B = return to battle menu.' if in_battle_bag else 'Bag is open. Left/Right = switch pocket. Up/Down = scroll. A = use. B = close.'}"
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
    model = SONNET if game_state.get("in_battle", False) else HAIKU

    messages = build_messages(game_state, recent_actions, progress_summary)

    # Add assistant prefill for Haiku — force JSON output
    if "haiku" in model:
        messages.append({"role": "assistant", "content": '{"action": "'})

    print(
        f"  [claude:action] Requesting action for state={gstate} "
        f"map={game_state.get('map_name', '?')} battle={game_state.get('in_battle', False)}"
    )
    started = time.monotonic()
    response = client.messages.create(
        model=model,
        max_tokens=180,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=messages,
    )
    elapsed = time.monotonic() - started
    print(f"  [claude:action] Response received in {elapsed:.2f}s")

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
