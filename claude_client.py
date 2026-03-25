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

OVERWORLD CONTROLS:
- A: interact, advance dialogue. If Dialogue is True, press A first.
- But if pressing A repeatedly is not changing the situation, stop repeating A and try a different action.
- Do NOT move into walls. Check ADJACENT TILES — only move in directions marked "walkable" or "door" or "stairs" or "map edge".
- Prioritize unvisited tiles. Avoid overvisited tiles.
- Do NOT walk back onto any tile you visited in your last 3 moves — keep moving forward.
- INTERACT with nearby objects marked [NOT yet interacted] — walk toward them and press A.
- Indoor exits (1F): usually south. Outdoor exits: walk to map edge.
- Read DIALOGUE TEXT carefully — NPCs give hints about what to do next.

BATTLE CONTROLS:
- You see the battle action menu (Fight/Bag/Pokemon/Run) and your moves with PP.
- Use Left/Right/Up/Down to move the cursor, A to confirm selection.
- To attack: select Fight, then pick a move with PP remaining, press A.
- Press B to go back from move select to action menu.
- You CANNOT exit battle with B. To flee, move cursor to Run and press A.
- Low HP? Consider Run or use a healing item from Bag.

BAG/POKEMON/SUMMARY: Press B to go back.

Respond ONLY with valid JSON:
{"action": "<button>", "reason": "<why, under 40 words>", "display": "<casual 25-word viewer summary>"}"""

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


def select_building_door(map_name: str,
                         objective: str,
                         player_pos: tuple[int, int],
                         candidates: list[dict],
                         rejected: list[tuple[int, int]] | None = None,
                         discoveries: list[dict] | None = None) -> dict | None:
    """Ask Claude to choose the most likely building entrance for the current goal.

    Returns {"x": int, "y": int, "reason": str} or None on failure.
    """
    if not candidates:
        return None

    rejected = rejected or []
    discoveries = discoveries or []
    candidate_lines = []
    for idx, cand in enumerate(candidates, start=1):
        label = cand.get("label", "unknown")
        candidate_lines.append(
            f"{idx}. door at ({cand['x']},{cand['y']}) distance {cand['distance']} steps"
            f" discovered_as={label}"
        )

    rejected_text = ", ".join(f"({x},{y})" for x, y in rejected) if rejected else "none"
    discovery_lines = []
    for discovery in discoveries:
        discovery_lines.append(
            f"- ({discovery['x']},{discovery['y']}) -> {discovery['map_name']}"
        )
    discovery_text = chr(10).join(discovery_lines) if discovery_lines else "- none yet"
    prompt = (
        "You are helping an AI agent play Pokemon FireRed.\n"
        f"Current outdoor map: {map_name}\n"
        f"Current objective: {objective}\n"
        f"Player position: ({player_pos[0]},{player_pos[1]})\n"
        "Choose the most likely correct building entrance based on your knowledge of FireRed town layouts.\n"
        "Use the discovered buildings below as learned facts. Prefer an unknown door when known doors do not match the goal.\n"
        "Discovered building entrances on this map:\n"
        f"{discovery_text}\n"
        f"Rejected doors already proven wrong for this task: {rejected_text}\n"
        "Candidate doors:\n"
        f"{chr(10).join(candidate_lines)}\n\n"
        "Reply ONLY with JSON like "
        '{"x": 12, "y": 8, "reason": "Likely Pokecenter location"}'
    )

    try:
        response = client.messages.create(
            model=HAIKU,
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        return None

    raw_text = response.content[0].text.strip()
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

    if not isinstance(parsed, dict):
        return None
    if "x" not in parsed or "y" not in parsed:
        return None
    try:
        x = int(parsed["x"])
        y = int(parsed["y"])
    except (TypeError, ValueError):
        return None

    reason = str(parsed.get("reason", "")).strip()[:120]
    return {"x": x, "y": y, "reason": reason}


def select_route_exit(map_name: str,
                      objective: str,
                      player_pos: tuple[int, int],
                      candidates: list[dict]) -> dict | None:
    """Ask Claude to choose the most likely route exit for the current travel goal."""
    if not candidates:
        return None

    candidate_lines = []
    for idx, cand in enumerate(candidates, start=1):
        candidate_lines.append(
            f"{idx}. {cand['side']} exit centered at ({cand['x']},{cand['y']}) "
            f"distance {cand['distance']} steps span {cand['span']}"
        )

    prompt = (
        "You are helping an AI agent play Pokemon FireRed.\n"
        f"Current outdoor map: {map_name}\n"
        f"Current objective: {objective}\n"
        f"Player position: ({player_pos[0]},{player_pos[1]})\n"
        "Choose the most likely route/map exit based on the travel goal and your knowledge of FireRed map connections.\n"
        "Distance is only a weak tiebreaker. Do NOT choose an exit mainly because it is closer if another exit better matches the objective.\n"
        "Prefer the exit that best matches the intended destination or story progress, even if it is farther away.\n"
        "Candidate exits:\n"
        f"{chr(10).join(candidate_lines)}\n\n"
        "Reply ONLY with JSON like "
        '{"x": 25, "y": 33, "reason": "South exit leads back toward Pallet Town"}'
    )

    try:
        response = client.messages.create(
            model=HAIKU,
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        return None

    raw_text = response.content[0].text.strip()
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

    if not isinstance(parsed, dict) or "x" not in parsed or "y" not in parsed:
        return None
    try:
        x = int(parsed["x"])
        y = int(parsed["y"])
    except (TypeError, ValueError):
        return None

    reason = str(parsed.get("reason", "")).strip()[:120]
    return {"x": x, "y": y, "reason": reason}


def classify_navigation_intent(map_name: str,
                               current_objective: str,
                               strategy_objective: str,
                               indoor: bool,
                               hp_ratio: float,
                               party_count: int,
                               visible_doors: int,
                               visible_objects: int,
                               visible_summary: str,
                               current_state: str,
                               previous_intent: str | None = None) -> dict | None:
    """Ask Claude to choose the best high-level overworld navigation intent.

    Returns {"intent": str, "reason": str} or None on failure.
    """
    prompt = (
        "Pokemon FireRed AI agent navigation. Choose ONE intent.\n"
        "Intents: go_to_building, go_to_route_exit, talk_to_npc, interact_with_object, leave_building, train, none.\n\n"
        f"Map: {map_name} ({'indoor' if indoor else 'outdoor'}) | HP: {hp_ratio:.0%} | Party: {party_count}\n"
        f"Objective: {current_objective or '(none)'}\n"
        f"Strategy: {strategy_objective or '(none)'}\n"
        f"Doors/exits: {visible_doors} | Objects/NPCs: {visible_objects}\n"
        f"Interactables: {visible_summary or 'none'}\n"
        f"Nav state: {current_state} | Previous: {previous_intent or 'none'}\n\n"
        "Rules: At destination, prefer building/NPC over route_exit. "
        "In Pokecenter/Mart needing service, prefer talk_to_npc. "
        "No Pokemon + nearby triggers = interact_with_object. "
        "Wrong building = leave_building. train = only for grinding.\n\n"
        'Reply ONLY with JSON: {"intent":"...","reason":"..."}'
    )

    try:
        response = client.messages.create(
            model=HAIKU,
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        return None

    raw_text = response.content[0].text.strip()
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

    if not isinstance(parsed, dict):
        return None
    intent = str(parsed.get("intent", "")).strip()
    if intent not in {"go_to_building", "go_to_route_exit", "talk_to_npc", "interact_with_object", "leave_building", "train", "none"}:
        return None
    reason = str(parsed.get("reason", "")).strip()[:120]
    return {"intent": intent, "reason": reason}


def select_npc_target(map_name: str,
                      current_objective: str,
                      strategy_objective: str,
                      player_pos: tuple[int, int],
                      candidates: list[dict]) -> dict | None:
    """Ask Claude to choose the most relevant visible NPC/object target."""
    if not candidates:
        return None

    candidate_lines = []
    for idx, cand in enumerate(candidates, start=1):
        candidate_lines.append(
            f"{idx}. id={cand['local_id']} label={cand['label']} at ({cand['x']},{cand['y']}) "
            f"distance {cand['distance']} talked {cand.get('interaction_count', 0)}x"
        )

    prompt = (
        "You are helping an AI agent play Pokemon FireRed.\n"
        "Choose the SINGLE most relevant visible NPC or interactable for the current local objective.\n"
        f"Current map: {map_name}\n"
        f"Current objective: {current_objective or '(none)'}\n"
        f"Strategy objective: {strategy_objective or '(none)'}\n"
        f"Player position: ({player_pos[0]},{player_pos[1]})\n"
        "Visible candidates:\n"
        f"{chr(10).join(candidate_lines)}\n\n"
        "Rules:\n"
        "- Prefer the NPC most likely to advance the stated objective.\n"
        "- If one NPC has already been talked to many times without progress, prefer a different plausible NPC.\n"
        "- If the objective is to talk to a key story character, choose the candidate most likely to be that character.\n\n"
        'Reply ONLY with JSON like {"local_id": 3, "reason": "Most likely to be Professor Oak."}'
    )

    try:
        response = client.messages.create(
            model=HAIKU,
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        return None

    raw_text = response.content[0].text.strip()
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

    if not isinstance(parsed, dict) or "local_id" not in parsed:
        return None
    try:
        local_id = int(parsed["local_id"])
    except (TypeError, ValueError):
        return None
    reason = str(parsed.get("reason", "")).strip()[:120]
    return {"local_id": local_id, "reason": reason}


def select_interactable_target(map_name: str,
                               current_objective: str,
                               strategy_objective: str,
                               party_count: int,
                               player_pos: tuple[int, int],
                               candidates: list[dict]) -> dict | None:
    """Ask Claude to choose the most relevant non-NPC interactable target."""
    if not candidates:
        return None

    candidate_lines = []
    for idx, cand in enumerate(candidates, start=1):
        candidate_lines.append(
            f"{idx}. id={cand['id']} label={cand['label']} at ({cand['x']},{cand['y']}) "
            f"distance {cand['distance']} interacted {cand.get('interaction_count', 0)}x"
        )

    prompt = (
        "You are helping an AI agent play Pokemon FireRed.\n"
        "Choose the SINGLE most relevant visible non-NPC interactable for the current objective.\n"
        f"Current map: {map_name}\n"
        f"Current objective: {current_objective or '(none)'}\n"
        f"Strategy objective: {strategy_objective or '(none)'}\n"
        f"Party count: {party_count}\n"
        f"Player position: ({player_pos[0]},{player_pos[1]})\n"
        "Visible candidates:\n"
        f"{chr(10).join(candidate_lines)}\n\n"
        "Rules:\n"
        "- Prefer triggers, items, or other interactables most likely to advance the current objective.\n"
        "- If one candidate has already been interacted with many times without progress, prefer a different plausible candidate.\n"
        "- Walk-on triggers are often important story/event tiles.\n\n"
        "- If the party is empty, prefer the trigger or item choice most likely to start the required starter/progression event.\n\n"
        "- If the party is empty and a visible Pokeball/item object is present, prefer the actual object over nearby trigger tiles.\n\n"
        'Reply ONLY with JSON like {"id":"obj:7","reason":"This trigger likely starts the required event."}'
    )

    try:
        response = client.messages.create(
            model=HAIKU,
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        return None

    raw_text = response.content[0].text.strip()
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

    if not isinstance(parsed, dict) or "id" not in parsed:
        return None
    reason = str(parsed.get("reason", "")).strip()[:120]
    return {"id": str(parsed["id"]).strip(), "reason": reason}


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
                   stuck_warning: str | None = None,
                   exploration_summary: str | None = None) -> list[dict]:
    """Build the messages array for the API call (no screenshot)."""
    # Section 1: Recent actions (strategy context only, no coordinates)
    parts = []
    gstate_early = game_state.get("game_state", "overworld")
    if recent_actions and gstate_early == "overworld":
        history = "\n".join(
            f"  {a['action']} — {_strip_coordinates(a['reason'])}"
            for a in recent_actions[-3:]
        )
        parts.append(f"RECENT ACTIONS (last {len(recent_actions[-3:])}):\n{history}")

    parts.append(f"CURRENT PROGRESS: {progress_summary}")

    if exploration_summary:
        parts.append(exploration_summary)

    if stuck_warning:
        parts.append(f"WARNING: {stuck_warning}")

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
        battle_info = f"Enemy: Lv{gs.get('enemy_level', '?')} HP:{gs.get('enemy_hp', '?')}"

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

    elif gstate == "shop_buy":
        bag = gs.get("bag_items")
        money = gs.get("money", "?")
        items_display = ""
        if bag:
            pocket_key = "Items"
            items = bag["pockets"].get(pocket_key, [])
            if items:
                item_lines = [f"    {it['name']} x{it['quantity']}" for it in items[:8]]
                items_display = "\nYour items: " + ", ".join(f"{it['name']} x{it['quantity']}" for it in items[:8])
        parts.append(
            f"=== POKEMART BUY MENU ===\n"
            f"Money: ${money}{items_display}\n"
            f"You're in the buy menu. Up/Down = select item. A = buy. B = cancel/exit."
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
               recent_actions: list[dict], progress_summary: str,
               stuck_warning: str | None = None,
               force_sonnet: bool = False,
               exploration_summary: str | None = None) -> tuple[dict, dict]:
    """Call Claude and return (parsed_action, usage_info).

    Returns:
        parsed_action: {"action": str, "reason": str}
        usage_info: {"model": str, "input_tokens": int, "output_tokens": int}
    """
    model = SONNET if force_sonnet else HAIKU

    messages = build_messages(game_state, recent_actions,
                              progress_summary, stuck_warning,
                              exploration_summary)

    # Add assistant prefill for Haiku — force JSON output
    if "haiku" in model:
        messages.append({"role": "assistant", "content": '{"action": "'})

    response = client.messages.create(
        model=model,
        max_tokens=150,
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
