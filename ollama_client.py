"""Ollama API wrapper — drop-in replacement for claude_client.py using a local LLM."""

import base64
import io
import json
import random
import re
import requests
from PIL import Image

from game_knowledge import get_relevant_knowledge, _find_navigation_knowledge

OLLAMA_BASE = "http://localhost:11434"
MODEL = "qwen3:8b"

# Aliases so agent.py imports still work
HAIKU = MODEL
SONNET = MODEL

SYSTEM_PROMPT = """You are an AI playing Pokemon FireRed (GBA). Your goal: earn all 8 badges and defeat the Elite Four.

You receive STRUCTURED DATA about the game state. The header tells you what mode you're in.

OVERWORLD CONTROLS:
- A: interact, advance dialogue. If Dialogue is True, press A first.
- CRITICAL: A only works when you are 1 tile away from an NPC/object. If an NPC is multiple tiles away, you MUST walk toward them first using Up/Down/Left/Right. Do NOT press A when far away.
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

IMPORTANT: You must respond with ONLY valid JSON. No thinking, no explanation, no markdown.
Format: {"action": "<button>", "reason": "<why, under 40 words>", "display": "<casual 25-word viewer summary>"}
Valid buttons: A, B, Up, Down, Left, Right, Start, Select"""

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


def _ollama_chat(messages: list[dict], model: str = MODEL,
                 max_tokens: int = 400, system: str | None = None) -> str:
    """Send a chat request to Ollama and return the response text."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,  # Disable thinking mode for speed
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.7,
        },
    }
    if system:
        # Prepend system message
        payload["messages"] = [{"role": "system", "content": system}] + payload["messages"]

    resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    content = data["message"].get("content", "").strip()
    # Fallback: if content is empty but thinking has content, use that
    if not content:
        content = data["message"].get("thinking", "").strip()
    return content


def _parse_json_response(raw_text: str) -> dict | None:
    """Try to extract a JSON object from LLM output, handling thinking tags and markdown."""
    # Strip <think>...</think> blocks (Qwen3 thinking mode)
    cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()

    # Strip markdown code fences
    cleaned = re.sub(r'```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'```\s*$', '', cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON substring
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
    return None


def generate_pokemon_nickname(species_name: str = "", theme: str = "") -> str:
    """Generate a short Pokemon-related nickname."""
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
        raw = _ollama_chat([{"role": "user", "content": prompt}], max_tokens=20)
        # Strip thinking tags
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        cleaned = re.sub(r"[^A-Za-z]", "", raw.splitlines()[0]).upper()
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
    """Choose the most likely building entrance for the current goal."""
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
    knowledge = _find_navigation_knowledge(map_name)
    knowledge_block = f"\n{knowledge}\n" if knowledge else ""
    prompt = (
        "You are helping an AI agent play Pokemon FireRed.\n"
        f"Current outdoor map: {map_name}\n"
        f"Current objective: {objective}\n"
        f"Player position: ({player_pos[0]},{player_pos[1]})\n"
        f"{knowledge_block}"
        "Choose the most likely correct building entrance based on the game knowledge above and the discovered buildings below.\n"
        "Use the discovered buildings below as learned facts. Prefer an unknown door when known doors do not match the goal.\n"
        "Discovered building entrances on this map:\n"
        f"{discovery_text}\n"
        f"Rejected doors already proven wrong for this task: {rejected_text}\n"
        "Candidate doors:\n"
        f"{chr(10).join(candidate_lines)}\n\n"
        "Reply ONLY with JSON like "
        '{"x": 12, "y": 8, "reason": "Likely Pokecenter location"}\n'
        "No other text. Just the JSON."
    )

    try:
        raw_text = _ollama_chat([{"role": "user", "content": prompt}], max_tokens=120)
    except Exception:
        return None

    parsed = _parse_json_response(raw_text)
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
    """Choose the most likely route exit for the current travel goal."""
    if not candidates:
        return None

    candidate_lines = []
    for idx, cand in enumerate(candidates, start=1):
        candidate_lines.append(
            f"{idx}. {cand['side']} exit centered at ({cand['x']},{cand['y']}) "
            f"distance {cand['distance']} steps span {cand['span']}"
        )

    knowledge = _find_navigation_knowledge(map_name)
    knowledge_block = f"\n{knowledge}\n" if knowledge else ""
    prompt = (
        "You are helping an AI agent play Pokemon FireRed.\n"
        f"Current outdoor map: {map_name}\n"
        f"Current objective: {objective}\n"
        f"Player position: ({player_pos[0]},{player_pos[1]})\n"
        f"{knowledge_block}"
        "Choose the most likely route/map exit based on the game knowledge above and the travel goal.\n"
        "Distance is only a weak tiebreaker. Do NOT choose an exit mainly because it is closer if another exit better matches the objective.\n"
        "Prefer the exit that best matches the intended destination or story progress, even if it is farther away.\n"
        "Candidate exits:\n"
        f"{chr(10).join(candidate_lines)}\n\n"
        "Reply ONLY with JSON like "
        '{"x": 25, "y": 33, "reason": "South exit leads back toward Pallet Town"}\n'
        "No other text. Just the JSON."
    )

    try:
        raw_text = _ollama_chat([{"role": "user", "content": prompt}], max_tokens=120)
    except Exception:
        return None

    parsed = _parse_json_response(raw_text)
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
    """Choose the best high-level overworld navigation intent."""
    prompt = (
        "You are helping an AI agent play Pokemon FireRed.\n"
        "Choose the SINGLE best immediate overworld navigation intent.\n"
        "Allowed intents: go_to_building, go_to_route_exit, talk_to_npc, interact_with_object, leave_building, train, none.\n\n"
        f"Current map: {map_name}\n"
        f"Map type: {'indoor' if indoor else 'outdoor'}\n"
        f"Current objective: {current_objective or '(none)'}\n"
        f"Strategy objective: {strategy_objective or '(none)'}\n"
        f"Lead HP ratio: {hp_ratio:.2f}\n"
        f"Party count: {party_count}\n"
        f"Visible doors/exits on map: {visible_doors}\n"
        f"Visible objects/NPCs on map: {visible_objects}\n"
        f"Visible interactables summary: {visible_summary or 'none'}\n"
        f"Current nav state: {current_state}\n"
        f"Previous nav intent: {previous_intent or 'none'}\n\n"
        "Rules:\n"
        "- If the player has reached the destination area and now needs a local building or NPC, prefer go_to_building or talk_to_npc over go_to_route_exit.\n"
        "- If the player has no Pokemon yet and there are nearby non-NPC interactables or trigger tiles, prefer interact_with_object over leaving.\n"
        "- If the player is inside a Pokemon Center and still needs service from the counter/nurse, prefer talk_to_npc over leaving or wandering.\n"
        "- If the player is inside a Mart and still needs to buy, sell, or receive a story item from the clerk, prefer talk_to_npc over leaving or wandering.\n"
        "- Use interact_with_object when the best next step is to step onto or inspect a nearby trigger, item, sign, or other non-NPC interactable.\n"
        "- Use leave_building when inside the wrong building or when the objective is to go outside.\n"
        "- Use go_to_route_exit only when the best next step is leaving the current map.\n"
        "- Use talk_to_npc when already in the correct building or standing at the relevant NPC/counter.\n"
        "- Use train only when the objective is clearly about training or wild encounters.\n"
        "- Use none if no structured intent clearly applies.\n\n"
        'Reply ONLY with JSON like {"intent":"go_to_building","reason":"Need to find Oak\'s Lab in the current town."}\n'
        "No other text. Just the JSON."
    )

    try:
        raw_text = _ollama_chat([{"role": "user", "content": prompt}], max_tokens=120)
    except Exception:
        return None

    parsed = _parse_json_response(raw_text)
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
    """Choose the most relevant visible NPC/object target."""
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
        'Reply ONLY with JSON like {"local_id": 3, "reason": "Most likely to be Professor Oak."}\n'
        "No other text. Just the JSON."
    )

    try:
        raw_text = _ollama_chat([{"role": "user", "content": prompt}], max_tokens=120)
    except Exception:
        return None

    parsed = _parse_json_response(raw_text)
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
    """Choose the most relevant non-NPC interactable target."""
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
        "- Walk-on triggers are often important story/event tiles.\n"
        "- If the party is empty, prefer the trigger or item choice most likely to start the required starter/progression event.\n"
        "- If the party is empty and a visible Pokeball/item object is present, prefer the actual object over nearby trigger tiles.\n\n"
        'Reply ONLY with JSON like {"id":"obj:7","reason":"This trigger likely starts the required event."}\n'
        "No other text. Just the JSON."
    )

    try:
        raw_text = _ollama_chat([{"role": "user", "content": prompt}], max_tokens=120)
    except Exception:
        return None

    parsed = _parse_json_response(raw_text)
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
    text = re.sub(r'\(?\d+\s*,\s*\d+\)?', '', text)
    text = re.sub(r'[xy]\s*[=:]\s*\d+', '', text)
    text = re.sub(r'player_[xy]\s*[=:]\s*\d+', '', text)
    text = re.sub(r'PLAYERS?_HOUSE_[12]F', '', text)
    text = re.sub(r'\brows?\s*\d+[-–]?\d*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcol(?:umn)?s?\s*\d+[-–]?\d*', '', text, flags=re.IGNORECASE)
    text = re.sub(r"['\"\(][.#@*]['\"\)]", '', text)
    text = re.sub(r'\bminimap\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\btiles?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bunexplored\b', 'new', text, flags=re.IGNORECASE)
    text = re.sub(r'\bvisit count\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'  +', ' ', text).strip()
    return text


def build_messages(game_state: dict,
                   recent_actions: list[dict], progress_summary: str,
                   stuck_warning: str | None = None,
                   exploration_summary: str | None = None) -> list[dict]:
    """Build the messages array for the API call (no screenshot)."""
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

    gs = game_state
    gstate = gs.get("game_state", "overworld")

    party_info = ""
    party = gs.get("party", [])
    if party:
        party_strs = [f"{p['name']} Lv{p['level']} {p['hp']}/{p['max_hp']}HP" for p in party]
        party_info = f"\nParty: {', '.join(party_strs)}"

    dialogue_text = gs.get("dialogue_text", "")
    dialogue_suspect = gs.get("dialogue_text_suspect", False)
    dialogue_line = ""
    if dialogue_text:
        dialogue_line = f"\nDIALOGUE TEXT: \"{dialogue_text}\""
    elif dialogue_suspect:
        dialogue_line = "\nDIALOGUE TEXT: unavailable or unreliable; do not assume repeated A will help."

    if gstate == "battle":
        enemy_name = gs.get("enemy_species", "?")
        battle_info = f"Enemy: Lv{gs.get('enemy_level', '?')} HP:{gs.get('enemy_hp', '?')}"

        moves = gs.get("battle_moves", [])
        action_labels = ["Fight", "Bag", "Pokemon", "Run"]
        action_cursor = gs.get("battle_action_cursor", 0)
        move_cursor = gs.get("battle_move_cursor", 0)
        menu_state = gs.get("battle_menu_state", 1)

        if menu_state == 2:
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
                battle_info += f"\n  Top-Left: {move_slots[0]}  |  Top-Right: {move_slots[1]}"
                battle_info += f"\n  Bot-Left: {move_slots[2]}  |  Bot-Right: {move_slots[3]}"
            pos_names = {0: "Top-Left", 1: "Top-Right", 2: "Bot-Left", 3: "Bot-Right"}
            battle_info += f"\nCursor is on: {cursor_move or '???'} ({pos_names.get(move_cursor, '?')})."
            # Add explicit navigation hints
            nav_hints = []
            cur_x, cur_y = move_cursor % 2, move_cursor // 2
            for m in moves:
                if m.get("pp", 0) > 0:
                    sx, sy = m["slot"] % 2, m["slot"] // 2
                    steps = []
                    if sx > cur_x:
                        steps.append("Right")
                    elif sx < cur_x:
                        steps.append("Left")
                    if sy > cur_y:
                        steps.append("Down")
                    elif sy < cur_y:
                        steps.append("Up")
                    if steps:
                        nav_hints.append(f"To reach {m['name']}: press {' then '.join(steps)}, then A")
                    else:
                        nav_hints.append(f"{m['name']} is selected — press A NOW to use it")
            if nav_hints:
                battle_info += "\n" + "\n".join(nav_hints)
            battle_info += "\nB = back to action menu."

        elif menu_state == 1:
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
            battle_info += "\nBattle animation in progress. Press A to advance text/animation."

        battle_knowledge = get_relevant_knowledge(
            gs.get("badges", 0), gs.get("map_name", ""), context="battle"
        )
        battle_knowledge_block = f"\n{battle_knowledge}" if battle_knowledge else ""
        parts.append(
            f"=== BATTLE ===\n"
            f"{battle_info}\n"
            f"{party_info}{dialogue_line}{battle_knowledge_block}"
        )

    elif gstate == "bag":
        bag = gs.get("bag_items")
        if bag:
            pocket_keys = ["Items", "KeyItems", "PokeBalls", "TMs", "Berries"]
            display_names = ["Items", "Key Items", "Poke Balls", "TMs & HMs", "Berries"]
            current_idx = bag["current_pocket"]
            tabs = []
            for i, dn in enumerate(display_names):
                tabs.append(f"[{dn}]" if i == current_idx else dn)
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
    """Call local LLM and return (parsed_action, usage_info)."""
    messages = build_messages(game_state, recent_actions,
                              progress_summary, stuck_warning,
                              exploration_summary)

    raw_text = _ollama_chat(messages, model=MODEL, max_tokens=400,
                            system=SYSTEM_PROMPT)

    # Parse JSON response, with fallback
    parsed = _parse_json_response(raw_text)

    VALID_BUTTONS = {"A", "B", "Up", "Down", "Left", "Right", "Start", "Select"}
    BUTTON_NORMALIZE = {b.lower(): b for b in VALID_BUTTONS}

    if parsed and "action" in parsed:
        raw_action = parsed["action"]
        if isinstance(raw_action, str):
            first_word = raw_action.strip().split()[0] if raw_action.strip() else ""
            normalized = BUTTON_NORMALIZE.get(first_word.lower())
        else:
            normalized = None
        if normalized:
            parsed["action"] = normalized
        else:
            parsed = None

    if not parsed or "action" not in parsed:
        # Strip thinking tags before fallback search
        clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        for btn in ["Start", "Select", "Up", "Down", "Left", "Right"]:
            if re.search(r'\b' + btn + r'\b', clean_text):
                parsed = {"action": btn, "reason": clean_text[:80]}
                break
        if not parsed:
            for btn in ["A", "B"]:
                if re.search(r'(?<![a-zA-Z])' + btn + r'(?![a-zA-Z])', clean_text):
                    parsed = {"action": btn, "reason": clean_text[:80]}
                    break
        if not parsed:
            parsed = {"action": "A", "reason": "Failed to parse, defaulting to A"}

    # Local models have no token billing, but keep the interface compatible
    usage_info = {
        "model": MODEL,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read": 0,
        "cache_creation": 0,
    }

    return parsed, usage_info
