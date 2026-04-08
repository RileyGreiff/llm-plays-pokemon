"""BizHawk file-based bridge — Python communicates via files instead of sockets.

No more blocking socket calls. Python writes command.txt, Lua reads it,
processes the command, writes response.txt, Python reads it.
"""

import json
import base64
import io
import time
import os
import re
import numpy as np
from PIL import Image

from memory import MAP_NAMES, POKEMON_NAMES, MOVE_NAMES, ITEM_NAMES, BAG_POCKET_NAMES

BRIDGE_DIR = "bridge"
CMD_FILE = os.path.join(BRIDGE_DIR, "command.txt")
RESP_FILE = os.path.join(BRIDGE_DIR, "response.txt")
READY_FILE = os.path.join(BRIDGE_DIR, "ready.txt")


def _send_command(command: str, timeout: float = 10.0) -> str:
    """Send a command to BizHawk via file and wait for response."""
    # Clean up any stale response
    try:
        os.remove(RESP_FILE)
    except FileNotFoundError:
        pass

    # Write command file
    with open(CMD_FILE, "w") as f:
        f.write(command)

    # Wait for response file
    start = time.time()
    while time.time() - start < timeout:
        try:
            with open(RESP_FILE, "r") as f:
                result = f.read()
            if result:
                os.remove(RESP_FILE)
                return result
        except FileNotFoundError:
            pass
        time.sleep(0.05)  # 50ms poll — fast response, no lag

    raise TimeoutError(f"No response from BizHawk after {timeout}s for: {command[:30]}")


_collision_cache: dict[int, tuple[int, int, list[str]]] = {}


def get_collision_grid(map_id: int) -> tuple[int, int, list[str]] | None:
    """Fetch collision grid for current map. Returns (width, height, rows) or None.

    Each row is a string of:
    - '0' walkable
    - '1' blocked
    - 'D' door/entrance
    - 'S' stairs
    - 'G' tall grass
    Results are cached per map_id.
    """
    if map_id in _collision_cache:
        return _collision_cache[map_id]

    try:
        result = _send_command("COLLISION")
    except (ConnectionError, TimeoutError, OSError):
        return None

    if result.startswith("ERROR"):
        print(f"  [collision] {result}")
        return None

    try:
        parts = result.split("|", 1)
        if len(parts) != 2:
            print(f"  [collision] Bad response format (no | separator)")
            return None
        header, grid_data = parts
        dims = header.split(",")
        if len(dims) != 2:
            print(f"  [collision] Bad header: {header[:40]}")
            return None
        w, h = int(dims[0]), int(dims[1])
        rows = [grid_data[i * w:(i + 1) * w] for i in range(h)]
        _collision_cache[map_id] = (w, h, rows)
        print(f"  [collision] Loaded {w}x{h} grid for map {map_id}")
        return (w, h, rows)
    except (ValueError, IndexError) as e:
        print(f"  [collision] Parse error: {e}")
        return None


# Common graphicsId -> label mapping for FireRed
_GFX_LABELS = {
    0x59: "Pokeball item",  # OBJ_EVENT_GFX_ITEM_BALL
    0x5C: "Pokeball",       # Starter Pokeballs on table
    0x5E: "Pokeball",       # Another Pokeball variant
}


def _parse_objects_raw(raw: str) -> tuple[list[dict], int | None] | tuple[None, None]:
    """Parse object data from a raw string (shared by get_objects and inline parsing)."""
    if not raw:
        return None, None

    objects = []
    player_facing = None
    for entry in raw.split("|"):
        if not entry:
            continue
        try:
            parts = entry.split(",")
            if parts[0] == "P":
                player_facing = int(parts[1])
                continue
            local_id, gfx_id, x, y, facing = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            MAP_OFFSET = 7
            x -= MAP_OFFSET
            y -= MAP_OFFSET
            label = _GFX_LABELS.get(gfx_id, "NPC")
            objects.append({
                "local_id": local_id,
                "gfx_id": gfx_id,
                "x": x, "y": y,
                "facing": facing,
                "label": label,
            })
        except (ValueError, IndexError):
            continue

    return (objects if objects else None), player_facing


def get_objects(game_state: dict | None = None) -> tuple[list[dict], int | None] | tuple[None, None]:
    """Get active Object Events (NPCs, items).

    If game_state is provided and contains inline object data from GAMESTATE_FR,
    uses that instead of sending a separate command (avoids extra lag).

    Returns:
        (objects_list, player_facing) or (None, None).
        player_facing: 1=down, 2=up, 3=left, 4=right
    """
    if game_state and game_state.get("_objects_raw"):
        return _parse_objects_raw(game_state["_objects_raw"])

    try:
        result = _send_command("OBJECTS")
    except (ConnectionError, TimeoutError, OSError):
        return None, None

    if not result or result.startswith("ERROR"):
        return None, None

    return _parse_objects_raw(result)


_bg_events_cache: dict[int, list[dict]] = {}
_map_connections_cache: dict[int, list[dict]] = {}
_warp_events_cache: dict[int, list[dict]] = {}


def get_bg_events(map_id: int) -> list[dict]:
    """Fetch BG events and coord events for the current map. Cached per map_id."""
    if map_id in _bg_events_cache:
        return _bg_events_cache[map_id]

    try:
        result = _send_command("BG_EVENTS")
    except (ConnectionError, TimeoutError, OSError):
        return []

    if not result or result.startswith("ERROR"):
        return []

    events = []
    for entry in result.split("|"):
        if not entry:
            continue
        try:
            parts = entry.split(",")
            etype = parts[0]
            x, y = int(parts[1]), int(parts[2])
            kind = int(parts[3])
            label = parts[4]
            events.append({
                "type": "bg" if etype == "B" else "coord",
                "x": x, "y": y,
                "kind": kind,
                "label": label,
            })
        except (ValueError, IndexError):
            continue

    _bg_events_cache[map_id] = events
    print(f"  [bg_events] Loaded {len(events)} events for map {map_id}")
    return events


def get_map_connections(map_id: int) -> list[dict]:
    """Fetch map connections for the current map. Cached per map_id."""
    if map_id in _map_connections_cache:
        return _map_connections_cache[map_id]

    try:
        result = _send_command("MAP_CONNECTIONS")
    except (ConnectionError, TimeoutError, OSError):
        return []

    if result.startswith("ERROR"):
        return []
    if not result:
        _map_connections_cache[map_id] = []
        return []

    connections = []
    for entry in result.split("|"):
        if not entry:
            continue
        try:
            direction, offset_str, group_str, num_str = entry.split(",")
            map_group = int(group_str)
            map_num = int(num_str)
            target_name = MAP_NAMES.get((map_group, map_num), f"MAP_{map_group}_{map_num}")
            connections.append({
                "direction": direction,
                "offset": int(offset_str),
                "map_group": map_group,
                "map_num": map_num,
                "map_name": target_name,
            })
        except (ValueError, IndexError):
            continue

    _map_connections_cache[map_id] = connections
    print(f"  [connections] Loaded {len(connections)} connections for map {map_id}")
    return connections


def get_warp_events(map_id: int) -> list[dict]:
    """Fetch warp events for the current map. Cached per map_id."""
    if map_id in _warp_events_cache:
        return _warp_events_cache[map_id]

    try:
        result = _send_command("WARP_EVENTS")
    except (ConnectionError, TimeoutError, OSError):
        return []

    if result.startswith("ERROR"):
        return []
    if not result:
        _warp_events_cache[map_id] = []
        return []

    warps = []
    for entry in result.split("|"):
        if not entry:
            continue
        try:
            x_str, y_str, warp_id_str, group_str, num_str = entry.split(",")
            map_group = int(group_str)
            map_num = int(num_str)
            target_name = MAP_NAMES.get((map_group, map_num), f"MAP_{map_group}_{map_num}")
            warps.append({
                "x": int(x_str),
                "y": int(y_str),
                "destination_warp_id": int(warp_id_str),
                "map_group": map_group,
                "map_num": map_num,
                "destination_map_id": map_group * 256 + map_num,
                "destination_map": target_name,
            })
        except (ValueError, IndexError):
            continue

    _warp_events_cache[map_id] = warps
    print(f"  [warp_events] Loaded {len(warps)} warps for map {map_id}")
    return warps


def debug_objects():
    """Send DEBUG_OBJECTS command to dump all object slots to Lua console."""
    try:
        _send_command("DEBUG_OBJECTS")
        print("  [debug] DEBUG_OBJECTS sent — check BizHawk Lua console")
    except (ConnectionError, TimeoutError, OSError):
        print("  [debug] Failed to send DEBUG_OBJECTS")


def screenshot() -> Image.Image:
    """Capture the current emulator screen and return as a PIL Image."""
    raw = _send_command("SCREENSHOT")
    img_bytes = base64.b64decode(raw)
    return Image.open(io.BytesIO(img_bytes))


def press_button(button: str, frames: int = 16) -> None:
    """Press a button for the given number of frames."""
    valid = {"A", "B", "Up", "Down", "Left", "Right", "Start", "Select"}
    if button not in valid:
        raise ValueError(f"Invalid button '{button}'. Must be one of: {valid}")
    _send_command(f"PRESS {button} {frames}")


def read_memory(address: int, size: int = 1) -> int:
    """Read a value from a memory address."""
    result = _send_command(f"READ {address} {size}")
    try:
        return int(result)
    except ValueError:
        raise ValueError(f"read_memory(0x{address:04X}) failed, got: {result}")


def _parse_bag_data(raw: str) -> dict:
    """Parse bag data from Lua response.

    Format: pocket,cursor,scroll;Items=id:qty,id:qty;KeyItems=...;...
    """
    sections = raw.split(";")
    ui_parts = sections[0].split(",")
    pocket_idx = int(ui_parts[0])
    cursor_pos = int(ui_parts[1])
    scroll_pos = int(ui_parts[2])

    pockets = {}
    for section in sections[1:]:
        if "=" not in section:
            continue
        pocket_name, items_str = section.split("=", 1)
        items = []
        if items_str:
            for item_entry in items_str.split(","):
                if ":" not in item_entry:
                    continue
                item_id, qty = item_entry.split(":")
                item_id = int(item_id)
                qty = int(qty)
                name = ITEM_NAMES.get(item_id, f"Item#{item_id}")
                items.append({"id": item_id, "name": name, "quantity": qty})
        pockets[pocket_name] = items

    return {
        "current_pocket": pocket_idx,
        "pocket_name": BAG_POCKET_NAMES.get(pocket_idx, f"Pocket {pocket_idx}"),
        "cursor": cursor_pos,
        "scroll": scroll_pos,
        "pockets": pockets,
    }


def read_game_state() -> dict:
    """Read FireRed game state via GAMESTATE_FR command."""
    result = _send_command("GAMESTATE_FR")

    # Parse: "val1,val2,...|gamestate|dialoguetext|OBJ:objdata"
    parts = result.split("|", 2)
    if len(parts) < 2:
        raise ValueError(f"GAMESTATE response missing separator: {result[:80]}")
    mem_part = parts[0]
    game_state_str = parts[1] if len(parts) >= 2 else "unknown"
    rest = parts[2] if len(parts) >= 3 else ""

    # Split dialogue text from object data (separated by |OBJ:) and bag data (|BAG:)
    obj_raw = None
    bag_raw = None
    if "|OBJ:" in rest:
        text_part, obj_raw = rest.split("|OBJ:", 1)
        # Bag data comes after OBJ data
        if obj_raw and "|BAG:" in obj_raw:
            obj_raw, bag_raw = obj_raw.split("|BAG:", 1)
    else:
        text_part = rest

    print(f"  [debug] GAMESTATE mem: {mem_part[:120]}")

    try:
        values = [int(v) for v in mem_part.split(",")]
    except ValueError:
        raise ValueError(f"GAMESTATE bad memory values: {mem_part[:80]}")

    if len(values) < 71:
        raise ValueError(f"GAMESTATE_FR count: got {len(values)}, expected at least 71")

    _ACTION_LABELS = ["Fight", "Bag", "Pokemon", "Run"]

    state = {
        "player_x": values[0],
        "player_y": values[1],
        "map_bank": values[2],
        "map_num": values[3],
        "in_battle": values[4] > 0,
        "is_trainer_battle": values[4] == 2,
        "game_state": game_state_str,
        "party_count": values[5],
        "badges": values[6],
        "in_dialogue": values[7] > 0,
        "battle_action_cursor": values[8],
        "battle_move_cursor": values[9],
        "battle_menu_state": values[10] if len(values) > 10 else 0,
    }

    map_key = (state["map_bank"], state["map_num"])
    state["map_id"] = state["map_bank"] * 256 + state["map_num"]
    state["map_name"] = MAP_NAMES.get(map_key, f"MAP_{state['map_bank']}_{state['map_num']}")

    PARTY_START = 11  # after px,py,mapbank,mapnum,battlers,partycount,badges,dialogue,actionCursor,moveCursor,battleMenuState
    PARTY_SLOT_SIZE = 8  # species, level, hp, maxhp, move1, move2, move3, move4
    party = []
    for i in range(min(state["party_count"], 6)):
        idx = PARTY_START + i * PARTY_SLOT_SIZE
        species_id = values[idx]
        level = values[idx + 1]
        hp = values[idx + 2]
        max_hp = values[idx + 3]
        name = POKEMON_NAMES.get(species_id, f"Unknown({species_id})")
        move_ids = [values[idx + 4 + m] for m in range(4)]
        moves = [MOVE_NAMES.get(mid, f"Move#{mid}") for mid in move_ids if mid > 0]
        party.append({"name": name, "level": level, "hp": hp, "max_hp": max_hp, "moves": moves})
    state["party"] = party

    if party:
        state["player_hp"] = party[0]["hp"]
        state["player_level"] = party[0]["level"]
    else:
        state["player_hp"] = 0
        state["player_level"] = 0

    def _value_or_default(idx: int, default: int = 0) -> int:
        return values[idx] if idx < len(values) else default

    inventory_flags_idx = PARTY_START + 6 * PARTY_SLOT_SIZE  # 11 + 48 = 59
    state["has_pokedex"] = _value_or_default(inventory_flags_idx) > 0
    state["has_oaks_parcel"] = _value_or_default(inventory_flags_idx + 1) > 0
    state["has_ss_ticket"] = _value_or_default(inventory_flags_idx + 2) > 0
    state["has_silph_scope"] = _value_or_default(inventory_flags_idx + 3) > 0
    state["has_poke_flute"] = _value_or_default(inventory_flags_idx + 4) > 0
    state["has_secret_key"] = _value_or_default(inventory_flags_idx + 5) > 0
    state["has_card_key"] = _value_or_default(inventory_flags_idx + 6) > 0
    state["has_lift_key"] = _value_or_default(inventory_flags_idx + 7) > 0
    state["has_tea"] = _value_or_default(inventory_flags_idx + 8) > 0
    state["has_bicycle"] = _value_or_default(inventory_flags_idx + 9) > 0
    state["has_bike_voucher"] = _value_or_default(inventory_flags_idx + 10) > 0
    state["has_gold_teeth"] = _value_or_default(inventory_flags_idx + 11) > 0
    state["has_tri_pass"] = _value_or_default(inventory_flags_idx + 12) > 0
    state["has_rainbow_pass"] = _value_or_default(inventory_flags_idx + 13) > 0
    hm_flags_idx = inventory_flags_idx + 14
    state["has_hm01_cut"] = _value_or_default(hm_flags_idx) > 0
    state["has_hm02_fly"] = _value_or_default(hm_flags_idx + 1) > 0
    state["has_hm03_surf"] = _value_or_default(hm_flags_idx + 2) > 0
    state["has_hm04_strength"] = _value_or_default(hm_flags_idx + 3) > 0
    state["has_hm05_flash"] = _value_or_default(hm_flags_idx + 4) > 0
    state["has_hm06_rock_smash"] = _value_or_default(hm_flags_idx + 5) > 0
    state["has_hm07_waterfall"] = _value_or_default(hm_flags_idx + 6) > 0
    state["has_national_dex"] = _value_or_default(hm_flags_idx + 7) > 0
    if state["has_pokedex"]:
        # Once the player has the Pokedex, the Oak's Parcel step is necessarily complete.
        state["has_oaks_parcel"] = True
    state["owned_hms"] = [
        name for name, owned in [
            ("Cut", state["has_hm01_cut"]),
            ("Fly", state["has_hm02_fly"]),
            ("Surf", state["has_hm03_surf"]),
            ("Strength", state["has_hm04_strength"]),
            ("Flash", state["has_hm05_flash"]),
            ("Rock Smash", state["has_hm06_rock_smash"]),
            ("Waterfall", state["has_hm07_waterfall"]),
        ] if owned
    ]
    state["owned_key_progress_items"] = [
        name for name, owned in [
            ("Pokedex", state["has_pokedex"]),
            ("Oaks Parcel", state["has_oaks_parcel"]),
            ("S.S. Ticket", state["has_ss_ticket"]),
            ("Silph Scope", state["has_silph_scope"]),
            ("Poke Flute", state["has_poke_flute"]),
            ("Secret Key", state["has_secret_key"]),
            ("Card Key", state["has_card_key"]),
            ("Lift Key", state["has_lift_key"]),
            ("Tea", state["has_tea"]),
            ("Bicycle", state["has_bicycle"]),
            ("Bike Voucher", state["has_bike_voucher"]),
            ("Gold Teeth", state["has_gold_teeth"]),
            ("Tri-Pass", state["has_tri_pass"]),
            ("Rainbow Pass", state["has_rainbow_pass"]),
        ] if owned
    ]

    extended_payload_present = len(values) >= hm_flags_idx + 8
    enemy_idx = hm_flags_idx + 8 if extended_payload_present else PARTY_START + 6 * PARTY_SLOT_SIZE
    state["enemy_species"] = _value_or_default(enemy_idx)
    state["enemy_hp"] = _value_or_default(enemy_idx + 2)
    state["enemy_level"] = _value_or_default(enemy_idx + 1)

    moves_idx = enemy_idx + 4
    battle_moves = []
    for m in range(4):
        move_id = values[moves_idx + m]
        pp = values[moves_idx + 4 + m]
        if move_id > 0:
            move_name = MOVE_NAMES.get(move_id, f"Move#{move_id}")
            battle_moves.append({"name": move_name, "pp": pp, "slot": m})
    state["battle_moves"] = battle_moves

    cb2_idx = moves_idx + 8
    state["cb2_raw"] = _value_or_default(cb2_idx)

    state["dialogue_text"] = text_part.strip() if text_part else ""
    state["dialogue_text_suspect"] = False
    if state["dialogue_text"]:
        normalized_dialogue = re.sub(r"[^A-Za-z0-9]", "", state["dialogue_text"]).upper()
        party_names = {
            re.sub(r"[^A-Za-z0-9]", "", mon.get("name", "")).upper()
            for mon in party
            if mon.get("name")
        }
        if normalized_dialogue and normalized_dialogue in party_names:
            state["dialogue_text_suspect"] = True
            state["dialogue_text"] = ""
        elif len(normalized_dialogue) <= 6 and " " not in state["dialogue_text"] and state["dialogue_text"].isupper():
            state["dialogue_text_suspect"] = True
            state["dialogue_text"] = ""
    if state["dialogue_text"]:
        print(f"  [dialogue] \"{state['dialogue_text']}\"")
    elif state["dialogue_text_suspect"]:
        print("  [dialogue] Ignoring suspicious dialogue buffer")

    state["_objects_raw"] = obj_raw

    # Parse bag data if present
    state["bag_items"] = None
    if bag_raw:
        try:
            state["bag_items"] = _parse_bag_data(bag_raw)
            pocket_name = state["bag_items"]["pocket_name"]
            cursor = state["bag_items"]["cursor"]
            scroll = state["bag_items"]["scroll"]
            item_count = sum(len(v) for v in state["bag_items"]["pockets"].values())
            pocket_keys = ["Items", "KeyItems", "PokeBalls", "TMs", "Berries"]
            current_idx = state["bag_items"]["current_pocket"]
            pocket_key = pocket_keys[current_idx] if current_idx < len(pocket_keys) else pocket_keys[0]
            visible_items = state["bag_items"]["pockets"].get(pocket_key, [])
            selected_idx = scroll + cursor
            selected_name = ""
            if 0 <= selected_idx < len(visible_items):
                selected_name = visible_items[selected_idx].get("name", "")
            print(
                f"  [bag] Pocket: {pocket_name}, {item_count} total items | "
                f"cursor={cursor} scroll={scroll} selected_idx={selected_idx} "
                f"selected={selected_name or '<none>'}"
            )
        except Exception as e:
            print(f"  [bag] Parse error: {e}")

    return state


def frames_similar(img_a: Image.Image, img_b: Image.Image, threshold: float = 0.95) -> bool:
    """Return True if two frames are more than `threshold` similar (0-1)."""
    arr_a = np.asarray(img_a.convert("L"), dtype=np.float32)
    arr_b = np.asarray(img_b.convert("L"), dtype=np.float32)
    if arr_a.shape != arr_b.shape:
        return False
    diff = np.abs(arr_a - arr_b).mean()
    similarity = 1.0 - (diff / 255.0)
    return similarity >= threshold


def test_connection() -> bool:
    """Wait for BizHawk Lua script to create the ready file, then test with PING."""
    os.makedirs(BRIDGE_DIR, exist_ok=True)

    # Clean up stale files
    for f in [CMD_FILE, RESP_FILE]:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    print(f"[bridge] Waiting for BizHawk Lua script...")
    print(f"[bridge] Load ROM, then load bizhawk_bridge.lua in Tools > Lua Console")

    for attempt in range(180):  # 3 minutes
        if os.path.exists(READY_FILE):
            print("[bridge] Lua script detected!")
            try:
                os.remove(READY_FILE)
            except FileNotFoundError:
                pass
            time.sleep(0.5)

            # Test with PING
            for ping_attempt in range(5):
                try:
                    result = _send_command("PING", timeout=5.0)
                    if result == "PONG":
                        print("[bridge] Bridge connected!")
                        return True
                except (TimeoutError, OSError):
                    pass
                time.sleep(1)

            print("[bridge] Lua script found but PING failed")
            return False

        time.sleep(1)

    print("[bridge] Lua script never started")
    return False


if __name__ == "__main__":
    if test_connection():
        print("\nConnected to BizHawk!")
        try:
            state = read_game_state()
            print(f"Game state: {json.dumps(state, indent=2)}")
        except Exception as e:
            print(f"Game state read failed: {e}")
            print("(Make sure ROM is loaded and you're in-game)")
    else:
        print("\nCould not connect. Steps:")
        print("  1. Run this Python script FIRST")
        print("  2. THEN launch BizHawk")
        print("  3. Load your ROM and the Lua script")
