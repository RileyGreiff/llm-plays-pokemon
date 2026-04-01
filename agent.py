"""Main game loop — reads state, calls Claude, presses buttons, logs everything."""

import asyncio
import json
import threading
import time
import sqlite3
import os
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
import websockets

load_dotenv()

import signal
from emulator import press_button, read_game_state, test_connection, get_collision_grid, get_objects, get_map_connections
from claude_client import get_action, get_navigation_target, HAIKU, SONNET
from exploration import ExplorationTracker
from navigation import PathState, plan_path_to_target, get_arrival_action
from world_knowledge import WorldKnowledge
from progress import (load_progress, save_progress, update_progress,
                      get_summary_line, rethink_objective,
                      check_tier1_update, check_tier2_update,
                      get_tier1_objective)

# Cost per 1M tokens (as of 2025)
COST_TABLE = {
    HAIKU: {"input": 0.80, "output": 4.00, "cache_read": 0.08, "cache_creation": 1.00},
    SONNET: {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_creation": 3.75},
}

LOOP_INTERVAL = 5  # seconds between actions
SUMMARY_INTERVAL = 50  # generate rolling summary every N actions
FRAME_DIFF_THRESHOLD = 0.99  # skip API call if frames are this similar
MAX_FRAME_SKIPS = 1  # force API call after this many consecutive skips
WS_PORT = 8765

# Websocket state
_ws_clients: set = set()
_ws_latest: dict = {}
_ws_loop: asyncio.AbstractEventLoop | None = None


async def _ws_handler(websocket):
    """Handle a new overlay websocket connection."""
    global _ws_clients
    _ws_clients.add(websocket)
    try:
        if _ws_latest:
            await websocket.send(json.dumps(_ws_latest))
        async for _ in websocket:
            pass
    finally:
        _ws_clients.discard(websocket)


def start_overlay_server():
    """Start websocket server on a daemon thread."""
    global _ws_loop

    async def _serve():
        try:
            async with websockets.serve(_ws_handler, "localhost", WS_PORT):
                await asyncio.Future()
        except OSError:
            pass  # port in use, overlay won't work but agent continues

    def _run():
        global _ws_loop
        _ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_ws_loop)
        _ws_loop.run_until_complete(_serve())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(0.3)


def broadcast_overlay(data: dict):
    """Send overlay data to all connected websocket clients."""
    global _ws_latest, _ws_clients
    _ws_latest = data
    if not _ws_loop or not _ws_clients:
        return
    msg = json.dumps(data)
    dead = set()
    for client in list(_ws_clients):
        try:
            asyncio.run_coroutine_threadsafe(client.send(msg), _ws_loop)
        except Exception:
            dead.add(client)
    _ws_clients -= dead


def init_db(db_path: str = "logs/runs.db") -> sqlite3.Connection:
    """Initialize SQLite logging database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            reason TEXT,
            model TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cache_read INTEGER,
            cache_creation INTEGER,
            cost_usd REAL,
            game_state TEXT,
            player_x INTEGER,
            player_y INTEGER,
            map_id INTEGER,
            in_battle INTEGER
        )
    """)
    # Add columns if missing
    for col in ["screen_description TEXT", "progress_summary TEXT",
                "exploration_summary TEXT", "warnings TEXT"]:
        try:
            conn.execute(f"ALTER TABLE actions ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return conn


def calculate_cost(usage: dict) -> float:
    """Calculate USD cost from token usage."""
    if usage["model"] == "nav-state":
        return 0.0
    rates = COST_TABLE.get(usage["model"], COST_TABLE[HAIKU])
    cost = (
        usage["input_tokens"] * rates["input"] / 1_000_000
        + usage["output_tokens"] * rates["output"] / 1_000_000
        + usage["cache_read"] * rates["cache_read"] / 1_000_000
        + usage["cache_creation"] * rates["cache_creation"] / 1_000_000
    )
    return round(cost, 6)


def log_action(conn: sqlite3.Connection, action: dict, usage: dict,
               cost: float, game_state: dict,
               progress_summary: str = "", exploration_summary: str = "",
               warnings: str = "") -> None:
    """Log an action to the database."""
    conn.execute(
        """INSERT INTO actions
           (timestamp, action, reason, model, input_tokens, output_tokens,
            cache_read, cache_creation, cost_usd, game_state,
            player_x, player_y, map_id, in_battle, screen_description,
            progress_summary, exploration_summary, warnings)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.utcnow().isoformat(),
            action.get("action", ""),
            action.get("reason", ""),
            usage["model"],
            usage["input_tokens"],
            usage["output_tokens"],
            usage["cache_read"],
            usage["cache_creation"],
            cost,
            json.dumps(game_state),
            game_state.get("player_x", 0),
            game_state.get("player_y", 0),
            game_state.get("map_id", 0),
            int(game_state.get("in_battle", False)),
            action.get("screen", ""),
            progress_summary,
            exploration_summary or "",
            warnings or "",
        ),
    )
    conn.commit()


def get_session_stats(conn: sqlite3.Connection) -> dict:
    """Get running totals for the current session."""
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(cost_usd), 0) FROM actions"
    ).fetchone()
    return {"total_actions": row[0], "total_cost": round(row[1], 4)}


def get_hourly_actions(conn: sqlite3.Connection) -> list[dict]:
    """Get actions from the last hour for summarization."""
    rows = conn.execute("""
        SELECT action, reason, map_id, player_x, player_y, in_battle
        FROM actions
        WHERE timestamp > datetime('now', '-1 hour')
        ORDER BY id
    """).fetchall()
    return [{"action": r[0], "reason": r[1], "map_id": r[2],
             "x": r[3], "y": r[4], "battle": r[5]} for r in rows]


def generate_hourly_summary(conn: sqlite3.Connection) -> str | None:
    """Generate a 50-word summary of the last hour using Claude."""
    actions = get_hourly_actions(conn)
    if not actions:
        return None

    # Build a compact log for Claude to summarize
    action_log = []
    for a in actions:
        action_log.append(f"{a['action']}: {a['reason'][:60]}")

    log_text = "\n".join(action_log[-60:])  # last 60 actions max

    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=HAIKU,
        max_tokens=100,
        system="Summarize this Pokemon FireRed gameplay hour in 50 words or less. Only report what ACTUALLY happened based on the action log. Do NOT invent progress or claim success that isn't shown. Focus on: locations visited, battles fought, items obtained, actual milestones reached. Be honest if the player got stuck.",
        messages=[{"role": "user", "content": log_text}],
    )
    return response.content[0].text.strip()


def print_hourly_report(conn: sqlite3.Connection) -> None:
    """Print cost breakdown for the last hour."""
    row = conn.execute("""
        SELECT COUNT(*), COALESCE(SUM(cost_usd), 0),
               COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0),
               COALESCE(SUM(cache_read), 0)
        FROM actions
        WHERE timestamp > datetime('now', '-1 hour')
    """).fetchone()
    actions, cost, inp, out, cached = row
    total = get_session_stats(conn)
    print(f"\n{'=' * 60}")
    print(f"  HOURLY REPORT")
    print(f"  Last hour: {actions} actions, ${cost:.4f}")
    print(f"  Tokens — in: {inp:,}  out: {out:,}  cached: {cached:,}")
    print(f"  All time: {total['total_actions']} actions, ${total['total_cost']:.4f}")
    print(f"{'=' * 60}\n")


@dataclass
class NamingContext:
    """Tracks an in-progress nickname entry flow."""

    prompt_step: int = 0
    naming_step: int = 0
    last_party_count: int = 0


def _is_probable_naming_screen(game_state: dict, naming: NamingContext) -> bool:
    """Return True when the explicit naming-screen state is active."""
    return game_state.get("game_state") == "naming"


def _is_nickname_prompt(game_state: dict, naming: NamingContext) -> bool:
    """Return True when the explicit nickname yes/no prompt is active."""
    return game_state.get("game_state") == "nickname_prompt"


def _maybe_begin_naming(game_state: dict, naming: NamingContext):
    """Reset prompt sequencing when the player first receives a Pokemon."""
    current_party_count = game_state.get("party_count", 0)
    if naming.last_party_count == 0 and current_party_count > 0:
        naming.prompt_step = 0
        naming.naming_step = 0
    naming.last_party_count = current_party_count


def _get_naming_action(game_state: dict, naming: NamingContext) -> dict | None:
    """Return the next deterministic nickname-prompt action, if any."""
    if _is_nickname_prompt(game_state, naming):
        naming.naming_step = 0
        if naming.prompt_step == 0:
            naming.prompt_step = 1
            return {
                "action": "Down",
                "reason": "Structured naming: move to 'No' on the nickname prompt",
                "display": "Selecting No on the nickname prompt.",
            }
        naming.prompt_step = 0
        return {
            "action": "A",
            "reason": "Structured naming: confirm 'No' on the nickname prompt",
            "display": "Declining the nickname prompt.",
        }
    if not _is_probable_naming_screen(game_state, naming):
        naming.naming_step = 0
        return None
    naming.prompt_step = 0
    if naming.naming_step == 0:
        naming.naming_step = 1
        return {
            "action": "Start",
            "reason": "Structured naming: open the OK confirmation on the naming screen",
            "display": "Opening the naming OK option.",
        }
    naming.naming_step = 0
    return {
        "action": "A",
        "reason": "Structured naming: confirm OK on the naming screen",
        "display": "Confirming the naming screen selection.",
    }


def _battle_action_button_toward(cursor: int, target: int) -> str:
    """Return the next directional/A press for the 2x2 battle action menu."""
    if cursor == target:
        return "A"
    positions = {
        0: (0, 0),  # Fight
        1: (1, 0),  # Bag
        2: (0, 1),  # Pokemon
        3: (1, 1),  # Run
    }
    cur_x, cur_y = positions.get(cursor, (0, 0))
    tgt_x, tgt_y = positions.get(target, (0, 0))
    if cur_x < tgt_x:
        return "Right"
    if cur_x > tgt_x:
        return "Left"
    if cur_y < tgt_y:
        return "Down"
    if cur_y > tgt_y:
        return "Up"
    return "A"


def _get_battle_menu_action(game_state: dict, llm_action: dict | None = None) -> dict | None:
    """Deterministically drive the 2x2 battle action menu when intent is obvious."""
    if not game_state.get("in_battle", False):
        return None
    if game_state.get("battle_menu_state", 0) != 1:
        return None

    reason_text = ""
    if llm_action:
        reason_text = " ".join(
            str(llm_action.get(key, "")) for key in ("action", "reason", "display")
        ).lower()

    hp = game_state.get("player_hp", 0)
    max_hp = max(game_state.get("party", [{}])[0].get("max_hp", 0), 1) if game_state.get("party") else 1
    low_hp = hp > 0 and hp / max_hp <= 0.3
    wants_run = any(token in reason_text for token in ("run", "flee", "escape"))
    if not wants_run and not low_hp:
        return None

    cursor = game_state.get("battle_action_cursor", 0)
    button = _battle_action_button_toward(cursor, 3)
    return {
        "action": button,
        "reason": "Structured battle: navigate the action menu toward Run using the live cursor state",
        "display": "Trying to flee from battle.",
    }


def main():
    print("LLM Plays Pokemon")
    print("=" * 40)

    # Test BizHawk connection
    if not test_connection():
        print("ERROR: Cannot connect to BizHawk.")
        print("Make sure BizHawk is running with the Lua bridge script loaded.")
        return

    print("Connected to BizHawk!")

    # Start overlay websocket server
    start_overlay_server()
    print(f"Overlay server running on ws://localhost:{WS_PORT}")

    conn = init_db()
    explorer = ExplorationTracker()
    knowledge = WorldKnowledge()
    path_state = PathState()
    naming = NamingContext()
    progress = load_progress()
    recent_actions: list[dict] = []
    action_count = progress.get("total_actions", 0)

    last_hourly_report = time.time()
    last_map_id = None
    last_game_state_str = None
    last_pos: tuple[int, int] | None = None  # for door-learning on map transitions
    last_tile_type: str | None = None         # "D"/"S"/None at last position
    hourly_summaries: list[dict] = []  # last 4 hourly summaries for overlay
    shutting_down = False

    def _shutdown_handler(sig, frame):
        nonlocal shutting_down
        shutting_down = True

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    print(f"Loop interval: {LOOP_INTERVAL}s")
    print(f"Resuming from action #{action_count}")
    print("Starting game loop... (Ctrl+C to stop)\n")

    try:
        while not shutting_down:
            loop_start = time.time()

            # 1. Read game state (screenshot disabled — using structured data + text)
            try:
                game_state = read_game_state()
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                print(f"  [err] Failed to read game state: {e}")
                time.sleep(5)
                continue

            # Door/warp transitions sometimes produce a single bogus "unknown + in_battle"
            # read with no enemy present. Skip one loop so it doesn't churn objectives.
            if (
                game_state.get("game_state") == "unknown"
                and game_state.get("in_battle", False)
                and game_state.get("enemy_species", 0) == 0
                and last_game_state_str == "overworld"
            ):
                print("  [transition] Ignoring transient doorway state glitch")
                time.sleep(0.2)
                continue

            _maybe_begin_naming(game_state, naming)
            naming_ui_active = _is_nickname_prompt(game_state, naming) or _is_probable_naming_screen(game_state, naming)

            print(
                "  [state] "
                f"pos=({game_state.get('player_x')},{game_state.get('player_y')}) "
                f"map={game_state.get('map_name')} "
                f"hp={game_state.get('player_hp')} "
                f"battle={game_state.get('in_battle')} "
                f"parcel={game_state.get('has_oaks_parcel', False)} "
                f"pokedex={game_state.get('has_pokedex', False)} "
                f"cb2=0x{game_state.get('cb2_raw', 0):08X}"
            )
            if naming_ui_active:
                print("  [naming] Naming UI active; ignoring player-position wall checks")

            # Record tile visit
            map_id = game_state.get("map_id", 0)
            px = game_state.get("player_x", 0)
            py = game_state.get("player_y", 0)
            explorer.record_visit(map_id, px, py)

            # --- World knowledge: learn from map transitions ---
            map_changed = (map_id != last_map_id) and last_map_id is not None
            map_name = game_state.get("map_name", "UNKNOWN")

            _outdoor_tags = ("TOWN", "CITY", "ROUTE", "LAKE", "ISLAND")
            is_outdoor_map = any(tag in map_name.upper() for tag in _outdoor_tags)

            if map_changed:
                # If we walked through a door/stairs, label it with destination
                if last_tile_type in ("D", "S") and last_pos is not None:
                    knowledge.learn_door(last_map_id, last_pos[0], last_pos[1], map_name)
                    print(f"  [knowledge] Door at map {last_map_id} ({last_pos[0]},{last_pos[1]}) -> {map_name}")
                # Learn map edge connections (outdoor maps only)
                if is_outdoor_map:
                    map_connections = get_map_connections(map_id)
                    knowledge.learn_map_edges(map_id, map_connections)
                # Clear path on map change — need to re-plan
                path_state.clear()

            # Track current tile type for door learning
            current_tile_type = None
            collision = None
            if not game_state.get("in_battle", False):
                collision = get_collision_grid(map_id)
                if collision:
                    grid_w, grid_h, grid_rows = collision
                    if 0 <= px < grid_w and 0 <= py < grid_h:
                        current_tile_type = grid_rows[py][px]

            last_pos = (px, py)
            last_tile_type = current_tile_type
            last_map_id = map_id

            # 3. Update progress
            progress = update_progress(progress, game_state, action_count)
            progress_summary = get_summary_line(progress)

            # 3a. Tiered objective updates
            current_game_state_str = game_state.get("game_state", "unknown")
            state_changed = (current_game_state_str != last_game_state_str) and last_game_state_str is not None

            entered_battle = current_game_state_str == "battle" and last_game_state_str != "battle"
            exited_battle = current_game_state_str == "overworld" and last_game_state_str not in ("overworld", None)

            if state_changed:
                print(f"  [state change] {last_game_state_str} -> {current_game_state_str}")
            last_game_state_str = current_game_state_str

            check_tier1_update(progress, game_state)

            in_battle_now = current_game_state_str == "battle"
            planner_state = dict(game_state)
            planner_state["_progress_context"] = progress
            if entered_battle or exited_battle:
                progress["tier2_last_action"] = action_count - 50
            check_tier2_update(progress, planner_state, action_count, in_battle=in_battle_now)

            periodic_interval = 10 if in_battle_now else 25
            needs_objective = map_changed or entered_battle or exited_battle or (action_count > 0 and action_count % periodic_interval == 0)
            if needs_objective:
                trigger = "map_change" if map_changed else "battle_enter" if entered_battle else "battle_exit" if exited_battle else "periodic"
                print(f"  [tier3] Triggered by: {trigger} (state={current_game_state_str})")
                try:
                    tier2 = progress.get("tier2_objective", "")
                    new_objective = rethink_objective(planner_state, tier2, in_battle=in_battle_now)
                    progress["current_objective"] = new_objective
                    print(f"  [tier3] {new_objective}")
                except Exception as e:
                    print(f"  [tier3] Failed: {e}")

            # 4. Gather overworld data
            is_overworld = current_game_state_str == "overworld"
            objects = None
            player_facing = None
            exploration_summary = None
            naming_action = None

            if is_overworld:
                if not collision:
                    collision = get_collision_grid(map_id)
                objects, player_facing = get_objects(game_state)
                # Learn map edges on first visit (outdoor maps only)
                if is_outdoor_map and not knowledge.get_map_edges(map_id):
                    map_connections = get_map_connections(map_id)
                    knowledge.learn_map_edges(map_id, map_connections)
                exploration_summary = explorer.get_summary(
                    map_id, map_name, px, py, collision, objects,
                    world_knowledge=knowledge,
                ) or None
                if exploration_summary:
                    print(f"  [explore]\n{exploration_summary}")
            else:
                print(f"  [skip] Skipping map/object data (state={current_game_state_str})")

            naming_action = _get_naming_action(game_state, naming)
            if naming_action:
                print(f"  [naming] {naming_action['reason']}")

            # Clear path if battle started or we left overworld
            if entered_battle or (not is_overworld and path_state.active):
                path_state.clear()

            # 5. Choose next action
            NO_COST = {"model": "nav-state", "input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0}

            battle_menu_action = _get_battle_menu_action(game_state)
            if battle_menu_action:
                action = battle_menu_action
                usage = NO_COST
            elif (
                game_state.get("game_state") == "unknown"
                and not game_state.get("in_battle", False)
                and not game_state.get("in_dialogue", False)
                and "POKECENTER" in game_state.get("map_name", "")
            ):
                action = {"action": "B", "reason": "Backing out of unknown Pokemon Center menu", "display": "Backing out of menu."}
                usage = NO_COST
            elif naming_action:
                action = naming_action
                usage = NO_COST
            elif game_state.get("in_dialogue", False) and not game_state.get("in_battle", False):
                action = {"action": "A", "reason": "Advance dialogue", "display": "Advancing dialogue."}
                usage = NO_COST
                # Learn NPC info from dialogue
                if path_state.target_type == "npc" and path_state.target_npc_id is not None:
                    dialogue_text = game_state.get("dialogue_text", "")
                    if dialogue_text:
                        knowledge.learn_npc(map_id, path_state.target_npc_id, map_name, dialogue_text)
                        print(f"  [knowledge] NPC #{path_state.target_npc_id}: \"{dialogue_text[:60]}\"")
            elif is_overworld and collision:
                # --- Path-based navigation ---
                # Check if path is stalled (position unchanged for several steps)
                if path_state.active:
                    if path_state.last_pos == (px, py):
                        path_state.stalled_steps += 1
                    else:
                        path_state.stalled_steps = 0
                    path_state.last_pos = (px, py)
                    # If stalled for 3+ steps, clear path and re-plan
                    if path_state.stalled_steps >= 3:
                        print(f"  [path] Stalled for {path_state.stalled_steps} steps, re-planning")
                        path_state.clear()

                # If no active path, ask LLM for a target (try up to 3 times with different targets)
                if not path_state.active and exploration_summary:
                    for _attempt in range(3):
                        nav_result = get_navigation_target(exploration_summary, progress_summary)
                        if not nav_result:
                            break
                        planned = plan_path_to_target(
                            nav_result["target"], collision, px, py, objects
                        )
                        if planned:
                            planned.reason = nav_result["reason"]
                            planned.display = nav_result["display"]
                            path_state.target_type = planned.target_type
                            path_state.target_id = planned.target_id
                            path_state.target_pos = planned.target_pos
                            path_state.target_npc_id = planned.target_npc_id
                            path_state.path = planned.path
                            path_state.reason = planned.reason
                            path_state.display = planned.display
                            path_state.stalled_steps = 0
                            path_state.last_pos = (px, py)
                            print(f"  [path] Target: {nav_result['target']} ({nav_result['reason']}) — {len(planned.path)} steps")
                            break
                        else:
                            print(f"  [path] No path found to {nav_result['target']}, retrying...")

                # Execute path or arrival action
                if path_state.active and path_state.path:
                    step = path_state.path.pop(0)
                    action = {"action": step, "reason": f"Pathfinding to {path_state.target_id}: {path_state.reason}", "display": path_state.display}
                    usage = NO_COST
                elif path_state.active and not path_state.path:
                    # Arrived at target
                    arrival = get_arrival_action(path_state, px, py, collision)
                    if arrival:
                        action = {"action": arrival, "reason": f"Arrived at {path_state.target_id}: {path_state.reason}", "display": path_state.display}
                    else:
                        action = {"action": "A", "reason": f"Arrived at {path_state.target_id}", "display": path_state.display}
                    usage = NO_COST
                    path_state.clear()
                else:
                    # No valid path found after retries — press A and try again next tick
                    action = {"action": "A", "reason": "No reachable target found, waiting", "display": "Waiting for a valid path."}
                    usage = NO_COST
            else:
                # Non-overworld (battle, bag, etc.) — call LLM
                action, usage = get_action(game_state, recent_actions, progress_summary)

            # 6. Press the button
            button = action.get("action", "A")
            reason = action.get("reason", "no reason given")
            try:
                press_frames = 16
                if naming_action or (game_state.get("in_dialogue", False) and not game_state.get("in_battle", False)):
                    press_frames = 2
                press_button(button, frames=press_frames)
            except (TimeoutError, ConnectionError, OSError) as e:
                print(f"  [err] Failed to press button: {e}")
                time.sleep(5)
                continue

            # 6b. NPC interaction detection
            in_dialogue = game_state.get("in_dialogue", False)
            if button == "A" and objects and player_facing is not None:
                facing_offsets = {1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)}
                dx, dy = facing_offsets.get(player_facing, (0, 0))
                faced_x, faced_y = px + dx, py + dy
                faced_obj = None
                for obj in objects:
                    if obj["x"] == faced_x and obj["y"] == faced_y:
                        faced_obj = obj
                        break
                if faced_obj:
                    try:
                        new_state = read_game_state()
                        new_dialogue = new_state.get("in_dialogue", False)
                        if new_dialogue and not in_dialogue:
                            count = explorer.record_interaction(map_id, faced_obj["local_id"], faced_obj["label"])
                            dialogue_text = new_state.get("dialogue_text", "")
                            knowledge.learn_npc(map_id, faced_obj["local_id"], map_name, dialogue_text)
                            print(f"  [interact] Talked to {faced_obj['label']} (id={faced_obj['local_id']}) — {count}x total")
                    except (TimeoutError, ConnectionError, OSError, ValueError):
                        pass

            # 7. Calculate cost and log
            cost = calculate_cost(usage)
            log_action(conn, action, usage, cost, game_state,
                       progress_summary=progress_summary,
                       exploration_summary=exploration_summary or "")
            action_count += 1

            # 8. Update recent actions
            recent_actions.append({"action": button, "reason": reason, "px": px, "py": py})
            if len(recent_actions) > 50:
                recent_actions = recent_actions[-50:]

            # 9. Periodic saves
            if action_count % SUMMARY_INTERVAL == 0:
                save_progress(progress)
                explorer.save()
                knowledge.save()
                print("  Progress, exploration, and knowledge saved.")

            # 10. Broadcast to overlay
            stats = get_session_stats(conn)
            broadcast_overlay({
                "action": button,
                "reason": action.get("display", reason),
                "model": usage["model"],
                "in_battle": game_state.get("in_battle", False),
                "action_count": action_count,
                "total_cost": stats["total_cost"],
                "badges": progress.get("badges", 0),
                "maps_visited": len(progress.get("maps_visited", [])),
                "objective": progress.get("current_objective", ""),
                "tier1_objective": progress.get("tier1_objective", ""),
                "tier2_objective": progress.get("tier2_objective", ""),
                "nav_target": path_state.target_id or "",
                "nav_path_len": len(path_state.path),
                "player_x": game_state.get("player_x", 0),
                "player_y": game_state.get("player_y", 0),
                "map_name": game_state.get("map_name", ""),
                "player_hp": game_state.get("player_hp", 0),
                "player_level": game_state.get("player_level", 0),
                "party_count": game_state.get("party_count", 0),
                "party": game_state.get("party", []),
                "has_oaks_parcel": game_state.get("has_oaks_parcel", False),
                "has_pokedex": game_state.get("has_pokedex", False),
                "hourly_summaries": hourly_summaries,
            })

            # 11. Print status
            if usage["model"] == "nav-state":
                model_short = "Nav"
            else:
                model_short = "Haiku" if "haiku" in usage["model"] else "Sonnet"
            battle_str = " [BATTLE]" if game_state.get("in_battle") else ""
            safe_reason = reason.encode("ascii", errors="replace").decode("ascii")
            print(
                f"[{action_count:>4}] {model_short:<6} | "
                f"{button:<6} | {safe_reason:<50} | "
                f"${cost:.4f} (total: ${stats['total_cost']:.4f})"
                f"{battle_str}"
            )

            # 12. Hourly cost report + summary
            if time.time() - last_hourly_report >= 3600:
                print_hourly_report(conn)
                try:
                    summary = generate_hourly_summary(conn)
                    if summary:
                        hourly_summaries.append({
                            "time": datetime.now().strftime("%H:%M"),
                            "text": summary,
                        })
                        hourly_summaries = hourly_summaries[-4:]
                        print(f"  [hourly] {summary}")
                except Exception as e:
                    print(f"  [err] Hourly summary failed: {e}")
                last_hourly_report = time.time()

            # 13. Wait for next loop
            elapsed = time.time() - loop_start
            wait = max(0, LOOP_INTERVAL - elapsed)
            if wait > 0:
                time.sleep(wait)

    finally:
        # Graceful shutdown
        stats = get_session_stats(conn)
        save_progress(progress)
        explorer.save()
        knowledge.save()
        print(f"\n\nStopped. Total actions: {action_count}, "
              f"Total cost: ${stats['total_cost']:.4f}")
        print("Progress saved to logs/progress.json")
        conn.close()


if __name__ == "__main__":
    main()
