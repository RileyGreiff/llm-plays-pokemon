"""Main game loop — reads state, calls local LLM via Ollama, presses buttons, logs everything."""

import asyncio
import json
import random
import threading
import time
import sqlite3
import os
from dataclasses import dataclass
from datetime import datetime
import websockets

import signal
from emulator import press_button, read_game_state, test_connection, get_collision_grid, get_objects, get_bg_events, get_map_connections
from ollama_client import get_action, HAIKU, SONNET, _strip_coordinates
from anti_stuck import AntiStuck
from exploration import ExplorationTracker, bfs_path_hint, bfs_to_unvisited
from ollama_navigation import NavContext, WorldSnapshot, decide_nav_action
from ollama_progress import (load_progress, save_progress, update_progress,
                             get_summary_line, rethink_objective,
                             check_tier1_update, check_tier2_update,
                             get_tier1_objective)

# Local model — no cost
COST_TABLE = {
    HAIKU: {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_creation": 0.0},
    SONNET: {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_creation": 0.0},
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
    """Generate a 50-word summary of the last hour using local LLM."""
    from ollama_client import _ollama_chat
    actions = get_hourly_actions(conn)
    if not actions:
        return None

    action_log = []
    for a in actions:
        action_log.append(f"{a['action']}: {a['reason'][:60]}")

    log_text = "\n".join(action_log[-60:])

    return _ollama_chat(
        [{"role": "user", "content": log_text}],
        max_tokens=100,
        system="Summarize this Pokemon FireRed gameplay hour in 50 words or less. Only report what ACTUALLY happened based on the action log. Do NOT invent progress or claim success that isn't shown. Focus on: locations visited, battles fought, items obtained, actual milestones reached. Be honest if the player got stuck.",
    )


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
    anti_stuck = AntiStuck()
    explorer = ExplorationTracker()
    navigator = NavContext()
    naming = NamingContext()
    progress = load_progress()
    recent_actions: list[dict] = []
    action_count = progress.get("total_actions", 0)

    last_hourly_report = time.time()
    last_map_id = None
    last_game_state_str = None
    active_path_hint = None  # BFS hint string, persists across ticks
    path_hint_ttl = 0        # ticks remaining to show the hint
    overworld_actions = 0    # counts overworld actions for autopilot trigger
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

            # Detect walls: if last action was a direction and position didn't change
            if recent_actions and not naming_ui_active:
                last_btn = recent_actions[-1]["action"]
                if last_btn in ("Up", "Down", "Left", "Right"):
                    last_x = recent_actions[-1].get("px", px)
                    last_y = recent_actions[-1].get("py", py)
                    if px == last_x and py == last_y:
                        explorer.record_wall(map_id, last_x, last_y, last_btn)

            # Position-based stuck detection (replaces frame diffing) — overworld only
            stuck_warning_frame = None
            blocked_dir = None
            is_in_battle = game_state.get("in_battle", False)
            if recent_actions and not is_in_battle and not naming_ui_active:
                last_action = recent_actions[-1]["action"]
                if last_action in ("Up", "Down", "Left", "Right"):
                    last_x = recent_actions[-1].get("px", px)
                    last_y = recent_actions[-1].get("py", py)
                    if px == last_x and py == last_y:
                        blocked_dir = last_action  # triggers new BFS hint
                        stuck_warning_frame = f"Your last action ({last_action}) had no effect — you hit a wall. Pick a DIFFERENT direction."
                        print(f"  [!] POSITION UNCHANGED — '{last_action}' hit a wall")

            # 3. Check for stuck conditions
            if naming_ui_active:
                stuck_warning, _force_sonnet = None, False
            else:
                stuck_warning, _force_sonnet = anti_stuck.check(
                    recent_actions, game_state, action_count
                )
            force_sonnet = False  # Haiku only for gameplay; Sonnet used only for objectives
            if stuck_warning:
                print(f"  [!] STUCK: {stuck_warning}")

            # Combine warnings
            all_warnings = "\n".join(w for w in [stuck_warning, stuck_warning_frame] if w) or None

            # 4. Update progress
            progress = update_progress(progress, game_state, action_count)
            progress_summary = get_summary_line(progress)

            # 4a. Tiered objective updates
            map_changed = (map_id != last_map_id) and last_map_id is not None
            current_game_state_str = game_state.get("game_state", "unknown")
            state_changed = (current_game_state_str != last_game_state_str) and last_game_state_str is not None

            # Track battle entry/exit using game_state_str (not in_battle, which fires during transition)
            entered_battle = current_game_state_str == "battle" and last_game_state_str != "battle"
            exited_battle = current_game_state_str == "overworld" and last_game_state_str not in ("overworld", None)

            last_map_id = map_id
            if state_changed:
                print(f"  [state change] {last_game_state_str} -> {current_game_state_str}")
            last_game_state_str = current_game_state_str

            # Tier 1: Updates only when badge count changes
            check_tier1_update(progress, game_state)

            # Tier 2: Updates every 50 actions or battle enter/exit.
            # Do not force a refresh on ordinary map changes, or the strategy can
            # thrash between neighboring maps (e.g. Route 1 <-> Viridian City).
            in_battle_now = current_game_state_str == "battle"
            planner_state = dict(game_state)
            planner_state["_progress_context"] = progress
            if entered_battle or exited_battle:
                progress["tier2_last_action"] = action_count - 50  # guarantee trigger
            check_tier2_update(progress, planner_state, action_count, in_battle=in_battle_now)

            # Tier 3: Updates every 25 actions (10 in battle), on map change, or battle enter/exit
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

            # 4b. Get collision grid, objects, bg events, and exploration summary
            # Skip expensive overworld data when in battle/bag/pokemon menus
            is_overworld = current_game_state_str == "overworld"
            map_name = game_state.get("map_name", "UNKNOWN")
            collision = None
            objects = None
            player_facing = None
            bg_events = []
            map_connections = []
            exploration_summary = None
            nav_action = None
            naming_action = None

            if is_overworld:
                collision = get_collision_grid(map_id)
                objects, player_facing = get_objects(game_state)
                if objects:
                    for obj in objects:
                        obj["interaction_count"] = explorer.get_interaction_count(map_id, obj["local_id"])
                bg_events = get_bg_events(map_id)
                map_connections = get_map_connections(map_id)
                exploration_summary = explorer.get_summary(map_id, map_name, px, py, collision, objects, bg_events) or None
                if exploration_summary:
                    print(f"  [explore]\n{exploration_summary}")
                world = WorldSnapshot(
                    game_state=current_game_state_str,
                    map_id=map_id,
                    map_name=map_name,
                    player_x=px,
                    player_y=py,
                    in_battle=game_state.get("in_battle", False),
                    in_dialogue=game_state.get("in_dialogue", False),
                    collision=collision,
                    objects=objects,
                    bg_events=bg_events,
                    connections=map_connections,
                    player_facing=player_facing,
                    current_objective=progress.get("current_objective", ""),
                    strategy_objective=progress.get("tier2_objective", ""),
                    party=game_state.get("party", []),
                )
                try:
                    nav_action = decide_nav_action(navigator, world)
                except Exception as e:
                    print(f"  [nav] ERROR: {e}")
                    nav_action = None
                if nav_action:
                    print(f"  [nav] {navigator.state}: {nav_action['reason']}")
                else:
                    print(f"  [nav] No action (state={navigator.state}, intent={navigator.intent})")
            else:
                print(f"  [skip] Skipping map/object data (state={current_game_state_str})")

            naming_action = _get_naming_action(game_state, naming)
            if naming_action:
                print(f"  [naming] {naming_action['reason']}")

            # 4b2. Autopilot — every 50 overworld actions, BFS toward unvisited tile
            if is_overworld:
                overworld_actions += 1
            if is_overworld and collision and not nav_action and overworld_actions % 50 == 0 and overworld_actions > 0:
                # Build visited set from exploration data
                visit_data = explorer.maps.get(map_id, {}).get("visits", {})
                visited_set = set()
                for key in visit_data:
                    vx, vy = key.split(",")
                    visited_set.add((int(vx), int(vy)))

                path_steps = bfs_to_unvisited(collision, px, py, visited_set)
                if path_steps:
                    print(f"  [autopilot] Navigating {len(path_steps)} steps toward unvisited tile")
                    for i, direction in enumerate(path_steps):
                        try:
                            press_button(direction)
                            explorer.record_visit(map_id, px, py)
                            action_count += 1
                            recent_actions.append({"action": direction, "reason": "autopilot", "px": px, "py": py})
                            if len(recent_actions) > 50:
                                recent_actions = recent_actions[-50:]
                            print(f"  [autopilot] Step {i+1}/{len(path_steps)}: {direction}")
                        except (TimeoutError, ConnectionError, OSError):
                            print(f"  [autopilot] Button press failed, aborting")
                            break
                        # Check if battle started — bail out
                        time.sleep(1)
                        try:
                            check_state = read_game_state()
                            px = check_state.get("player_x", px)
                            py = check_state.get("player_y", py)
                            if check_state.get("in_battle", False):
                                print(f"  [autopilot] Battle triggered, handing back to LLM")
                                break
                        except (TimeoutError, ConnectionError, OSError, ValueError):
                            break
                    print(f"  [autopilot] Done, resuming LLM control")
                    continue
                else:
                    print(f"  [autopilot] No unvisited tiles reachable, skipping")

            # 4c. Hint if player is standing on a door/warp tile
            if collision:
                grid_w, grid_h, grid_rows = collision
                if 0 <= py < grid_h and 0 <= px < grid_w and grid_rows[py][px] in ("D", "S"):
                    door_hint = "You are standing ON a door/warp tile (D). You must walk the correct direction to pass through it. Try all directions (Up, Down, Left, Right) while on this tile until one works."
                    all_warnings = "\n".join(w for w in [all_warnings, door_hint] if w)
                    print(f"  [hint] Player on door tile at ({px},{py})")

            # 4d. BFS path hint — recompute on new wall hit, persist for 5 ticks
            if blocked_dir and collision:
                hint = bfs_path_hint(collision, px, py, blocked_dir)
                if hint:
                    active_path_hint = hint
                    path_hint_ttl = 5
            if active_path_hint and path_hint_ttl > 0:
                all_warnings = "\n".join(w for w in [all_warnings, active_path_hint] if w)
                path_hint_ttl -= 1
                print(f"  [hint] {active_path_hint} (ttl={path_hint_ttl})")
            else:
                active_path_hint = None

            # 5. Choose next action.
            # Dialogue advancement is deterministic: if the game says dialogue is active,
            # advance it before asking either the nav layer or the LLM for movement.
            battle_menu_action = _get_battle_menu_action(game_state)
            if battle_menu_action:
                action = battle_menu_action
                usage = {
                    "model": "nav-state",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read": 0,
                    "cache_creation": 0,
                }
            elif (
                game_state.get("game_state") == "unknown"
                and not game_state.get("in_battle", False)
                and not game_state.get("in_dialogue", False)
                and "POKECENTER" in game_state.get("map_name", "")
            ):
                action = {
                    "action": "B",
                    "reason": "Structured menu escape: unknown non-dialogue UI in a Pokemon Center, backing out with B",
                    "display": "Backing out of an unknown Pokemon Center menu.",
                }
                usage = {
                    "model": "nav-state",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read": 0,
                    "cache_creation": 0,
                }
            elif naming_action:
                action = naming_action
                usage = {
                    "model": "nav-state",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read": 0,
                    "cache_creation": 0,
                }
            elif nav_action:
                # Nav has a structured path — use it and clear random recovery
                if anti_stuck.in_random_recovery():
                    anti_stuck.random_recovery_remaining = 0
                action = nav_action
                usage = {
                    "model": "nav-state",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read": 0,
                    "cache_creation": 0,
                }
            elif anti_stuck.in_random_recovery() and is_overworld:
                anti_stuck.consume_random_recovery_turn()
                button = random.choice(["Up", "Down", "Left", "Right"])
                remaining = anti_stuck.random_recovery_remaining
                action = {
                    "action": button,
                    "reason": (
                        "Structured stuck recovery: no tile movement for 5 turns, "
                        f"taking a random movement action ({remaining} recovery turns remain after this one)"
                    ),
                    "display": "Random recovery movement.",
                }
                usage = {
                    "model": "nav-state",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read": 0,
                    "cache_creation": 0,
                }
            elif game_state.get("in_dialogue", False) and not game_state.get("in_battle", False):
                action = {
                    "action": "A",
                    "reason": "Structured dialogue: advance active dialogue before taking any movement action",
                    "display": "Advancing dialogue.",
                }
                usage = {
                    "model": "nav-state",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read": 0,
                    "cache_creation": 0,
                }
            else:
                action, usage = get_action(
                    game_state=game_state,
                    recent_actions=recent_actions,
                    progress_summary=progress_summary,
                    stuck_warning=all_warnings,
                    force_sonnet=force_sonnet,
                    exploration_summary=exploration_summary,
                )

            # 6. Press the button (prevent undoing last directional move, except in dialogue/battle)
            button = action.get("action", "A")
            reason = action.get("reason", "no reason given")
            in_dialogue = game_state.get("in_dialogue", False)
            in_battle = game_state.get("in_battle", False)
            if usage["model"] != "nav-state" and not in_dialogue and not in_battle:
                opposites = {"Left": "Right", "Right": "Left", "Up": "Down", "Down": "Up"}
                if recent_actions and button in opposites:
                    last_btn = recent_actions[-1]["action"]
                    if opposites.get(last_btn) == button:
                        # Pick a perpendicular direction instead
                        perpendicular = {"Left": "Up", "Right": "Down", "Up": "Right", "Down": "Left"}
                        new_btn = perpendicular[button]
                        print(f"  [blocked] {button} would undo {last_btn}, redirecting to {new_btn}")
                        reason = f"Blocked {button} (would undo {last_btn}), trying {new_btn}"
                        button = new_btn
            try:
                press_frames = 16
                if naming_action or (game_state.get("in_dialogue", False) and not game_state.get("in_battle", False)):
                    press_frames = 2
                press_button(button, frames=press_frames)
            except (TimeoutError, ConnectionError, OSError) as e:
                print(f"  [err] Failed to press button: {e}")
                time.sleep(5)
                continue

            # 6b. Interaction detection: if A was pressed, check if interaction happened
            if button == "A" and objects and player_facing is not None:
                # Determine faced tile based on player facing
                # FireRed facing: 1=down, 2=up, 3=left, 4=right
                facing_offsets = {1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)}
                dx, dy = facing_offsets.get(player_facing, (0, 0))
                faced_x, faced_y = px + dx, py + dy

                # Find object on the faced tile
                faced_obj = None
                for obj in objects:
                    if obj["x"] == faced_x and obj["y"] == faced_y:
                        faced_obj = obj
                        break

                if faced_obj:
                    # Check if interaction happened by reading new game state
                    try:
                        new_state = read_game_state()
                        new_dialogue = new_state.get("in_dialogue", False)
                        # Dialogue started = interaction confirmed
                        if new_dialogue and not in_dialogue:
                            count = explorer.record_interaction(map_id, faced_obj["local_id"], faced_obj["label"])
                            print(f"  [interact] Confirmed interaction with {faced_obj['label']} (id={faced_obj['local_id']}) — {count}x total")
                        else:
                            # Check if object vanished (item pickup)
                            new_objects, _ = get_objects()
                            if new_objects is not None:
                                new_ids = {o["local_id"] for o in new_objects}
                                if faced_obj["local_id"] not in new_ids:
                                    count = explorer.record_interaction(map_id, faced_obj["local_id"], faced_obj["label"])
                                    print(f"  [interact] Object {faced_obj['label']} (id={faced_obj['local_id']}) collected! — {count}x")
                    except (TimeoutError, ConnectionError, OSError, ValueError):
                        pass  # don't break the loop for interaction tracking

            # 7. Calculate cost and log
            cost = calculate_cost(usage)
            log_action(conn, action, usage, cost, game_state,
                       progress_summary=progress_summary,
                       exploration_summary=exploration_summary or "",
                       warnings=all_warnings or "")
            action_count += 1


            # 8. (objective now set by Sonnet every 25 actions, not by Haiku)

            # 8b. Update recent actions (include position for wall detection)
            recent_actions.append({"action": button, "reason": reason, "px": px, "py": py})
            if len(recent_actions) > 50:
                recent_actions = recent_actions[-50:]

            # 9. Fresh start every 240 actions (keep minimap data)
            if action_count % 240 == 0 and action_count > 0:
                print(f"  [reset] Action #{action_count} — wiping recent context for fresh start")
                # Rethink tier 3 objective using Sonnet before clearing context
                try:
                    tier2 = progress.get("tier2_objective", "")
                    planner_state = dict(game_state)
                    planner_state["_progress_context"] = progress
                    new_objective = rethink_objective(planner_state, tier2, in_battle=is_in_battle)
                    progress["current_objective"] = new_objective
                    print(f"  [rethink] New objective: {new_objective}")
                except Exception as e:
                    print(f"  [rethink] Failed: {e}")
                    progress["current_objective"] = ""
                recent_actions.clear()

            # 9b. Save progress periodically
            if action_count % SUMMARY_INTERVAL == 0:
                save_progress(progress)
                explorer.save()
                print("  Progress and exploration data saved.")

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
                "nav_state": navigator.state.value,
                "nav_intent": navigator.intent or "",
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
            forced_str = " [FORCED]" if force_sonnet and "sonnet" in usage["model"] else ""
            safe_reason = reason.encode("ascii", errors="replace").decode("ascii")
            print(
                f"[{action_count:>4}] {model_short:<6} | "
                f"{button:<6} | {safe_reason:<50} | "
                f"${cost:.4f} (total: ${stats['total_cost']:.4f})"
                f"{battle_str}{forced_str}"
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
        print(f"\n\nStopped. Total actions: {action_count}, "
              f"Total cost: ${stats['total_cost']:.4f}")
        print("Progress saved to logs/progress.json")
        conn.close()


if __name__ == "__main__":
    main()
