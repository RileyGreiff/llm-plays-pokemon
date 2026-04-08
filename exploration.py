"""Per-map exploration heatmap, wall tracking, and deterministic path helpers."""

import json
import os
import re
from collections import deque

SAVE_PATH = "logs/exploration.json"

PASSABLE_TILES = {"0", "D", "S", "G"}
CARDINAL_STEPS = [("Up", 0, -1), ("Down", 0, 1), ("Left", -1, 0), ("Right", 1, 0)]
OUTDOOR_MAP_TAGS = ("TOWN", "CITY", "ROUTE", "LAKE", "ISLAND")
INDOOR_MAP_TAGS = (
    "POKECENTER",
    "MART",
    "GYM",
    "HOUSE",
    "MUSEUM",
    "LAB",
    "GATE",
    "ROOM",
    "DOJO",
    "MANSION",
    "HIDEOUT",
    "CAVE",
    "TUNNEL",
)


def is_passable(tile: str) -> bool:
    """Return True if the tile can be walked on."""
    return tile in PASSABLE_TILES


def is_outdoor_map_name(map_name: str) -> bool:
    """Return True for outdoor maps and False for indoor/interior maps."""
    upper = (map_name or "").upper()
    if any(tag in upper for tag in INDOOR_MAP_TAGS):
        return False
    if re.search(r"_(?:B?\d+F)$", upper):
        return False
    return any(tag in upper for tag in OUTDOOR_MAP_TAGS)


def _neighbors() -> list[tuple[str, int, int]]:
    return CARDINAL_STEPS


def path_to_nearest_tile(collision_grid: tuple[int, int, list[str]],
                         player_x: int, player_y: int,
                         predicate,
                         max_steps: int = 300) -> tuple[list[str], tuple[int, int]] | tuple[None, None]:
    """BFS to the nearest tile matching `predicate`.

    The predicate receives `(x, y, tile, grid_w, grid_h, grid_rows)`.
    """
    try:
        grid_w, grid_h, grid_rows = collision_grid
        start = (player_x, player_y)
        queue = deque([(start, [])])
        seen = {start}

        while queue:
            (cx, cy), path = queue.popleft()
            tile = grid_rows[cy][cx]
            if predicate(cx, cy, tile, grid_w, grid_h, grid_rows):
                return path[:max_steps], (cx, cy)

            if len(path) >= max_steps:
                continue

            for dir_name, dx, dy in _neighbors():
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < grid_w and 0 <= ny < grid_h and (nx, ny) not in seen:
                    if is_passable(grid_rows[ny][nx]):
                        seen.add((nx, ny))
                        queue.append(((nx, ny), path + [dir_name]))

        return None, None
    except Exception:
        return None, None


def path_to_target_tile(collision_grid: tuple[int, int, list[str]],
                        player_x: int, player_y: int,
                        target_x: int, target_y: int,
                        max_steps: int = 500) -> list[str] | None:
    """BFS to a specific reachable tile."""
    try:
        grid_w, grid_h, grid_rows = collision_grid
        if not (0 <= target_x < grid_w and 0 <= target_y < grid_h):
            return None
        if not is_passable(grid_rows[target_y][target_x]):
            return None

        start = (player_x, player_y)
        queue = deque([(start, [])])
        seen = {start}

        while queue:
            (cx, cy), path = queue.popleft()
            if (cx, cy) == (target_x, target_y):
                return path[:max_steps]

            if len(path) >= max_steps:
                continue

            for dir_name, dx, dy in _neighbors():
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < grid_w and 0 <= ny < grid_h and (nx, ny) not in seen:
                    if is_passable(grid_rows[ny][nx]):
                        seen.add((nx, ny))
                        queue.append(((nx, ny), path + [dir_name]))

        return None
    except Exception:
        return None


def path_to_nearest_door(collision_grid: tuple[int, int, list[str]],
                         player_x: int, player_y: int,
                         max_steps: int = 300) -> tuple[list[str], tuple[int, int]] | tuple[None, None]:
    """Find the nearest reachable door/entrance tile."""
    return path_to_nearest_tile(
        collision_grid,
        player_x,
        player_y,
        lambda x, y, tile, *_: tile == "D",
        max_steps=max_steps,
    )


def path_to_nearest_grass(collision_grid: tuple[int, int, list[str]],
                          player_x: int, player_y: int,
                          max_steps: int = 300) -> tuple[list[str], tuple[int, int]] | tuple[None, None]:
    """Find the nearest reachable tall grass tile."""
    return path_to_nearest_tile(
        collision_grid,
        player_x,
        player_y,
        lambda x, y, tile, *_: tile == "G",
        max_steps=max_steps,
    )


def path_to_map_edge(collision_grid: tuple[int, int, list[str]],
                     player_x: int, player_y: int,
                     max_steps: int = 300) -> tuple[list[str], tuple[int, int]] | tuple[None, None]:
    """Find the nearest reachable walkable tile on the edge of the current map."""
    return path_to_nearest_tile(
        collision_grid,
        player_x,
        player_y,
        lambda x, y, tile, grid_w, grid_h, _rows: (
            is_passable(tile)
            and (x == 0 or y == 0 or x == grid_w - 1 or y == grid_h - 1)
        ),
        max_steps=max_steps,
    )


def _bfs_to_targets(grid_w, grid_h, grid_rows, player_x, player_y,
                    targets, max_steps):
    """BFS from player to any target tile. Returns (path, obj, pos) or Nones."""
    if not targets:
        return None, None, None
    start = (player_x, player_y)
    queue = deque([(start, [])])
    seen = {start}
    target_lookup = {(tx, ty): obj for tx, ty, obj in targets}

    while queue:
        (cx, cy), path = queue.popleft()
        if (cx, cy) in target_lookup:
            return path[:max_steps], target_lookup[(cx, cy)], (cx, cy)
        if len(path) >= max_steps:
            continue
        for dir_name, dx, dy in _neighbors():
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid_w and 0 <= ny < grid_h and (nx, ny) not in seen:
                if is_passable(grid_rows[ny][nx]):
                    seen.add((nx, ny))
                    queue.append(((nx, ny), path + [dir_name]))
    return None, None, None


def _counter_talk_targets(grid_w, grid_h, grid_rows, objects, match_fn):
    """Find tiles 2 away from an NPC across a blocked counter tile."""
    targets = []
    for obj in objects or []:
        if not match_fn(obj):
            continue
        for _, dx, dy in _neighbors():
            mid_x, mid_y = obj["x"] + dx, obj["y"] + dy
            tx, ty = obj["x"] + dx * 2, obj["y"] + dy * 2
            if (0 <= mid_x < grid_w and 0 <= mid_y < grid_h
                    and 0 <= tx < grid_w and 0 <= ty < grid_h
                    and not is_passable(grid_rows[mid_y][mid_x])
                    and is_passable(grid_rows[ty][tx])):
                targets.append((tx, ty, obj))
    return targets


def path_to_adjacent_object(collision_grid: tuple[int, int, list[str]],
                            player_x: int, player_y: int,
                            objects: list[dict],
                            match_fn,
                            max_steps: int = 40) -> tuple[list[str], dict, tuple[int, int]] | tuple[None, None, None]:
    """Path to a walkable tile adjacent to the first matching object.

    Also considers tiles 2 away across a counter (blocked tile), since
    FireRed allows talking across counters.
    """
    try:
        grid_w, grid_h, grid_rows = collision_grid

        # 1. Direct adjacency targets (1 tile away, walkable)
        direct_targets: list[tuple[int, int, dict]] = []
        for obj in objects or []:
            if not match_fn(obj):
                continue
            for _, dx, dy in _neighbors():
                tx, ty = obj["x"] + dx, obj["y"] + dy
                if 0 <= tx < grid_w and 0 <= ty < grid_h and is_passable(grid_rows[ty][tx]):
                    direct_targets.append((tx, ty, obj))

        # 2. Try BFS to direct targets first
        if direct_targets:
            result = _bfs_to_targets(grid_w, grid_h, grid_rows, player_x, player_y,
                                     direct_targets, max_steps)
            if result[0] is not None:
                return result

        # 3. Direct targets unreachable or none — try counter-talk (2 tiles away)
        counter_targets = _counter_talk_targets(grid_w, grid_h, grid_rows, objects, match_fn)
        if counter_targets:
            result = _bfs_to_targets(grid_w, grid_h, grid_rows, player_x, player_y,
                                     counter_targets, max_steps)
            if result[0] is not None:
                return result

        return None, None, None
    except Exception:
        return None, None, None


def edge_exit_action(target_pos: tuple[int, int],
                     collision_grid: tuple[int, int, list[str]]) -> str | None:
    """Choose the outward movement needed to leave the current map from an edge tile."""
    try:
        x, y = target_pos
        grid_w, grid_h, _ = collision_grid
        if y == 0:
            return "Up"
        if y == grid_h - 1:
            return "Down"
        if x == 0:
            return "Left"
        if x == grid_w - 1:
            return "Right"
        return None
    except Exception:
        return None


def facing_direction(from_pos: tuple[int, int], to_pos: tuple[int, int]) -> str | None:
    """Return the button needed to face from one tile toward an adjacent tile."""
    fx, fy = from_pos
    tx, ty = to_pos
    dx = tx - fx
    dy = ty - fy
    lookup = {
        (0, -1): "Up",
        (0, 1): "Down",
        (-1, 0): "Left",
        (1, 0): "Right",
    }
    return lookup.get((dx, dy))


def bfs_path_hint(collision_grid: tuple[int, int, list[str]],
                  player_x: int, player_y: int,
                  blocked_dir: str, max_nodes: int = 200,
                  max_steps: int = 8) -> str | None:
    """BFS to find a walkable detour around an obstacle.

    When the player hits a wall going in `blocked_dir`, find the shortest path
    to the nearest reachable tile that is strictly further in that direction.
    Returns a string like "Down, Down, Right, Up" or None if no path found.
    Silently returns None on any error so it never disrupts the game loop.
    """
    try:
        grid_w, grid_h, grid_rows = collision_grid

        target_check = {
            "Up":    lambda x, y: y < player_y,
            "Down":  lambda x, y: y > player_y,
            "Left":  lambda x, y: x < player_x,
            "Right": lambda x, y: x > player_x,
        }.get(blocked_dir)
        if not target_check:
            return None

        start = (player_x, player_y)
        queue = deque([(start, [])])
        visited = {start}
        nodes_checked = 0
        neighbors = _neighbors()

        while queue and nodes_checked < max_nodes:
            (cx, cy), path = queue.popleft()
            nodes_checked += 1

            if path and target_check(cx, cy):
                steps = path[:max_steps]
                return f"PATH HINT: To continue {blocked_dir}, try: {', '.join(steps)}"

            if len(path) >= max_steps:
                continue

            for dir_name, dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < grid_w and 0 <= ny < grid_h and (nx, ny) not in visited:
                    tile = grid_rows[ny][nx]
                    if is_passable(tile):
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [dir_name]))

        return None
    except Exception:
        return None


def bfs_to_unvisited(collision_grid: tuple[int, int, list[str]],
                     player_x: int, player_y: int,
                     visited: set[tuple[int, int]],
                     max_steps: int = 15) -> list[str] | None:
    """BFS to find shortest path to nearest unvisited walkable tile.

    Returns a list of direction strings (e.g. ["Down", "Down", "Right"]),
    or None if no unvisited tile is reachable within max_steps.
    """
    try:
        grid_w, grid_h, grid_rows = collision_grid

        start = (player_x, player_y)
        queue = deque([(start, [])])
        seen = {start}
        neighbors = _neighbors()

        while queue:
            (cx, cy), path = queue.popleft()

            # Found an unvisited walkable tile (skip start)
            if path and (cx, cy) not in visited:
                return path[:max_steps]

            if len(path) >= max_steps:
                continue

            for dir_name, dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < grid_w and 0 <= ny < grid_h and (nx, ny) not in seen:
                    tile = grid_rows[ny][nx]
                    if is_passable(tile):
                        seen.add((nx, ny))
                        queue.append(((nx, ny), path + [dir_name]))

        return None
    except Exception:
        return None


def _group_doors(doors: list[tuple[int, int, str]]) -> list[list[tuple[int, int, str]]]:
    """Group door tiles within 1 tile of each other into doorways."""
    if not doors:
        return []
    used = [False] * len(doors)
    groups = []
    for i, (x1, y1, t1) in enumerate(doors):
        if used[i]:
            continue
        group = [(x1, y1, t1)]
        used[i] = True
        # Flood-fill: find all connected doors within 1 tile
        queue = [i]
        while queue:
            ci = queue.pop(0)
            cx, cy, _ = doors[ci]
            for j, (x2, y2, t2) in enumerate(doors):
                if not used[j] and abs(x2 - cx) <= 1 and abs(y2 - cy) <= 1:
                    used[j] = True
                    group.append((x2, y2, t2))
                    queue.append(j)
        groups.append(group)
    return groups


class ExplorationTracker:
    def __init__(self):
        # {map_id: {"visits": {"x,y": count}, "walls": ["x,y,dir", ...]}}
        self.maps: dict[int, dict] = {}
        self._load()

    def _load(self):
        if os.path.exists(SAVE_PATH):
            try:
                with open(SAVE_PATH, "r") as f:
                    raw = json.load(f)
                # Convert string keys back to int for map_id
                self.maps = {int(k): v for k, v in raw.items()}
            except (json.JSONDecodeError, ValueError):
                self.maps = {}

    def save(self):
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        with open(SAVE_PATH, "w") as f:
            json.dump(self.maps, f)

    def _ensure_map(self, map_id: int):
        if map_id not in self.maps:
            self.maps[map_id] = {"visits": {}, "walls": [], "interactions": {}}

    def record_visit(self, map_id: int, x: int, y: int):
        self._ensure_map(map_id)
        key = f"{x},{y}"
        visits = self.maps[map_id]["visits"]
        visits[key] = visits.get(key, 0) + 1

    def record_wall(self, map_id: int, x: int, y: int, direction: str):
        self._ensure_map(map_id)
        wall = f"{x},{y},{direction}"
        if wall not in self.maps[map_id]["walls"]:
            self.maps[map_id]["walls"].append(wall)

    def record_interaction(self, map_id: int, local_id: int, label: str) -> int:
        """Record a confirmed interaction with an object. Returns new count."""
        self._ensure_map(map_id)
        interactions = self.maps[map_id].setdefault("interactions", {})
        key = str(local_id)
        interactions[key] = interactions.get(key, 0) + 1
        return interactions[key]

    def get_interaction_count(self, map_id: int, local_id: int) -> int:
        """Get how many times we've interacted with an object."""
        if map_id not in self.maps:
            return 0
        interactions = self.maps[map_id].get("interactions", {})
        return interactions.get(str(local_id), 0)

    def get_summary(self, map_id: int, map_name: str,
                    player_x: int = None, player_y: int = None,
                    collision_grid: tuple[int, int, list[str]] = None,
                    objects: list[dict] = None,
                    bg_events: list[dict] = None,
                    world_knowledge=None,
                    entry_pos: tuple[int, int] | None = None) -> str:
        """Generate exploration summary showing discovered doors/exits and NPCs.

        Uses world_knowledge to label doors and NPCs as known or unknown.
        entry_pos: where the player spawned on this map (used to mark entrance doors).
        """
        if player_x is None or player_y is None:
            return ""

        is_outdoor = is_outdoor_map_name(map_name)

        lines = [f"MAP: {map_name} ({'outdoor' if is_outdoor else 'indoor'})"]

        # --- NEARBY EXITS (doors/stairs + map edges) ---
        exit_lines = []

        if collision_grid:
            grid_w, grid_h, grid_rows = collision_grid

            # Collect all door/stair tiles, then group adjacent ones into doorways
            raw_doors = []
            for gy in range(grid_h):
                for gx in range(grid_w):
                    if grid_rows[gy][gx] in ("D", "S"):
                        if world_knowledge:
                            world_knowledge.ensure_door(map_id, gx, gy)
                        raw_doors.append((gx, gy, grid_rows[gy][gx]))

            # Group door tiles within 1 tile of each other into doorways
            doorways = _group_doors(raw_doors)

            for doorway in doorways:
                # Use the center tile as the representative coordinate
                rep_x = doorway[len(doorway) // 2][0]
                rep_y = doorway[len(doorway) // 2][1]
                tile_code = doorway[0][2]
                dist = abs(rep_x - player_x) + abs(rep_y - player_y)

                # Label from world knowledge (use best known label from any tile in group)
                label = "unknown"
                if world_knowledge:
                    for dx, dy, _ in doorway:
                        dl = world_knowledge.get_door_label(map_id, dx, dy)
                        if dl != "unknown":
                            label = dl
                            break

                tile_type = "stairs" if tile_code == "S" else "door"

                # Check if this doorway is the entrance (player spawned here)
                is_entrance = False
                if entry_pos:
                    for dx, dy, _ in doorway:
                        if abs(dx - entry_pos[0]) <= 1 and abs(dy - entry_pos[1]) <= 1:
                            is_entrance = True
                            break

                # Direction hint from player
                dy_diff = rep_y - player_y
                dx_diff = rep_x - player_x
                dir_parts = []
                if dy_diff < 0: dir_parts.append("north")
                if dy_diff > 0: dir_parts.append("south")
                if dx_diff < 0: dir_parts.append("west")
                if dx_diff > 0: dir_parts.append("east")
                dir_str = "-".join(dir_parts) if dir_parts else "here"

                # Hide the entrance we just used whenever other exits exist.
                # This prevents immediate backtracking after a map change.
                if is_entrance and len(doorways) > 1:
                    continue
                exit_lines.append((dist, f"  {tile_type} at ({rep_x},{rep_y}) -> {label} ({dist} tiles {dir_str})"))

        # Map edge connections (outdoor maps only — indoor maps use doors/warps)
        if world_knowledge and is_outdoor:
            edges = world_knowledge.get_map_edges(map_id)
            for edge in edges:
                exit_lines.append((0, f"  {edge['direction']} edge -> {edge['label']}"))

        if exit_lines:
            exit_lines.sort(key=lambda x: x[0])
            lines.append("NEARBY EXITS:")
            lines.extend(desc for _, desc in exit_lines[:8])

        # --- NEARBY NPCs ---
        npc_lines = []
        if objects:
            for obj in objects:
                dist = abs(obj["x"] - player_x) + abs(obj["y"] - player_y)
                local_id = obj["local_id"]
                if world_knowledge:
                    npc_info = world_knowledge.get_npc_label(map_id, local_id)
                else:
                    npc_info = None
                if npc_info:
                    npc_lines.append((dist, local_id, f"  ({obj['x']},{obj['y']}) NPC #{local_id} — {npc_info} ({dist} tiles away)"))
                else:
                    npc_lines.append((dist, local_id, f"  ({obj['x']},{obj['y']}) NPC #{local_id} — unknown ({dist} tiles away)"))

        if npc_lines:
            npc_lines.sort(key=lambda x: x[0])
            lines.append("NEARBY NPCs:")
            lines.extend(desc for _, _, desc in npc_lines[:8])

        return "\n".join(lines)
