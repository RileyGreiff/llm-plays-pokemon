"""Per-map exploration heatmap, wall tracking, and deterministic path helpers."""

import json
import os
from collections import deque

SAVE_PATH = "logs/exploration.json"

PASSABLE_TILES = {"0", "D", "S", "G"}
CARDINAL_STEPS = [("Up", 0, -1), ("Down", 0, 1), ("Left", -1, 0), ("Right", 1, 0)]


def is_passable(tile: str) -> bool:
    """Return True if the tile can be walked on."""
    return tile in PASSABLE_TILES


def _neighbors() -> list[tuple[str, int, int]]:
    return CARDINAL_STEPS


def path_to_nearest_tile(collision_grid: tuple[int, int, list[str]],
                         player_x: int, player_y: int,
                         predicate,
                         max_steps: int = 40) -> tuple[list[str], tuple[int, int]] | tuple[None, None]:
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
                        max_steps: int = 120) -> list[str] | None:
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
                         max_steps: int = 40) -> tuple[list[str], tuple[int, int]] | tuple[None, None]:
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
                          max_steps: int = 40) -> tuple[list[str], tuple[int, int]] | tuple[None, None]:
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
                     max_steps: int = 40) -> tuple[list[str], tuple[int, int]] | tuple[None, None]:
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

    def _tile_description(self, tile_char: str, x: int, y: int,
                          visited: dict, is_outdoor: bool) -> str:
        """Describe a tile in plain text."""
        visit_count = visited.get((x, y), 0)
        visit_str = f" (visited {visit_count}x)" if visit_count > 0 else " (unvisited)"

        if tile_char == "1":
            return "wall"
        elif tile_char == "G":
            return "tall grass" + visit_str
        elif tile_char == "S":
            return "stairs" + visit_str
        elif tile_char == "D":
            label = "building entrance" if is_outdoor else "door/stairs"
            return label + visit_str
        elif tile_char == "0":
            return "walkable" + visit_str
        else:
            return "walkable" + visit_str

    def get_summary(self, map_id: int, map_name: str,
                    player_x: int = None, player_y: int = None,
                    collision_grid: tuple[int, int, list[str]] = None,
                    objects: list[dict] = None,
                    bg_events: list[dict] = None) -> str:
        """Generate pre-computed exploration summary.

        Instead of an ASCII grid, returns plain text describing:
        - Adjacent tiles (Up/Down/Left/Right)
        - Nearby doors/stairs
        - Unvisited tile count and direction
        - Overvisited tile warnings
        """
        self._ensure_map(map_id)
        data = self.maps[map_id]
        visits = data["visits"]

        if player_x is None or player_y is None:
            return ""

        # Parse visited coordinates
        visited = {}
        for key, count in visits.items():
            x, y = key.split(",")
            visited[(int(x), int(y))] = count

        # Detect outdoor map
        is_outdoor = any(tag in map_name.upper() for tag in
                         ["TOWN", "CITY", "ROUTE", "LAKE", "ISLAND"])

        lines = [f"MAP: {map_name} ({'outdoor' if is_outdoor else 'indoor'})"]

        # Adjacent tiles
        if collision_grid:
            grid_w, grid_h, grid_rows = collision_grid
            directions = {
                "Up": (player_x, player_y - 1),
                "Down": (player_x, player_y + 1),
                "Left": (player_x - 1, player_y),
                "Right": (player_x + 1, player_y),
            }
            # Build object lookup by position for quick access
            obj_at = {}
            if objects:
                for obj in objects:
                    obj_at[(obj["x"], obj["y"])] = obj

            # Build BG/coord event lookup by position
            bg_at = {}
            if bg_events:
                for ev in bg_events:
                    bg_at[(ev["x"], ev["y"])] = ev

            adj_parts = []
            interactions = self.maps[map_id].get("interactions", {})
            for dir_name, (tx, ty) in directions.items():
                if 0 <= tx < grid_w and 0 <= ty < grid_h:
                    tile = grid_rows[ty][tx]
                    obj_here = obj_at.get((tx, ty))
                    bg_here = bg_at.get((tx, ty))
                    if obj_here:
                        icount = interactions.get(str(obj_here["local_id"]), 0)
                        tag = f"talked {icount}x" if icount > 0 else "NOT yet interacted"
                        desc = f"{obj_here['label']} [{tag}] (face {dir_name} and press A)"
                    elif bg_here:
                        if bg_here["type"] == "coord":
                            desc = f"script trigger (walk onto this tile to activate)"
                        else:
                            desc = f"interactable {bg_here['label']} (face {dir_name} and press A)"
                    else:
                        desc = self._tile_description(tile, tx, ty, visited, is_outdoor)
                    adj_parts.append(f"  {dir_name}: {desc}")
                else:
                    if is_outdoor:
                        adj_parts.append(f"  {dir_name}: map edge (walk here to leave to next area)")
                    else:
                        adj_parts.append(f"  {dir_name}: wall (map boundary)")
            lines.append("ADJACENT TILES:")
            lines.extend(adj_parts)

            # Find all doors/stairs on the map
            doors = []
            for gy in range(grid_h):
                for gx in range(grid_w):
                    if grid_rows[gy][gx] in ("D", "S"):
                        dist = abs(gx - player_x) + abs(gy - player_y)
                        label = "stairs" if grid_rows[gy][gx] == "S" else (
                            "entrance" if is_outdoor else "door/stairs")
                        doors.append((gx, gy, dist, label))
            doors.sort(key=lambda d: d[2])
            if doors:
                door_strs = [f"({d[0]},{d[1]}) {d[3]} {d[2]} tiles away" for d in doors[:5]]
                lines.append(f"DOORS/EXITS: {', '.join(door_strs)}")

        # Objects (NPCs, items, Pokeballs) + BG events (script triggers, signs)
        all_interactables = []
        if objects:
            interactions = self.maps[map_id].get("interactions", {})
            for obj in objects:
                dist = abs(obj["x"] - player_x) + abs(obj["y"] - player_y)
                icount = interactions.get(str(obj["local_id"]), 0)
                tag = f" [talked {icount}x]" if icount > 0 else " [NOT yet interacted]"
                all_interactables.append((dist, f"({obj['x']},{obj['y']}) {obj['label']} {dist} tiles away{tag}"))
        if bg_events:
            for ev in bg_events:
                dist = abs(ev["x"] - player_x) + abs(ev["y"] - player_y)
                if ev["type"] == "coord":
                    all_interactables.append((dist, f"({ev['x']},{ev['y']}) walk-on trigger {dist} tiles away"))
                else:
                    all_interactables.append((dist, f"({ev['x']},{ev['y']}) {ev['label']} {dist} tiles away"))
        if all_interactables:
            all_interactables.sort(key=lambda x: x[0])
            lines.append(f"OBJECTS: {', '.join(s for _, s in all_interactables[:8])}")

        if collision_grid:
            # Count unvisited walkable tiles and find nearest direction
            unvisited = []
            for gy in range(grid_h):
                for gx in range(grid_w):
                    if is_passable(grid_rows[gy][gx]) and (gx, gy) not in visited:
                        unvisited.append((gx, gy))

            if unvisited:
                # Find general direction to nearest cluster of unvisited
                nearest = min(unvisited, key=lambda p: abs(p[0] - player_x) + abs(p[1] - player_y))
                dx = nearest[0] - player_x
                dy = nearest[1] - player_y
                dir_hints = []
                if dy < 0: dir_hints.append("north")
                if dy > 0: dir_hints.append("south")
                if dx < 0: dir_hints.append("west")
                if dx > 0: dir_hints.append("east")
                dir_str = "-".join(dir_hints) if dir_hints else "here"
                lines.append(f"UNEXPLORED: {len(unvisited)} tiles, nearest to the {dir_str}")
            else:
                lines.append("UNEXPLORED: none — map fully explored")

        # Current tile visit count
        current_visits = visited.get((player_x, player_y), 0)
        lines.append(f"CURRENT TILE: visited {current_visits}x")

        # Overvisited warning
        sorted_visits = sorted(visited.items(), key=lambda x: x[1], reverse=True)
        top_overvisited = [(k, v) for k, v in sorted_visits if v >= 5][:3]
        if top_overvisited:
            ov = ", ".join(f"({k[0]},{k[1]})={v}x" for k, v in top_overvisited)
            lines.append(f"OVERVISITED: {ov} — avoid these tiles!")

        return "\n".join(lines)
