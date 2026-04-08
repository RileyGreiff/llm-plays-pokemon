"""Path-based overworld navigation for Pokemon FireRed.

The LLM picks a target (door, NPC, or map edge). This module handles
BFS pathfinding to that target and step-by-step execution.
"""

from dataclasses import dataclass, field
from exploration import (
    edge_exit_action,
    facing_direction,
    is_passable,
    path_to_adjacent_object,
    path_to_map_edge,
    path_to_nearest_door,
    path_to_target_tile,
)


@dataclass
class PathState:
    """Tracks the current navigation path being executed."""
    target_type: str | None = None        # "door", "npc", "edge"
    target_id: str | None = None          # "door:5,12", "npc:3", "edge:north"
    target_pos: tuple[int, int] | None = None  # where we're walking to
    target_npc_id: int | None = None      # local_id if targeting an NPC
    path: list[str] = field(default_factory=list)  # remaining BFS steps
    reason: str = ""
    display: str = ""
    stalled_steps: int = 0                # steps where position didn't change
    last_pos: tuple[int, int] | None = None

    def clear(self):
        self.target_type = None
        self.target_id = None
        self.target_pos = None
        self.target_npc_id = None
        self.path.clear()
        self.reason = ""
        self.display = ""
        self.stalled_steps = 0
        self.last_pos = None

    @property
    def active(self) -> bool:
        return self.target_type is not None


def parse_target(target_str: str) -> tuple[str, str]:
    """Parse a target string like 'door:5,12' into (type, value)."""
    parts = target_str.split(":", 1)
    if len(parts) != 2:
        return "", ""
    return parts[0], parts[1]


def _collision_with_npcs(
    collision: tuple[int, int, list[str]],
    objects: list[dict] | None,
    player_x: int, player_y: int,
    exclude_npc_id: int | None = None,
) -> tuple[int, int, list[str]]:
    """Return a copy of the collision grid with NPC positions marked as walls."""
    grid_w, grid_h, grid_rows = collision
    if not objects:
        return collision
    # Deep copy the rows so we can mutate them
    new_rows = [list(row) for row in grid_rows]
    for obj in objects:
        ox, oy = obj.get("x", -1), obj.get("y", -1)
        if (ox, oy) == (player_x, player_y):
            continue  # Don't block our own tile
        if exclude_npc_id is not None and obj.get("local_id") == exclude_npc_id:
            continue  # Don't block the NPC we're walking toward
        if 0 <= ox < grid_w and 0 <= oy < grid_h:
            new_rows[oy][ox] = "1"  # Mark as wall
    return (grid_w, grid_h, new_rows)


def _collision_avoiding_warps(
    collision: tuple[int, int, list[str]],
    player_x: int,
    player_y: int,
    allow_pos: tuple[int, int] | None = None,
) -> tuple[int, int, list[str]]:
    """Block non-target doors/stairs so routes don't accidentally warp."""
    grid_w, grid_h, grid_rows = collision
    new_rows = [list(row) for row in grid_rows]
    for y in range(grid_h):
        for x in range(grid_w):
            if (x, y) == (player_x, player_y):
                continue
            if allow_pos is not None and (x, y) == allow_pos:
                continue
            if new_rows[y][x] in ("D", "S"):
                new_rows[y][x] = "1"
    return (grid_w, grid_h, new_rows)


def plan_path_to_target(
    target_str: str,
    collision: tuple[int, int, list[str]],
    player_x: int,
    player_y: int,
    objects: list[dict] | None = None,
) -> PathState | None:
    """Create a PathState with BFS path to the given target.

    Returns None if no path can be found.
    """
    target_type, target_value = parse_target(target_str)

    if target_type in ("door", "stairs"):
        try:
            x_str, y_str = target_value.split(",")
            tx, ty = int(x_str), int(y_str)
        except (ValueError, IndexError):
            return None
        nav_grid = _collision_with_npcs(collision, objects, player_x, player_y)
        nav_grid = _collision_avoiding_warps(nav_grid, player_x, player_y, allow_pos=(tx, ty))
        path = path_to_target_tile(nav_grid, player_x, player_y, tx, ty)
        if path is None:
            return None
        # Doors often need one extra step to walk through the warp.
        # Stairs warp immediately on contact, so stop on the tile.
        if target_type == "door" and path:
            path.append(path[-1])
        state = PathState(
            target_type=target_type,
            target_id=target_str,
            target_pos=(tx, ty),
            path=path,
        )
        return state

    elif target_type == "npc":
        try:
            local_id = int(target_value)
        except ValueError:
            return None
        if not objects:
            return None
        # Find the NPC object
        target_obj = None
        for obj in objects:
            if obj.get("local_id") == local_id:
                target_obj = obj
                break
        if not target_obj:
            return None
        # Path to adjacent tile (handles counter-talk too)
        nav_grid = _collision_with_npcs(collision, objects, player_x, player_y,
                                         exclude_npc_id=local_id)
        nav_grid = _collision_avoiding_warps(nav_grid, player_x, player_y)
        path, _, pos = path_to_adjacent_object(
            nav_grid, player_x, player_y, objects,
            match_fn=lambda o: o.get("local_id") == local_id,
        )
        if path is None:
            return None
        state = PathState(
            target_type="npc",
            target_id=target_str,
            target_pos=pos,
            target_npc_id=local_id,
            path=path,
        )
        return state

    elif target_type == "edge":
        direction = target_value.lower()
        # Find path to nearest edge tile on the specified side
        grid_w, grid_h, grid_rows = collision

        def edge_predicate(x, y, tile, gw, gh, rows):
            if not is_passable(tile):
                return False
            if direction == "north" and y == 0:
                return True
            if direction == "south" and y == gh - 1:
                return True
            if direction == "west" and x == 0:
                return True
            if direction == "east" and x == gw - 1:
                return True
            return False

        from exploration import path_to_nearest_tile
        nav_grid = _collision_with_npcs(collision, objects, player_x, player_y)
        nav_grid = _collision_avoiding_warps(nav_grid, player_x, player_y)
        path, pos = path_to_nearest_tile(
            nav_grid, player_x, player_y, edge_predicate, max_steps=300
        )
        if path is None:
            def exit_predicate(x, y, tile, *_):
                if tile not in ("D", "S"):
                    return False
                if direction == "north":
                    return y < player_y
                if direction == "south":
                    return y > player_y
                if direction == "west":
                    return x < player_x
                if direction == "east":
                    return x > player_x
                return True

            path, pos = path_to_nearest_tile(
                nav_grid, player_x, player_y, exit_predicate, max_steps=300
            )
            if path is not None and pos is not None:
                state = PathState(
                    target_type="stairs" if collision[2][pos[1]][pos[0]] == "S" else "door",
                    target_id=f"{'stairs' if collision[2][pos[1]][pos[0]] == 'S' else 'door'}:{pos[0]},{pos[1]}",
                    target_pos=pos,
                    path=path,
                )
                return state

            path, pos = path_to_nearest_tile(
                nav_grid,
                player_x,
                player_y,
                lambda x, y, tile, *_: tile in ("D", "S"),
                max_steps=300,
            )
            if path is not None and pos is not None:
                state = PathState(
                    target_type="stairs" if collision[2][pos[1]][pos[0]] == "S" else "door",
                    target_id=f"{'stairs' if collision[2][pos[1]][pos[0]] == 'S' else 'door'}:{pos[0]},{pos[1]}",
                    target_pos=pos,
                    path=path,
                )
                return state
        if path is None:
            return None
        state = PathState(
            target_type="edge",
            target_id=target_str,
            target_pos=pos,
            path=path,
        )
        return state

    return None


def get_arrival_action(path_state: PathState,
                       player_x: int, player_y: int,
                       collision: tuple[int, int, list[str]] | None) -> str | None:
    """Return the button to press when we've arrived at the target.

    For doors: step onto the door tile (walk in the direction of the door).
    For NPCs: face the NPC and press A.
    For edges: step off the map edge.
    """
    if not path_state.target_pos:
        return None

    tx, ty = path_state.target_pos

    if path_state.target_type == "door":
        # We should be adjacent or on the door — face it and walk onto it
        direction = facing_direction((player_x, player_y), (tx, ty))
        if direction:
            return direction
        # Already on the door tile — try all directions to trigger the warp
        for d in ["Down", "Up", "Left", "Right"]:
            return d


    elif path_state.target_type == "stairs":
        direction = facing_direction((player_x, player_y), (tx, ty))
        if direction:
            return direction
    elif path_state.target_type == "npc":
        # Face the NPC and press A
        return "A"

    elif path_state.target_type == "edge":
        if collision:
            action = edge_exit_action((player_x, player_y), collision)
            if action:
                return action
        # Fallback: walk in the edge direction
        dir_map = {"north": "Up", "south": "Down", "east": "Right", "west": "Left"}
        edge_dir = path_state.target_id.split(":", 1)[1] if path_state.target_id else ""
        return dir_map.get(edge_dir.lower(), "Down")

    return None
