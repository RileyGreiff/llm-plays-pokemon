"""Encode game state dict into a normalized tensor for the policy network.

Output: ~310-dim float tensor containing:
- map bank one-hot (64 dims)
- map num one-hot (128 dims)
- x, y normalized (2 dims)
- active_flag one-hot (max 50 dims)
- episode_step normalized (1 dim)
- badges one-hot (8 dims)
- party_hp_pct (1 dim)
- party_count (1 dim)
- in_battle / in_dialogue (2 dims)
- facing one-hot (4 dims)
- local minimap (7x7 by default, flattened)
"""

import torch
import numpy as np
import emulator
from story_flags import FLAG_ORDER

# Dimensions
NUM_MAP_BANKS = 64
NUM_MAP_NUMS = 128
NUM_FLAGS = 50
NUM_ACTIONS = 6
NUM_FACING = 4

# Map coordinate ranges (GBA maps vary, normalize to 0-1)
MAX_COORD = 128
DEFAULT_MINIMAP_RADIUS = 3


def encode_state(
    game_state: dict,
    active_flag_id: str,
    episode_step: int,
    max_episode_steps: int,
    minimap: list[list[int]] | None = None,
    player_facing: int | None = None,
) -> torch.Tensor:
    """Convert game state to a flat tensor for the policy network.

    Args:
        game_state: dict from emulator.read_game_state()
        active_flag_id: current story flag being trained
        episode_step: current step in this episode
        max_episode_steps: max steps for timeout
        minimap: optional local tile grid (e.g. 9x9 around player)

    Returns:
        1D float tensor, size ~270 (exact size depends on minimap)
    """
    features = []

    # Encode bank/num separately so maps do not alias together.
    map_bank = min(max(game_state.get("map_bank", 0), 0), NUM_MAP_BANKS - 1)
    bank_onehot = np.zeros(NUM_MAP_BANKS, dtype=np.float32)
    bank_onehot[map_bank] = 1.0
    features.append(bank_onehot)

    map_num = min(max(game_state.get("map_num", 0), 0), NUM_MAP_NUMS - 1)
    num_onehot = np.zeros(NUM_MAP_NUMS, dtype=np.float32)
    num_onehot[map_num] = 1.0
    features.append(num_onehot)

    # Player position normalized (2 dims)
    x = min(game_state.get("player_x", 0) / MAX_COORD, 1.0)
    y = min(game_state.get("player_y", 0) / MAX_COORD, 1.0)
    features.append(np.array([x, y], dtype=np.float32))

    # Active flag one-hot (50 dims)
    flag_onehot = np.zeros(NUM_FLAGS, dtype=np.float32)
    if active_flag_id in FLAG_ORDER:
        flag_idx = FLAG_ORDER.index(active_flag_id)
        if flag_idx < NUM_FLAGS:
            flag_onehot[flag_idx] = 1.0
    features.append(flag_onehot)

    # Episode progress (1 dim) — urgency signal
    progress = min(episode_step / max(max_episode_steps, 1), 1.0)
    features.append(np.array([progress], dtype=np.float32))

    # Badges one-hot (8 dims)
    badges = game_state.get("badges", 0)
    badge_bits = np.array([(badges >> i) & 1 for i in range(8)], dtype=np.float32)
    features.append(badge_bits)

    # Party HP percentage (1 dim)
    party = game_state.get("party", [])
    if party:
        total_hp = sum(m.get("hp", 0) for m in party)
        total_max = sum(m.get("max_hp", 1) for m in party)
        hp_pct = total_hp / max(total_max, 1)
    else:
        hp_pct = 0.0
    features.append(np.array([hp_pct], dtype=np.float32))

    # Party count normalized (1 dim)
    party_count = min(game_state.get("party_count", 0) / 6.0, 1.0)
    features.append(np.array([party_count], dtype=np.float32))

    # In battle / in dialogue flags (2 dims)
    in_battle = 1.0 if game_state.get("in_battle", False) else 0.0
    in_dialogue = 1.0 if game_state.get("in_dialogue", False) else 0.0
    features.append(np.array([in_battle, in_dialogue], dtype=np.float32))

    # Player facing one-hot: 1=down, 2=up, 3=left, 4=right.
    facing_onehot = np.zeros(NUM_FACING, dtype=np.float32)
    if player_facing in (1, 2, 3, 4):
        facing_onehot[player_facing - 1] = 1.0
    features.append(facing_onehot)

    # Minimap (optional, flattened)
    if minimap is not None:
        flat = np.array(minimap, dtype=np.float32).flatten()
        # Normalize tile values: 0=walkable, 1=wall, 2=npc, 3=exit
        flat = flat / 3.0
        features.append(flat)

    return torch.from_numpy(np.concatenate(features))


def build_local_minimap(game_state: dict, radius: int = DEFAULT_MINIMAP_RADIUS) -> tuple[list[list[int]], int | None]:
    """Build a local tile-centered view around the player.

    Tile encoding:
    - 0: walkable / traversable
    - 1: blocked / unknown / off-map
    - 2: occupied by NPC or item object
    - 3: exit / doorway / stairs
    """
    size = radius * 2 + 1
    minimap = [[1 for _ in range(size)] for _ in range(size)]

    map_id = game_state.get("map_id", 0)
    collision = emulator.get_collision_grid(map_id)
    objects, player_facing = emulator.get_objects(game_state)

    player_x = game_state.get("player_x", 0)
    player_y = game_state.get("player_y", 0)

    if collision is not None:
        grid_w, grid_h, rows = collision
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                gx = player_x + dx
                gy = player_y + dy
                my = dy + radius
                mx = dx + radius

                if 0 <= gx < grid_w and 0 <= gy < grid_h:
                    tile = rows[gy][gx]
                    if tile == "1":
                        minimap[my][mx] = 1
                    elif tile in {"D", "S"}:
                        minimap[my][mx] = 3
                    else:
                        minimap[my][mx] = 0

    if objects:
        for obj in objects:
            ox = obj.get("x")
            oy = obj.get("y")
            if ox is None or oy is None:
                continue
            dx = ox - player_x
            dy = oy - player_y
            if -radius <= dx <= radius and -radius <= dy <= radius:
                minimap[dy + radius][dx + radius] = 2

    # Keep the player cell traversable rather than occupied.
    minimap[radius][radius] = 0
    return minimap, player_facing


def encode_navigation_state(
    game_state: dict,
    active_flag_id: str,
    episode_step: int,
    max_episode_steps: int,
    minimap_radius: int = DEFAULT_MINIMAP_RADIUS,
) -> torch.Tensor:
    """Encode game state with a local minimap and player-facing information."""
    minimap, player_facing = build_local_minimap(game_state, radius=minimap_radius)
    return encode_state(
        game_state=game_state,
        active_flag_id=active_flag_id,
        episode_step=episode_step,
        max_episode_steps=max_episode_steps,
        minimap=minimap,
        player_facing=player_facing,
    )


def get_state_size(
    minimap_size: int | None = None,
    minimap_radius: int = DEFAULT_MINIMAP_RADIUS,
) -> int:
    """Calculate total state tensor size.

    Args:
        minimap_size: total number of minimap cells; defaults to the standard
            square local minimap size.
    """
    if minimap_size is None:
        minimap_size = (minimap_radius * 2 + 1) ** 2
    base = NUM_MAP_BANKS + NUM_MAP_NUMS + 2 + NUM_FLAGS + 1 + 8 + 1 + 1 + 2 + NUM_FACING
    return base + minimap_size
