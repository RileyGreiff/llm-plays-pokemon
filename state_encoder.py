"""Encode game state dict into a normalized tensor for the policy network.

Output: ~270-dim float tensor containing:
- map_id one-hot (200 dims)
- x, y normalized (2 dims)
- active_flag one-hot (max 50 dims)
- episode_step normalized (1 dim)
- badges one-hot (8 dims)
- party_hp_pct (1 dim)
- minimap_flat (optional, variable)
"""

import torch
import numpy as np
from story_flags import FLAG_ORDER

# Dimensions
NUM_MAPS = 200
NUM_FLAGS = 50
NUM_ACTIONS = 6

# Map coordinate ranges (GBA maps vary, normalize to 0-1)
MAX_COORD = 128


def encode_state(
    game_state: dict,
    active_flag_id: str,
    episode_step: int,
    max_episode_steps: int,
    minimap: list[list[int]] | None = None,
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

    # Map ID one-hot (200 dims)
    map_id = game_state.get("map_id", 0) % NUM_MAPS
    map_onehot = np.zeros(NUM_MAPS, dtype=np.float32)
    map_onehot[map_id] = 1.0
    features.append(map_onehot)

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

    # Minimap (optional, flattened)
    if minimap is not None:
        flat = np.array(minimap, dtype=np.float32).flatten()
        # Normalize tile values: 0=walkable, 1=wall, 2=npc, 3=exit
        flat = flat / 3.0
        features.append(flat)

    return torch.from_numpy(np.concatenate(features))


def get_state_size(minimap_size: int = 0) -> int:
    """Calculate total state tensor size.

    Args:
        minimap_size: total number of minimap cells (e.g. 9*9=81), 0 if not used
    """
    base = NUM_MAPS + 2 + NUM_FLAGS + 1 + 8 + 1 + 1 + 2  # 265
    return base + minimap_size
