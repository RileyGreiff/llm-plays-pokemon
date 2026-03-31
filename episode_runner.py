"""Single episode execution — the core RL loop.

RL policy handles navigation. Battles and dialogue are auto-advanced with A
(spams Tackle / first move). No LLM dependency during training.
"""

import time
import emulator
from state_encoder import encode_navigation_state
from reward_function import compute_reward, EpisodeTracker
from replay_buffer import ReplayBuffer
from pokemon_policy import PokemonPolicy, ACTION_MAP
from story_flags import FLAG_BY_ID

# Minimum time between steps to avoid overwhelming the emulator
MIN_STEP_INTERVAL = 0.05  # 50ms


def run_episode(
    policy: PokemonPolicy,
    flag_id: str,
    buffer: ReplayBuffer,
    mastery: float = 0.0,
    verbose: bool = True,
) -> dict:
    """Run a single episode for one story flag.

    Args:
        policy: the RL navigation policy
        flag_id: which story flag to train
        buffer: replay buffer to collect experience into
        mastery: current mastery level for this flag (0-1)
        verbose: print step-by-step info

    Returns:
        dict with episode stats: {success, steps, reward_total, flag_id}
    """
    flag = FLAG_BY_ID[flag_id]
    tracker = EpisodeTracker()

    # Read initial state
    state = emulator.read_game_state()
    tracker.visited_maps.add(state.get("map_id", 0))

    if verbose:
        print(f"\n{'='*60}")
        print(f"EPISODE START: {flag.name} (mastery={mastery:.2f})")
        print(f"  Map: {state['map_name']} ({state['player_x']}, {state['player_y']})")
        print(f"  Party: {state['party_count']} pokemon, Badges: {state['badges']}")
        print(f"  Max steps: {flag.max_steps}")
        print(f"{'='*60}")

    # Check if flag is already completed (skip episode)
    if flag.check(state):
        if verbose:
            print(f"  Flag already completed! Skipping.")
        return {"success": True, "steps": 0, "reward_total": 0.0, "flag_id": flag_id, "skipped": True}

    done = False
    step = 0
    idle_steps = 0  # counts battle/dialogue steps to prevent infinite loops
    MAX_IDLE_STEPS = 500

    while not done and step < flag.max_steps:
        step_start = time.time()

        # Battles and dialogue: just spam A (uses Tackle / advances text)
        if state.get("in_battle", False) or state.get("in_dialogue", False):
            emulator.press_button("A", frames=16)
            state = emulator.read_game_state()
            idle_steps += 1
            if idle_steps >= MAX_IDLE_STEPS:
                if verbose:
                    print(f"  Stuck in battle/dialogue for {idle_steps} steps, aborting")
                break
            continue
        idle_steps = 0

        # RL policy handles navigation
        state_tensor = encode_navigation_state(
            state, flag_id, step, flag.max_steps
        )
        action, log_prob, value = policy.act(state_tensor)

        # Execute action
        button = ACTION_MAP[action]
        before_state = state
        emulator.press_button(button, frames=16)
        state = emulator.read_game_state()

        # Compute reward
        reward, done = compute_reward(
            before=before_state,
            after=state,
            flag_check=flag.check,
            flag_id=flag_id,
            tracker=tracker,
            mastery=mastery,
            max_steps=flag.max_steps,
        )

        # Store experience
        buffer.add(state_tensor, action, reward, value, log_prob, done)

        if verbose and step % 50 == 0:
            print(f"  Step {step:4d} | {state['map_name']:25s} | "
                  f"({state['player_x']:3d},{state['player_y']:3d}) | "
                  f"r={reward:+6.1f} | tiles={len(tracker.visited_tiles)}")

        step += 1

        # Rate limit
        elapsed = time.time() - step_start
        if elapsed < MIN_STEP_INTERVAL:
            time.sleep(MIN_STEP_INTERVAL - elapsed)

    # Episode finished
    success = flag.check(state)

    if verbose:
        result = "SUCCESS" if success else "FAILED"
        print(f"\n  EPISODE {result}: {flag.name}")
        print(f"  Steps: {step}, Reward: {tracker.reward_total:.1f}, "
              f"Tiles explored: {len(tracker.visited_tiles)}")

    return {
        "success": success,
        "steps": step,
        "reward_total": tracker.reward_total,
        "flag_id": flag_id,
        "tiles_explored": len(tracker.visited_tiles),
        "skipped": False,
    }


# --- LLM battle handling (disabled for training, re-enable later) ---
# To use: set USE_LLM_BATTLES = True and uncomment the import
# from claude_client import get_action
USE_LLM_BATTLES = False

def _handle_battle_llm(state: dict, verbose: bool = True):
    """Use Claude LLM to handle a battle turn.

    Currently disabled — battles use spam-A (Tackle) during RL training.
    To re-enable: set USE_LLM_BATTLES = True and uncomment the import above.
    Then replace the spam-A battle block in run_episode with:
        if state.get("in_battle", False):
            _handle_battle_llm(state, verbose)
            state = emulator.read_game_state()
            continue
    """
    from claude_client import get_action  # lazy import to avoid dependency when disabled
    try:
        if verbose:
            enemy = state.get("enemy_species", 0)
            print(f"  [battle] Enemy species={enemy}")

        parsed, usage = get_action(
            game_state=state,
            recent_actions=[],
            progress_summary="RL training episode — handle battle",
            force_sonnet=state.get("in_battle", False),
        )
        if parsed and "action" in parsed:
            button = parsed["action"]
            valid_buttons = {"A", "B", "Up", "Down", "Left", "Right"}
            if button in valid_buttons:
                emulator.press_button(button, frames=16)
                return
        emulator.press_button("A", frames=16)
    except Exception as e:
        if verbose:
            print(f"  [battle] LLM error: {e}, pressing A")
        emulator.press_button("A", frames=16)
