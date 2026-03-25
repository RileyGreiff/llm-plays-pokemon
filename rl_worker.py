"""RL Worker — runs episodes and writes experience to shared SQLite database.

Each worker connects to its own BizHawk instance via a unique bridge directory.
Workers periodically reload policy weights from disk (written by the central trainer).

Usage:
    python rl_worker.py --worker-id 0
    python rl_worker.py --worker-id 1
"""

import argparse
import os
import time
import torch

import emulator
from pokemon_policy import PokemonPolicy, ACTION_MAP
from state_encoder import encode_state, get_state_size
from reward_function import compute_reward, EpisodeTracker
from story_flags import FLAG_BY_ID, FLAG_ORDER
import rl_database as db

# How often to reload policy weights from disk (seconds)
WEIGHT_RELOAD_INTERVAL = 30.0

# Paths
SAVE_DIR = "rl_checkpoints"
POLICY_PATH = os.path.join(SAVE_DIR, "policy.pt")
SAVESTATE_DIR = "savestates"


def run_worker(worker_id: int, verbose: bool = True):
    """Main worker loop — run episodes forever, write experience to DB."""
    # Configure bridge for this worker
    bridge_dir = f"bridge_{worker_id}"
    emulator.set_bridge_dir(bridge_dir)
    print(f"[Worker {worker_id}] Bridge dir: {bridge_dir}")

    # Initialize policy
    state_size = get_state_size()
    policy = PokemonPolicy(state_size=state_size)

    # Load existing weights if available
    if os.path.exists(POLICY_PATH):
        _reload_weights(policy, worker_id)

    # Database connection
    db.init_db()
    conn = db.get_connection()

    last_reload = time.time()
    episode_num = 0

    try:
        while True:
            # Determine which flag to train
            flag_id = _get_active_flag(conn)
            if flag_id is None:
                print(f"[Worker {worker_id}] All flags mastered! Sleeping...")
                time.sleep(10)
                continue

            flag = FLAG_BY_ID[flag_id]
            episode_num += 1

            # Load savestate for this flag if available, otherwise use default
            savestate_path = _get_savestate_path(flag_id)
            if savestate_path:
                if not emulator.load_savestate(savestate_path):
                    print(f"[Worker {worker_id}] WARNING: Failed to load savestate {savestate_path}")
                else:
                    time.sleep(0.5)  # let emulator settle after load

            # Compute mastery from DB stats
            mastery = _get_flag_mastery(conn, flag_id)

            if verbose:
                print(f"\n[Worker {worker_id}] Episode {episode_num} | "
                      f"Flag: {flag.name} | Mastery: {mastery:.2f}")

            # Run episode and collect experience
            result, experiences = _run_episode_to_db(
                policy, flag_id, mastery, worker_id, verbose
            )

            # Write experience batch to DB
            if experiences:
                db.write_experience_batch(conn, worker_id, flag_id, experiences)

            # Write tile reward data for dashboard heatmap
            if result.get("tile_rewards"):
                db.write_tile_rewards_batch(conn, result["tile_rewards"])

            # Write episode stats
            db.write_episode_stats(
                conn, worker_id, flag_id,
                result["success"], result["steps"],
                result["reward_total"], result["tiles_explored"],
            )

            if verbose:
                status = "SUCCESS" if result["success"] else "FAILED"
                print(f"[Worker {worker_id}] {status} | Steps: {result['steps']} | "
                      f"Reward: {result['reward_total']:.1f} | "
                      f"Wrote {len(experiences)} experience rows")

            # Periodically reload policy weights
            if time.time() - last_reload > WEIGHT_RELOAD_INTERVAL:
                _reload_weights(policy, worker_id)
                last_reload = time.time()

    except KeyboardInterrupt:
        print(f"\n[Worker {worker_id}] Shutting down.")
    finally:
        conn.close()


def _run_episode_to_db(
    policy: PokemonPolicy,
    flag_id: str,
    mastery: float,
    worker_id: int,
    verbose: bool,
) -> tuple[dict, list[tuple]]:
    """Run one episode, return (result_dict, experience_tuples).

    Experience tuples: (step, state_tensor, action, reward, value, logprob, done)
    """
    flag = FLAG_BY_ID[flag_id]
    tracker = EpisodeTracker()
    experiences = []
    tile_rewards = []  # (map_id, map_name, x, y, reward, flag_id)

    state = emulator.read_game_state()

    # Seed starting map so agent doesn't get free new-map reward for spawn
    tracker.visited_maps.add(state.get("map_id", 0))

    # Check if flag already completed
    if flag.check(state):
        return {
            "success": True, "steps": 0, "reward_total": 0.0,
            "flag_id": flag_id, "tiles_explored": 0, "skipped": True,
        }, []

    done = False
    step = 0
    idle_steps = 0  # counts battle/dialogue steps to prevent infinite loops
    MAX_IDLE_STEPS = 500  # bail out if stuck in battle/dialogue too long

    while not done and step < flag.max_steps:
        step_start = time.time()

        # Battles and dialogue: spam A
        if state.get("in_battle", False) or state.get("in_dialogue", False):
            emulator.press_button("A", frames=16)
            state = emulator.read_game_state()
            idle_steps += 1
            if idle_steps >= MAX_IDLE_STEPS:
                if verbose:
                    print(f"  [W{worker_id}] Stuck in battle/dialogue for {idle_steps} steps, aborting")
                break
            continue
        idle_steps = 0

        # RL policy
        state_tensor = encode_state(state, flag_id, step, flag.max_steps)
        action, log_prob, value = policy.act(state_tensor)

        button = ACTION_MAP[action]
        before_state = state
        emulator.press_button(button, frames=16)
        state = emulator.read_game_state()

        reward, done = compute_reward(
            before=before_state, after=state,
            flag_check=flag.check, flag_id=flag_id,
            tracker=tracker, mastery=mastery, max_steps=flag.max_steps,
        )

        experiences.append((step, state_tensor, action, reward, value, log_prob, done))
        tile_rewards.append((
            state.get("map_id", 0), state.get("map_name", "unknown"),
            state.get("player_x", 0), state.get("player_y", 0),
            reward, value, flag_id,
        ))

        if verbose and step % 100 == 0:
            print(f"  [W{worker_id}] Step {step:4d} | {state['map_name']:20s} | "
                  f"tiles={len(tracker.visited_tiles)}")

        step += 1

        elapsed = time.time() - step_start
        if elapsed < 0.05:
            time.sleep(0.05 - elapsed)

    success = flag.check(state)
    return {
        "success": success,
        "steps": step,
        "reward_total": tracker.reward_total,
        "flag_id": flag_id,
        "tiles_explored": len(tracker.visited_tiles),
        "tile_rewards": tile_rewards,
        "skipped": False,
    }, experiences


def _reload_weights(policy: PokemonPolicy, worker_id: int):
    """Reload policy weights from disk."""
    if not os.path.exists(POLICY_PATH):
        return
    try:
        checkpoint = torch.load(POLICY_PATH, weights_only=True)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"[Worker {worker_id}] Reloaded policy weights")
    except Exception as e:
        print(f"[Worker {worker_id}] Weight reload failed: {e}")


def _get_savestate_path(flag_id: str) -> str | None:
    """Find savestate file for a flag. Checks for flag-specific then default."""
    # Flag-specific savestate (e.g., savestates/leave_house.State)
    flag_path = os.path.join(SAVESTATE_DIR, f"{flag_id}.State")
    if os.path.exists(flag_path):
        return os.path.abspath(flag_path)

    # Default savestate for game start
    default_path = os.path.join(SAVESTATE_DIR, "default.State")
    if os.path.exists(default_path):
        return os.path.abspath(default_path)

    return None


def _get_active_flag(conn) -> str | None:
    """Determine which flag to train based on DB episode stats."""
    for flag_id in FLAG_ORDER:
        flag = FLAG_BY_ID[flag_id]
        # Check prerequisite
        if flag.prerequisite and not _is_flag_mastered(conn, flag.prerequisite):
            return None  # prerequisite not met, nothing to train
        if not _is_flag_mastered(conn, flag_id):
            return flag_id
    return None


def _is_flag_mastered(conn, flag_id: str) -> bool:
    """Check if a flag is mastered based on DB stats."""
    stats = db.get_recent_episode_stats(conn, flag_id, limit=10)
    if len(stats) < 10:
        return False
    successes = sum(1 for s in stats if s["success"])
    if successes / len(stats) < 0.8:
        return False
    successful_steps = [s["steps"] for s in stats if s["success"]]
    if not successful_steps:
        return False
    # Use minimum successful steps as optimal baseline
    optimal = min(s["steps"] for s in db.get_recent_episode_stats(conn, flag_id, limit=100) if s["success"])
    avg = sum(successful_steps) / len(successful_steps)
    return avg < 2.0 * optimal


def _get_flag_mastery(conn, flag_id: str) -> float:
    """Get mastery score (0-1) from DB stats."""
    stats = db.get_recent_episode_stats(conn, flag_id, limit=10)
    if not stats:
        return 0.0
    return min(sum(1 for s in stats if s["success"]) / len(stats), 1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Worker")
    parser.add_argument("--worker-id", type=int, required=True, help="Worker instance ID (0, 1, 2, ...)")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    args = parser.parse_args()
    run_worker(args.worker_id, verbose=not args.quiet)
