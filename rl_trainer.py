"""Central RL Trainer — reads experience from SQLite, runs PPO updates, saves weights.

Runs as a separate process from workers. Polls the database for new experience
and triggers PPO updates when enough steps have accumulated.

Usage:
    python rl_trainer.py
    python rl_trainer.py --update-threshold 1024
"""

import argparse
import os
import time
import json
import torch

from pokemon_policy import PokemonPolicy
from ppo_trainer import PPOTrainer
from replay_buffer import ReplayBuffer
from state_encoder import get_state_size
from story_flags import FLAG_ORDER, FLAG_BY_ID
import rl_database as db

# Paths
SAVE_DIR = "rl_checkpoints"
POLICY_PATH = os.path.join(SAVE_DIR, "policy.pt")

# How often to poll for new experience (seconds)
POLL_INTERVAL = 5.0


def run_trainer(
    update_threshold: int = 2048,
    lr: float = 3e-4,
    verbose: bool = True,
):
    """Main trainer loop — poll DB, run PPO when enough experience, save weights."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    state_size = get_state_size()
    policy = PokemonPolicy(state_size=state_size)
    trainer = PPOTrainer(policy=policy, lr=lr)

    # Load existing weights
    if os.path.exists(POLICY_PATH):
        print("[Trainer] Loading existing policy weights...")
        trainer.load(POLICY_PATH)

    # Database
    db.init_db()
    conn = db.get_connection()

    total_steps_trained = 0
    update_count = 0

    print(f"[Trainer] Ready. Polling for experience (threshold={update_threshold} steps)...")

    try:
        while True:
            # Check how much unconsumed experience is available
            available = db.count_unconsumed(conn)

            if available < update_threshold:
                if verbose and available > 0:
                    print(f"[Trainer] {available}/{update_threshold} steps buffered, waiting...")
                time.sleep(POLL_INTERVAL)
                continue

            # Read experience from DB
            ids, experiences = db.read_unconsumed_experience(conn, state_size, limit=update_threshold)
            if not experiences:
                time.sleep(POLL_INTERVAL)
                continue

            print(f"\n[Trainer] Running PPO update on {len(experiences)} steps...")

            # Load into replay buffer
            buffer = ReplayBuffer()
            for exp in experiences:
                buffer.add(
                    exp["state"], exp["action"], exp["reward"],
                    exp["value"], exp["log_prob"], exp["done"],
                )

            # Compute mastery for entropy decay
            mastery = _compute_overall_mastery(conn)

            # PPO update
            stats = trainer.update(buffer, last_value=0.0, mastery=mastery)

            if stats:
                update_count += 1
                total_steps_trained += len(experiences)
                print(f"[Trainer] Update #{update_count} | "
                      f"policy_loss={stats['policy_loss']:.4f} | "
                      f"value_loss={stats['value_loss']:.4f} | "
                      f"entropy={stats['entropy']:.4f} | "
                      f"ent_coef={stats['entropy_coef']:.4f} | "
                      f"total_steps={total_steps_trained}")

            # Mark experience as consumed
            db.mark_consumed(conn, ids)

            # Save updated weights (workers will reload)
            trainer.save(POLICY_PATH)
            print(f"[Trainer] Weights saved to {POLICY_PATH}")

            # Periodic cleanup
            if update_count % 10 == 0:
                db.cleanup_old_experience(conn)

            # Print curriculum status
            if update_count % 5 == 0:
                _print_curriculum_status(conn)

    except KeyboardInterrupt:
        print("\n[Trainer] Shutting down.")
        trainer.save(POLICY_PATH)
        print("[Trainer] Final weights saved.")
    finally:
        conn.close()


def _compute_overall_mastery(conn) -> float:
    """Compute overall mastery from DB episode stats."""
    mastered = 0
    for flag_id in FLAG_ORDER:
        stats = db.get_recent_episode_stats(conn, flag_id, limit=10)
        if len(stats) >= 10:
            sr = sum(1 for s in stats if s["success"]) / len(stats)
            if sr >= 0.8:
                mastered += 1
    return mastered / max(len(FLAG_ORDER), 1)


def _print_curriculum_status(conn):
    """Print training progress across all flags."""
    print("\n--- Curriculum Status ---")
    for flag_id in FLAG_ORDER:
        flag = FLAG_BY_ID[flag_id]
        stats = db.get_recent_episode_stats(conn, flag_id, limit=10)
        total = len(db.get_recent_episode_stats(conn, flag_id, limit=1000))
        if not stats:
            sr = 0.0
            avg_steps = "N/A"
        else:
            sr = sum(1 for s in stats if s["success"]) / len(stats)
            successful = [s["steps"] for s in stats if s["success"]]
            avg_steps = f"{sum(successful)/len(successful):.0f}" if successful else "N/A"

        mastered = "YES" if sr >= 0.8 and len(stats) >= 10 else "NO "
        print(f"  {mastered} | {flag.name:30s} | total_ep={total:4d} | "
              f"sr={sr:.0%} | avg_steps={avg_steps}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Central RL Trainer")
    parser.add_argument("--update-threshold", type=int, default=256,
                        help="Min experience steps before PPO update (default 256)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_trainer(
        update_threshold=args.update_threshold,
        lr=args.lr,
        verbose=not args.quiet,
    )
