"""RL Agent — main entry point for PPO-based Pokemon navigation.

Wires together: curriculum scheduler, episode runner, PPO trainer.
Training loop: select flag → run episode → PPO update → repeat.
"""

import os
import sys
import time
import json
import torch

from pokemon_policy import PokemonPolicy
from ppo_trainer import PPOTrainer
from replay_buffer import ReplayBuffer
from curriculum_scheduler import CurriculumScheduler
from episode_runner import run_episode
from state_encoder import get_state_size
from reward_function import get_best_steps, set_best_steps
from story_flags import FLAG_BY_ID

# Paths
SAVE_DIR = "rl_checkpoints"
POLICY_PATH = os.path.join(SAVE_DIR, "policy.pt")
CURRICULUM_PATH = os.path.join(SAVE_DIR, "curriculum.json")
BEST_STEPS_PATH = os.path.join(SAVE_DIR, "best_steps.json")

# Training hyperparameters
LEARNING_RATE = 3e-4
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.05
VALUE_COEF = 0.5
UPDATE_EPOCHS = 4
BATCH_SIZE = 64
GAMMA = 0.99
GAE_LAMBDA = 0.95
MIN_BUFFER_SIZE = 128  # minimum steps before PPO update


def main():
    print("=" * 60)
    print("  Pokemon FireRed RL Agent (PPO)")
    print("  Start → Brock curriculum")
    print("=" * 60)

    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Initialize components
    state_size = get_state_size()
    print(f"State size: {state_size}")

    policy = PokemonPolicy(state_size=state_size)
    trainer = PPOTrainer(
        policy=policy,
        lr=LEARNING_RATE,
        clip_epsilon=CLIP_EPSILON,
        entropy_coef=ENTROPY_COEF,
        value_coef=VALUE_COEF,
        update_epochs=UPDATE_EPOCHS,
        batch_size=BATCH_SIZE,
    )
    buffer = ReplayBuffer(gamma=GAMMA, gae_lambda=GAE_LAMBDA)
    curriculum = CurriculumScheduler(save_path=CURRICULUM_PATH)

    # Load existing state if available
    if os.path.exists(POLICY_PATH):
        print("Loading saved policy...")
        trainer.load(POLICY_PATH)
    if os.path.exists(CURRICULUM_PATH):
        print("Loading curriculum state...")
        curriculum.load()
    if os.path.exists(BEST_STEPS_PATH):
        with open(BEST_STEPS_PATH) as f:
            set_best_steps(json.load(f))

    print("\nCurriculum status:")
    print(curriculum.status_summary())
    print()

    episode_num = 0
    total_steps = 0

    try:
        while True:
            # Get next flag to train
            flag_id = curriculum.get_active_flag_id()
            if flag_id is None:
                print("\n*** ALL FLAGS MASTERED! Training complete. ***")
                break

            flag = FLAG_BY_ID[flag_id]
            mastery = curriculum.get_flag_mastery(flag_id)
            episode_num += 1

            print(f"\n--- Episode {episode_num} | Flag: {flag.name} | "
                  f"Mastery: {mastery:.2f} | Overall: {curriculum.overall_mastery():.2f} ---")

            # Run episode
            buffer.clear()
            result = run_episode(
                policy=policy,
                flag_id=flag_id,
                buffer=buffer,
                mastery=mastery,
                verbose=True,
            )

            if result.get("skipped"):
                # Flag was already completed in game state — report success and continue
                curriculum.report_episode(flag_id, True, 0)
                continue

            total_steps += result["steps"]

            # Report to curriculum
            curriculum.report_episode(flag_id, result["success"], result["steps"])

            # PPO update if enough experience collected
            if len(buffer) >= MIN_BUFFER_SIZE:
                overall_mastery = curriculum.overall_mastery()
                train_stats = trainer.update(buffer, last_value=0.0, mastery=overall_mastery)
                if train_stats:
                    print(f"  [PPO] policy_loss={train_stats['policy_loss']:.4f} "
                          f"value_loss={train_stats['value_loss']:.4f} "
                          f"entropy={train_stats['entropy']:.4f} "
                          f"ent_coef={train_stats['entropy_coef']:.4f}")
            else:
                print(f"  [PPO] Buffer too small ({len(buffer)} steps), skipping update")

            # Save periodically
            if episode_num % 10 == 0:
                _save_all(trainer, curriculum)

            # Status update
            if episode_num % 5 == 0:
                print(f"\n--- Status after {episode_num} episodes, {total_steps} total steps ---")
                print(curriculum.status_summary())

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    finally:
        print("\nSaving final state...")
        _save_all(trainer, curriculum)
        print("Done.")


def _save_all(trainer: PPOTrainer, curriculum: CurriculumScheduler):
    """Save all training state to disk."""
    trainer.save(POLICY_PATH)
    curriculum.save()
    with open(BEST_STEPS_PATH, "w") as f:
        json.dump(get_best_steps(), f, indent=2)
    print(f"  [save] Policy, curriculum, and best steps saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
