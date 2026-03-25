"""PPO (Proximal Policy Optimization) trainer.

Standard PPO with clipped objective, value loss, and entropy bonus.
Entropy coefficient decays with overall curriculum mastery.
"""

import torch
import torch.nn as nn
from pokemon_policy import PokemonPolicy
from replay_buffer import ReplayBuffer


class PPOTrainer:
    """Trains the actor-critic policy using PPO."""

    def __init__(
        self,
        policy: PokemonPolicy,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        batch_size: int = 64,
    ):
        self.policy = policy
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        self.total_updates = 0

    def update(
        self, buffer: ReplayBuffer, last_value: float = 0.0, mastery: float = 0.0
    ) -> dict:
        """Run PPO update from collected experience.

        Args:
            buffer: replay buffer with rollout data
            last_value: bootstrap value for GAE
            mastery: 0-1 overall curriculum mastery (decays entropy)

        Returns:
            dict with training stats
        """
        if len(buffer) == 0:
            return {}

        # Decay entropy with mastery
        entropy_coef = max(0.001, self.entropy_coef * (1.0 - mastery))

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for epoch in range(self.update_epochs):
            batches = buffer.get_batches(self.batch_size, last_value)

            for batch in batches:
                states = batch["states"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Evaluate current policy on collected experience
                new_log_probs, values, entropy = self.policy.evaluate(states, actions)

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, returns)

                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = policy_loss + self.value_coef * value_loss + entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        self.total_updates += 1

        if num_updates == 0:
            return {}

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "entropy_coef": entropy_coef,
            "num_updates": num_updates,
            "total_updates": self.total_updates,
        }

    def save(self, path: str):
        """Save policy weights and optimizer state."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_updates": self.total_updates,
        }, path)

    def load(self, path: str):
        """Load policy weights and optimizer state."""
        checkpoint = torch.load(path, weights_only=True)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_updates = checkpoint.get("total_updates", 0)
