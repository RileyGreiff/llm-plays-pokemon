"""Episode experience buffer for PPO training.

Stores transitions from rollouts. Computes GAE (Generalized Advantage Estimation)
for PPO updates. Buffer is cleared after each training update.
"""

import torch
import numpy as np


class ReplayBuffer:
    """Stores rollout experience and computes advantages."""

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear()

    def clear(self):
        """Reset buffer for new rollout."""
        self.states: list[torch.Tensor] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.log_probs: list[float] = []
        self.dones: list[bool] = []

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Add a single transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def compute_advantages(self, last_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns.

        Args:
            last_value: bootstrap value for non-terminal final state

        Returns:
            (advantages, returns) tensors of shape (buffer_size,)
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 0.0 if self.dones[t] else 1.0
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 0.0 if self.dones[t] else 1.0

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        values_arr = np.array(self.values, dtype=np.float32)
        returns = advantages + values_arr

        return torch.from_numpy(advantages), torch.from_numpy(returns)

    def get_batches(
        self, batch_size: int, last_value: float = 0.0
    ) -> list[dict[str, torch.Tensor]]:
        """Compute advantages and return shuffled mini-batches.

        Args:
            batch_size: size of each mini-batch
            last_value: bootstrap value for GAE

        Returns:
            List of dicts with keys: states, actions, old_log_probs, advantages, returns
        """
        advantages, returns = self.compute_advantages(last_value)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32)

        n = len(self.states)
        indices = np.arange(n)
        np.random.shuffle(indices)

        batches = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            batches.append({
                "states": states[idx],
                "actions": actions[idx],
                "old_log_probs": old_log_probs[idx],
                "advantages": advantages[idx],
                "returns": returns[idx],
            })

        return batches
