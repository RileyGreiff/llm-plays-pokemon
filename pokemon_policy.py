"""Actor-critic policy network for PPO navigation.

Small network: shared trunk with separate actor (action logits) and critic (value) heads.
"""

import torch
import torch.nn as nn
from state_encoder import get_state_size, NUM_ACTIONS


class PokemonPolicy(nn.Module):
    """PPO actor-critic for Pokemon navigation."""

    def __init__(self, state_size: int | None = None, hidden: int = 256):
        super().__init__()
        if state_size is None:
            state_size = get_state_size()

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )

        # Actor head — action logits
        self.actor = nn.Linear(hidden // 2, NUM_ACTIONS)

        # Critic head — state value
        self.critic = nn.Linear(hidden // 2, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            state: (batch, state_size) or (state_size,) tensor

        Returns:
            (action_logits, state_value) tuple
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        features = self.trunk(state)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    def act(self, state: torch.Tensor) -> tuple[int, float, float]:
        """Select action for a single state during rollout.

        Returns:
            (action, log_prob, value) tuple
        """
        with torch.no_grad():
            logits, value = self.forward(state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def evaluate(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update (batched).

        Args:
            states: (batch, state_size)
            actions: (batch,) int tensor

        Returns:
            (log_probs, values, entropy) tuple
        """
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy


# Action index to button mapping
ACTION_MAP = {
    0: "Up",
    1: "Down",
    2: "Left",
    3: "Right",
    4: "A",
    5: "B",
}
