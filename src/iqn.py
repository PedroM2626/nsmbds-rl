"""Implicit Quantile Networks (IQN) for Distributional Reinforcement Learning.

Reference:
    Dabney et al., "Implicit Quantile Networks for Distributional Reinforcement Learning",
    ICML 2018. https://arxiv.org/abs/1806.06923
"""
from __future__ import annotations

import math
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OBS_DIM = 23
N_ACTIONS = 8


class CosineQuantileEmbedding(nn.Module):
    """Embeds scalar quantiles tau in (0, 1) into an embedding vector via cosine basis."""

    def __init__(self, num_cosines: int = 64, embedding_dim: int = 128):
        super().__init__()
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(num_cosines, embedding_dim),
            nn.ReLU()
        )
        # Register integer frequencies [0, 1, ..., num_cosines - 1] as non-trainable buffer
        self.register_buffer("freqs", torch.arange(num_cosines, dtype=torch.float32) * math.pi)

    def forward(self, taus: torch.Tensor) -> torch.Tensor:
        """taus: (B, N) or (B, N, 1) -> output: (B, N, embedding_dim)."""
        if taus.dim() == 2:
            taus = taus.unsqueeze(-1)  # (B, N, 1)
        # taus * freqs: (B, N, num_cosines)
        cos_features = torch.cos(taus * self.freqs)
        return self.net(cos_features)


class IQNNetwork(nn.Module):
    """Implicit Quantile Network predicting return distributions Z_tau(s, a)."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden_dim: int = 128,
        num_cosines: int = 64
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        # State representation network psi(s)
        self.psi = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Quantile embedding network phi(tau)
        self.phi = CosineQuantileEmbedding(num_cosines=num_cosines, embedding_dim=hidden_dim)

        # Output head processing element-wise product psi(s) * phi(tau)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, states: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """Computes quantiles Z_tau(s, a).

        Args:
            states: (B, obs_dim)
            taus: (B, N)
        Returns:
            quantiles: (B, N, n_actions)
        """
        batch_size = states.size(0)
        num_quantiles = taus.size(1)

        # State representation: (B, hidden_dim) -> (B, 1, hidden_dim)
        psi_s = self.psi(states).unsqueeze(1)

        # Quantile representation: (B, N, hidden_dim)
        phi_tau = self.phi(taus)

        # Multiplicative interaction: (B, N, hidden_dim)
        features = psi_s * phi_tau

        # Quantile values: (B, N, n_actions)
        quantiles = self.head(features)
        return quantiles

    @torch.no_grad()
    def get_q_values(self, states: torch.Tensor, num_quantiles: int = 32) -> torch.Tensor:
        """Approximates expected Q(s, a) = E_tau[Z_tau(s, a)] by sampling quantiles."""
        batch_size = states.size(0)
        taus = torch.rand(batch_size, num_quantiles, device=states.device)
        quantiles = self.forward(states, taus)  # (B, N, n_actions)
        return quantiles.mean(dim=1)  # (B, n_actions)


class ReplayBuffer:
    """Experience replay buffer for off-policy transition storage."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        ))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(-1),
            torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1),
            torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)
        )

    def __len__(self) -> int:
        return len(self.buffer)


def compute_quantile_huber_loss(
    current_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0
) -> torch.Tensor:
    """Computes the distributional quantile Huber loss between predicted and target quantiles.

    Args:
        current_quantiles: (B, N, 1) - Z_tau_i(s, a)
        target_quantiles: (B, 1, N') - T_j = r + gamma * Z_tau'_j(s', a*)
        taus: (B, N, 1) - quantiles used for current state
        kappa: threshold for Huber loss (default 1.0)
    Returns:
        Scalar loss
    """
    # Pairwise difference: (B, N, N')
    diff = target_quantiles - current_quantiles

    # Huber loss
    abs_diff = diff.abs()
    huber_loss = torch.where(
        abs_diff <= kappa,
        0.5 * (diff ** 2),
        kappa * (abs_diff - 0.5 * kappa)
    )

    # Quantile weighting: |tau - I(diff < 0)|
    with torch.no_grad():
        weight = (taus - (diff < 0).float()).abs()

    # Elementwise loss: (B, N, N')
    loss = (weight * huber_loss).mean(dim=2).sum(dim=1).mean()
    return loss


class IQNAgent:
    """Complete IQN Agent orchestrating action selection, network updates, and targets."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden_dim: int = 128,
        lr: float = 1e-4,
        gamma: float = 0.99,
        num_quantiles_n: int = 32,
        num_quantiles_n_prime: int = 32,
        num_quantiles_k: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.num_quantiles_n = num_quantiles_n
        self.num_quantiles_n_prime = num_quantiles_n_prime
        self.num_quantiles_k = num_quantiles_k

        self.online_net = IQNNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target_net = IQNNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

    def select_action(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """Selects action via epsilon-greedy policy over expected Q-values."""
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)

        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.online_net.get_q_values(state, num_quantiles=self.num_quantiles_k)
        return int(q_values.argmax(dim=-1).item())

    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 64) -> float:
        """Executes one step of quantile regression gradient descent."""
        if len(replay_buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, self.device)

        # 1. Sample N quantiles tau_i for current state: (B, N)
        taus = torch.rand(batch_size, self.num_quantiles_n, device=self.device)
        # Predict current quantiles: (B, N, n_actions)
        current_quantiles_all = self.online_net(states, taus)
        # Select taken action: (B, N, 1)
        action_indices = actions.unsqueeze(1).expand(-1, self.num_quantiles_n, -1)
        current_quantiles = current_quantiles_all.gather(dim=2, index=action_indices)

        # 2. Double DQN action selection for next state using online net: (B,)
        with torch.no_grad():
            next_q_values = self.online_net.get_q_values(next_states, num_quantiles=self.num_quantiles_k)
            best_next_actions = next_q_values.argmax(dim=-1, keepdim=True)  # (B, 1)

            # Sample N' quantiles tau'_j for target network: (B, N')
            target_taus = torch.rand(batch_size, self.num_quantiles_n_prime, device=self.device)
            target_quantiles_all = self.target_net(next_states, target_taus)  # (B, N', n_actions)
            best_action_indices = best_next_actions.unsqueeze(1).expand(-1, self.num_quantiles_n_prime, -1)
            next_quantiles = target_quantiles_all.gather(dim=2, index=best_action_indices)  # (B, N', 1)

            # Bellman target distribution: T_j = r + gamma * (1 - done) * Z_tau'_j
            # Target quantiles reshaped to (B, 1, N')
            targets = rewards.unsqueeze(1) + self.gamma * (1.0 - dones.unsqueeze(1)) * next_quantiles.permute(0, 2, 1)

        # 3. Compute Quantile Huber Loss
        loss = compute_quantile_huber_loss(
            current_quantiles,
            targets,
            taus.unsqueeze(-1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self, tau: float = 0.005):
        """Soft target network update: theta_target = tau * theta_online + (1 - tau) * theta_target."""
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
