"""Rainbow DQN: Combining Improvements in Deep Reinforcement Learning.

Integrates:
    1. Double Q-learning (van Hasselt et al., 2016)
    2. Prioritized Experience Replay (Schaul et al., 2016)
    3. Dueling Network Architecture (Wang et al., 2016)
    4. Multi-step Bootstrap Returns (Sutton, 1988)
    5. Distributional RL (Categorical / C51, Bellemare et al., 2017)
    6. Noisy Nets for Exploration (Fortunato et al., 2018)

Reference:
    Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning",
    AAAI 2018. https://arxiv.org/abs/1710.02298
"""
from __future__ import annotations

import math
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OBS_DIM = 23
N_ACTIONS = 8


class NoisyLinear(nn.Module):
    """Factorized Gaussian Noisy Linear Layer for exploration without epsilon-greedy."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowNetwork(nn.Module):
    """Categorical (C51) Dueling Noisy Network."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden_dim: int = 128,
        num_atoms: int = 51,
        v_min: float = -2.0,
        v_max: float = 16.0
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Fixed atom support
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Dueling streams: Value stream V(s) -> (num_atoms,)
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, num_atoms)
        )

        # Dueling streams: Advantage stream A(s, a) -> (n_actions * num_atoms,)
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, n_actions * num_atoms)
        )

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns action probabilities: (B, n_actions, num_atoms)."""
        feat = self.feature_net(x)

        val = self.value_stream(feat).view(-1, 1, self.num_atoms)
        adv = self.advantage_stream(feat).view(-1, self.n_actions, self.num_atoms)

        # Dueling combination
        q_atoms = val + adv - adv.mean(dim=1, keepdim=True)
        prob = F.softmax(q_atoms, dim=-1)
        return prob.clamp(min=1e-8)

    @torch.no_grad()
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Expected Q-values: (B, n_actions)."""
        prob = self.forward(x)  # (B, n_actions, num_atoms)
        return (prob * self.support).sum(dim=-1)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) with proportional prioritization."""

    def __init__(self, capacity: int = 50000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        max_p = self.priorities.max() if self.buffer else 1.0
        if max_p == 0.0:
            max_p = 1.0

        item = (
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item

        self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4, device: torch.device = torch.device("cpu")):
        N = len(self.buffer)
        prios = self.priorities[:N]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(N, batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.int64, device=device),
            torch.as_tensor(rewards, dtype=torch.float32, device=device),
            torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(dones, dtype=torch.float32, device=device),
            indices,
            torch.as_tensor(weights, dtype=torch.float32, device=device)
        )

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = float(abs(err) + 1e-6)

    def __len__(self) -> int:
        return len(self.buffer)


class NStepBuffer:
    """Accumulates n-step discounted transitions before pushing to main replay buffer."""

    def __init__(self, n_step: int = 3, gamma: float = 0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)

    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) < self.n_step:
            return None
        return self._get_transition()

    def flush(self):
        out = []
        while self.buffer:
            out.append(self._get_transition())
            self.buffer.popleft()
        return out

    def _get_transition(self):
        s0, a0 = self.buffer[0][0], self.buffer[0][1]
        R = 0.0
        for i, (_, _, r, sn, d) in enumerate(self.buffer):
            R += (self.gamma ** i) * r
            if d:
                return (s0, a0, R, sn, True)
        return (s0, a0, R, self.buffer[-1][3], self.buffer[-1][4])


class RainbowAgent:
    """Rainbow DQN agent orchestrating NoisyNet, PER, C51 projection and Double-DQN."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden_dim: int = 128,
        num_atoms: int = 51,
        v_min: float = -2.0,
        v_max: float = 16.0,
        lr: float = 1e-4,
        gamma: float = 0.99,
        n_step: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.gamma = gamma
        self.n_step = n_step

        self.online_net = RainbowNetwork(obs_dim, n_actions, hidden_dim, num_atoms, v_min, v_max).to(self.device)
        self.target_net = RainbowNetwork(obs_dim, n_actions, hidden_dim, num_atoms, v_min, v_max).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

    def select_action(self, obs: np.ndarray) -> int:
        """Selects action greedily according to expected value of NoisyNet."""
        with torch.no_grad():
            s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.online_net.get_q_values(s)
            return int(q_vals.argmax(dim=-1).item())

    def update(self, replay_buffer: PrioritizedReplayBuffer, batch_size: int = 64, beta: float = 0.4) -> float:
        """Executes one step of Categorical Distributional Projection loss with PER."""
        if len(replay_buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(
            batch_size, beta=beta, device=self.device
        )

        # 1. Double DQN action selection for next state
        with torch.no_grad():
            next_q = self.online_net.get_q_values(next_states)
            next_action = next_q.argmax(dim=-1)  # (B,)

            # Categorical distribution from target net
            next_prob = self.target_net(next_states)  # (B, n_actions, num_atoms)
            next_prob_a = next_prob[torch.arange(batch_size), next_action]  # (B, num_atoms)

            # Categorical projection: Tz = R + gamma^n * (1 - d) * z_j
            gamma_n = self.gamma ** self.n_step
            Tz = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * gamma_n * self.online_net.support.unsqueeze(0)
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)

            b = (Tz - self.v_min) / self.online_net.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            # Handle edge cases where l == u
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1

            # Distribute probability
            m = torch.zeros(batch_size, self.num_atoms, device=self.device)
            offset = (torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).long().unsqueeze(1).expand(batch_size, self.num_atoms).to(self.device))

            m.view(-1).index_add_(0, (l + offset).view(-1), (next_prob_a * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (next_prob_a * (b - l.float())).view(-1))

        # 2. Current predicted distribution
        current_prob = self.online_net(states)  # (B, n_actions, num_atoms)
        current_prob_a = current_prob[torch.arange(batch_size), actions]  # (B, num_atoms)

        # Cross entropy loss: - sum m * log(p)
        elementwise_loss = -(m * current_prob_a.log()).sum(dim=-1)

        # Weighted by PER importance weights
        loss = (elementwise_loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities with TD errors
        td_errors = elementwise_loss.detach().cpu().numpy()
        replay_buffer.update_priorities(indices, td_errors)

        # Reset noise for next step
        self.online_net.reset_noise()
        self.target_net.reset_noise()

        return float(loss.item())

    def update_target_network(self, tau: float = 0.005):
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
