import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MetaPolicy(nn.Module):
    """Daily meta-policy (PPO MLP) that outputs a single global price multiplier in [0, 2].

    One scalar α is broadcast to all regions each day. The meta observation is fully
    aggregated (no per-region signal), so per-region outputs are unjustified and waste
    sample efficiency given only 7 transitions per PPO update.

    Operates at day frequency: one select_action() call per day, one store_reward() call
    per day, and one update() call per episode.
    """

    def __init__(self, obs_dim, hidden_dim=128, lr=3e-4, gamma=0.99,
                 clip_eps=0.2, n_ppo_epochs=4, device='cpu',
                 clamped_buffer=False, zero_obs=False):
        super().__init__()
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.n_ppo_epochs = n_ppo_epochs
        self.device = device
        self.zero_obs = zero_obs

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        # Actor: mean ∈ (0, 2) via 2*sigmoid; single learnable log_std
        self.actor_head = nn.Linear(hidden_dim, 1)
        self.critic_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.6931]))  # std ≈ 0.5 initially

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.clamped_buffer = clamped_buffer

        self.obs_buf = []
        self.act_buf = []
        self.logp_buf = []
        self.val_buf = []
        self.rew_buf = []

        self.to(device)

    def _forward(self, obs_t):
        # Ablation: blind the meta-policy by feeding a constant zero observation.
        # Applied at every forward (action selection, PPO recompute, and the Picard
        # solver's mp._forward call), so all paths stay consistent.
        if self.zero_obs:
            obs_t = torch.zeros_like(obs_t)
        x = self.trunk(obs_t)
        mean = 2.0 * torch.sigmoid(self.actor_head(x))
        std = self.log_std.exp().clamp(1e-4, 1.0)
        value = self.critic_head(x).squeeze(-1)
        return mean, std, value

    def select_action(self, obs: np.ndarray) -> float:
        """Returns a single global price multiplier in [0, 2] and stores transition.

        clamped_buffer=True mirrors what Picard's _meta_forward does: stores the
        clamped action and evaluates logp at the clamped value. The default (False)
        stores the raw sample and evaluates logp there (the mathematically correct
        importance weight for PPO).
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std, value = self._forward(obs_t)
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.sample()

        if self.clamped_buffer:
            stored_action = raw_action.clamp(0.0, 2.0)
            logp = dist.log_prob(stored_action).sum(-1)
        else:
            stored_action = raw_action
            logp = dist.log_prob(raw_action).sum(-1)

        self.obs_buf.append(obs_t)
        self.act_buf.append(stored_action)
        self.logp_buf.append(logp)
        self.val_buf.append(value)

        return float(raw_action.clamp(0.0, 2.0).squeeze().item())

    def store_reward(self, reward: float):
        self.rew_buf.append(float(reward))

    def append_transition(self, obs: np.ndarray, act: np.ndarray,
                          logp: float, value: float, reward: float):
        """Explicitly append a pre-computed transition to the PPO buffers.

        Used by PicardSolver.commit() to populate buffers from converged
        day results instead of going through select_action() + store_reward().
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        self.obs_buf.append(obs_t)
        self.act_buf.append(torch.FloatTensor(act).unsqueeze(0).to(self.device))
        self.logp_buf.append(torch.tensor(logp, dtype=torch.float32).to(self.device))
        self.val_buf.append(torch.tensor(value, dtype=torch.float32).unsqueeze(0).to(self.device))
        self.rew_buf.append(float(reward))

    def update(self) -> dict:
        """PPO update using one episode's worth of daily transitions."""
        if not self.rew_buf:
            return {}

        # Discounted returns (no bootstrapping — episode ends after last day)
        returns = []
        G = 0.0
        for r in reversed(self.rew_buf):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns_t = torch.FloatTensor(returns).to(self.device)

        obs_t = torch.cat(self.obs_buf, 0)
        acts_t = torch.cat(self.act_buf, 0)
        old_logps_t = torch.stack(self.logp_buf).detach()
        old_vals_t = torch.cat(self.val_buf, 0).detach()

        advantages = returns_t - old_vals_t
        adv_mean = advantages.mean().item()
        adv_std = advantages.std().item() if advantages.numel() > 1 else 0.0
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss_val = critic_loss_val = 0.0
        for _ in range(self.n_ppo_epochs):
            mean, std, values = self._forward(obs_t)
            dist = torch.distributions.Normal(mean, std)
            new_logps = dist.log_prob(acts_t).sum(-1)

            ratio = (new_logps - old_logps_t).exp()
            surr = torch.min(
                ratio * advantages,
                ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages,
            )
            actor_loss = -surr.mean()
            critic_loss = (returns_t - values).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer.step()

            actor_loss_val = actor_loss.item()
            critic_loss_val = critic_loss.item()

        self.obs_buf.clear()
        self.act_buf.clear()
        self.logp_buf.clear()
        self.val_buf.clear()
        self.rew_buf.clear()

        return {
            "meta_actor_loss": actor_loss_val,
            "meta_critic_loss": critic_loss_val,
            "meta_advantage_mean": adv_mean,
            "meta_advantage_std": adv_std,
        }


class HeuristicMetaPolicy:
    """Deterministic meta-policy for sanity checks. No learning.

    Supports a small set of fixed strategies, selected by `heuristic_name`:
      - "const_1":   α = 1.0 every day (should be a no-op equivalent to --meta_policy none).
      - "const_05":  α = 0.5 every day (always undercut).
      - "const_2":   α = 2.0 every day (always overprice, hits new upper bound).
      - "schedule_undercut_exploit": α = 0.5 for first half of episode days, α = 1.5 thereafter.
    """

    def __init__(self, heuristic_name: str, num_days: int):
        self.heuristic_name = heuristic_name
        self.num_days = max(1, num_days)
        self._day_idx = 0  # within-episode counter; wraps every num_days calls

    def select_action(self, obs) -> float:
        d = self._day_idx
        if self.heuristic_name == "const_1":
            alpha = 1.0
        elif self.heuristic_name == "const_05":
            alpha = 0.5
        elif self.heuristic_name == "const_2":
            alpha = 2.0
        elif self.heuristic_name == "schedule_undercut_exploit":
            half = self.num_days // 2
            alpha = 0.5 if d < half else 1.5
        else:
            raise ValueError(f"Unknown heuristic: {self.heuristic_name}")
        self._day_idx = (self._day_idx + 1) % self.num_days
        return float(alpha)

    def store_reward(self, reward: float):
        pass

    def update(self) -> dict:
        return {}
