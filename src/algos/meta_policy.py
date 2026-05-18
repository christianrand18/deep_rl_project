import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MetaPolicy(nn.Module):
    """Daily meta-policy (PPO MLP) that outputs per-region price multipliers in [0, 2].

    Operates at day frequency: one select_action() call per day, one store_reward() call
    per day, and one update() call per episode.
    """

    def __init__(self, obs_dim, n_regions, hidden_dim=128, lr=3e-4, gamma=0.99,
                 clip_eps=0.2, n_ppo_epochs=4, device='cpu'):
        super().__init__()
        self.n_regions = n_regions
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.n_ppo_epochs = n_ppo_epochs
        self.device = device

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        # Actor: mean ∈ (0, 2) via 2*sigmoid; fixed std=0.1 for exploration
        self.actor_head = nn.Linear(hidden_dim, n_regions)
        self.critic_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((n_regions,), -2.3))  # std ≈ 0.1 initially

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.obs_buf = []
        self.act_buf = []
        self.logp_buf = []
        self.val_buf = []
        self.rew_buf = []

        self.to(device)

    def _forward(self, obs_t):
        x = self.trunk(obs_t)
        mean = 2.0 * torch.sigmoid(self.actor_head(x))
        std = self.log_std.exp().clamp(1e-4, 1.0)
        value = self.critic_head(x).squeeze(-1)
        return mean, std, value

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Returns price multipliers in [0, 2] and stores transition in buffers."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std, value = self._forward(obs_t)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample().clamp(0.0, 2.0)
        logp = dist.log_prob(action).sum(-1)

        self.obs_buf.append(obs_t)
        self.act_buf.append(action)
        self.logp_buf.append(logp)
        self.val_buf.append(value)

        return action.squeeze(0).cpu().numpy()

    def store_reward(self, reward: float):
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

        return {"meta_actor_loss": actor_loss_val, "meta_critic_loss": critic_loss_val}
