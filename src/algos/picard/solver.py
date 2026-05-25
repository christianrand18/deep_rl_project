"""Picard fixed-point iteration coordinator for multi-day episode rollouts.

See SPEC.md §7 for the math. The solver is a *coordinator*, not a simulator:
the day simulation stays in main_a2c_multi_agent.py unchanged. The solver only
manages four injection points around the existing day loop:

    solver = PicardSolver(env, meta_policies, model_agents, args)

    for i_episode in epochs:
        env.reset()
        solver.begin_episode(i_episode, env)
        rerun = True

        while rerun:
            # Clear low-level buffers so only the final iteration reaches training_step().
            for a in model_agents:
                model_agents[a].rewards = []
                model_agents[a].saved_actions = []

            for i_day in range(args.num_days):
                if i_day > 0:
                    obs = env.reset_day()
                accumulator.reset(env)
                meta_multipliers = solver.prepare_day(i_day, env)  # ← injects brand_momentum + alpha

                done = False
                while not done:
                    ...  # existing inner loop completely unchanged

                env.update_brand_momentum(served_counts=day_served, total_demand=day_total_demand)
                meta_reward = {a: accumulator.profit[a] - accumulator.reb_cost[a] for a in [0, 1]}
                solver.record_day(i_day, env, accumulator, meta_reward)   # ← captures state

            rerun = solver.next_iteration(env)                            # ← convergence check

        result = solver.commit(meta_policies)      # ← populates meta PPO buffers; returns EpisodeResult
        # use result.day_results for WandB day-level logging
        # existing model_agents.training_step() and meta_policies.update() follow unchanged

Meta-policy select_action() and store_reward() are NOT called inside the
while-rerun loop; solver.commit() handles meta-policy buffer population.
Gate those calls in main on `not args.parallel_days`.
"""

from dataclasses import dataclass, field
from typing import Optional
import random as _random

import numpy as np
import torch


@dataclass
class DayState:
    """State carried from the end of day d-1 into the start of day d.

    brand_momentum: seeds env.brand_momentum before day d runs.
    meta_obs: 7-dim daily_state vector per operator; meta-policy uses this to
              produce alpha(d).
    """
    brand_momentum: dict    # {agent_id: float}
    meta_obs: dict          # {agent_id: np.ndarray[7]}

    def to_flat(self) -> np.ndarray:
        parts = []
        for a in sorted(self.brand_momentum.keys()):
            parts.append(np.array([self.brand_momentum[a]], dtype=np.float32))
            parts.append(self.meta_obs[a].astype(np.float32))
        return np.concatenate(parts)


@dataclass
class DayResult:
    """Outcome of one simulated day, ready for meta-policy buffer commit and logging."""
    day_idx: int
    next_state: DayState
    meta_obs_in: dict   # {agent_id: obs array} — what meta-policy saw to produce alpha(d)
    meta_out: dict      # {agent_id: (alpha_ndarray, logp_scalar, value_scalar)}
    meta_reward: dict   # {agent_id: float}
    wandb_metrics: dict = field(default_factory=dict)


@dataclass
class EpisodeResult:
    day_results: list       # ordered by day, length num_days
    K_used: int             # iterations until convergence (or max_iters)
    final_delta: float
    converged: bool
    delta_history: list     # one entry per Picard iteration


class PicardSolver:
    """Picard fixed-point iteration coordinator.

    Thin wrapper around the existing day loop. Does not own the simulation.
    See module docstring for the integration pattern.

    Strategies are plain overridable methods.
    Defaults: zero-init, full-replace update (pure Picard).
    """

    def __init__(self, env, meta_policies, model_agents, args,
                 max_iters: int = 6, tol: float = 1e-3,
                 episode_seed_base: int = 12345):
        self.env = env
        self.meta_policies = meta_policies or {}
        self.model_agents = model_agents
        self.args = args

        self.num_days = args.num_days
        self.n_regions = env.nregion
        self.agents = list(env.agents)
        self.meta_agents = sorted(self.meta_policies.keys())

        self.max_iters = max_iters
        self.tol = tol
        self.episode_seed_base = episode_seed_base

        # Per-episode state (reset by begin_episode)
        self._k: int = 0
        self._day_seeds: Optional[np.ndarray] = None
        self._z_noise: Optional[list] = None
        self._S_pred: Optional[list] = None
        self._day_results: list = []
        self._delta_history: list = []
        self._current_meta_out: dict = {}
        self._current_prev_state: Optional[DayState] = None
        self._last_result: Optional[EpisodeResult] = None

    # ── coordinator API ───────────────────────────────────────────────────────

    def begin_episode(self, episode_idx: int, env):
        """Initialise for a new episode. Call once, right after env.reset()."""
        self._k = 0
        self._day_seeds, self._z_noise = self._presample_noise(episode_idx)
        self._S_pred = self._pick_initial_guess(episode_idx)
        self._day_results = []
        self._delta_history = []
        self._last_result = None

    def prepare_day(self, i_day: int, env) -> dict:
        """Inject predicted state; return meta multipliers for this day.

        Call at day start — after reset_day() (if i_day > 0), before the
        while-not-done loop.

        Side effects:
          - Sets env.brand_momentum from the current Picard prediction.
          - Seeds np.random, env._shuffle_rng, and torch with the day's seed
            so the simulation is deterministic w.r.t. the iteration index.

        Returns {agent_id: alpha_ndarray}.
        """
        seed = int(self._day_seeds[i_day])
        np.random.seed(seed)
        env._shuffle_rng = _random.Random(seed)
        torch.manual_seed(seed)

        self._current_prev_state = self._S_pred[i_day]
        env.brand_momentum = dict(self._S_pred[i_day].brand_momentum)

        meta_out = self._meta_forward(self._S_pred[i_day], self._z_noise[i_day])
        self._current_meta_out = meta_out
        return {a: meta_out[a][0] for a in self.agents}

    def record_day(self, i_day: int, env, accumulator, meta_reward: dict):
        """Capture the day's outcome.

        Call after env.update_brand_momentum() and accumulator has been updated
        for the day.

        meta_reward: {agent_id: float}. Typically:
            {a: accumulator.profit[a] - accumulator.reb_cost[a] for a in [0, 1]}
        """
        accumulator.momentum_snapshot = dict(env.brand_momentum)
        next_state = DayState(
            brand_momentum=dict(env.brand_momentum),
            meta_obs={
                a: accumulator.daily_state(
                    a, i_day + 1, self.num_days, self.args.reward_scalar
                )
                for a in self.agents
            },
        )
        self._day_results.append(DayResult(
            day_idx=i_day,
            next_state=next_state,
            meta_obs_in={a: self._current_prev_state.meta_obs[a] for a in self.agents},
            meta_out=dict(self._current_meta_out),
            meta_reward=dict(meta_reward),
        ))

    def next_iteration(self, env) -> bool:
        """Check convergence after all N days.

        Returns True  → not yet converged; S_pred updated, env reset, re-run the loop.
        Returns False → done; call commit() next.
        """
        S_new = [self._S_pred[0]] + [r.next_state for r in self._day_results]
        delta = self._compute_delta(S_new, self._S_pred)
        self._delta_history.append(delta)

        if self._converged(delta) or self._k + 1 >= self.max_iters:
            self._last_result = EpisodeResult(
                day_results=list(self._day_results),
                K_used=self._k + 1,
                final_delta=delta,
                converged=self._converged(delta),
                delta_history=list(self._delta_history),
            )
            return False

        self._S_pred = self._update_guess(self._S_pred, S_new, self._k, self._delta_history)
        self._k += 1
        self._day_results = []
        return True  # caller must call env.reset() before the next pass

    def commit(self, meta_policies) -> EpisodeResult:
        """Populate meta-policy PPO buffers from the converged day results.

        Call once after next_iteration() returns False, before meta_policy.update().
        Returns the EpisodeResult for WandB logging and diagnostics.
        """
        if self._last_result is None:
            raise RuntimeError("commit() called before an episode completed.")
        for result in self._last_result.day_results:
            for a in self.meta_agents:
                alpha, logp, value = result.meta_out[a]
                meta_policies[a].append_transition(
                    obs=result.meta_obs_in[a],
                    act=alpha,
                    logp=logp,
                    value=value,
                    reward=result.meta_reward.get(a, 0.0),
                )
        return self._last_result

    # ── strategies (override to experiment) ───────────────────────────────────

    def _pick_initial_guess(self, episode_idx: int) -> list:
        """BASIC: zero-init. M=0.5, meta_obs zeros for every day."""
        return [self._zero_state() for _ in range(self.num_days + 1)]

    def _update_guess(self, S_old, S_new, iter_idx: int, delta_history: list) -> list:
        """BASIC: full replace — S^(k+1) = f(S^(k)) (pure Picard)."""
        return list(S_new)

    def _compute_delta(self, S_new, S_old) -> float:
        """Max-norm over (M, meta_obs) on days 1..N. Skips d=0 (fixed episode-start)."""
        if len(S_new) <= 1:
            return 0.0
        return max(
            float(np.max(np.abs(S_new[d].to_flat() - S_old[d].to_flat())))
            for d in range(1, len(S_new))
        )

    def _converged(self, delta: float) -> bool:
        return delta < self.tol

    # ── building blocks ───────────────────────────────────────────────────────

    def _zero_state(self) -> DayState:
        # M=0.5 matches env.reset(); meta_obs zeros match existing sequential
        # code path where the first day's meta_obs is np.zeros(7).
        return DayState(
            brand_momentum={a: 0.5 for a in self.agents},
            meta_obs={a: np.zeros(7, dtype=np.float32) for a in self.agents},
        )

    def _presample_noise(self, episode_idx: int):
        """Per-episode demand seeds and meta-policy Gaussian noise.

        Fixed for the whole episode so the Picard fixed point is well-defined.
        """
        rng = np.random.RandomState(self.episode_seed_base + episode_idx)
        day_seeds = rng.randint(0, 2**31 - 1, size=self.num_days)
        z_noise = [
            {a: rng.standard_normal(self.n_regions).astype(np.float32)
             for a in self.meta_agents}
            for _ in range(self.num_days)
        ]
        return day_seeds, z_noise

    def _meta_forward(self, prev_state: DayState, z_noise: dict) -> dict:
        """Deterministic meta-policy forward: alpha = clamp(mean(obs) + std*z, 0, 2).

        Pre-sampled z makes alpha a continuous deterministic function of state
        (same distribution as select_action(), just with a fixed noise draw).
        Returns {agent_id: (alpha_ndarray, logp_scalar, value_scalar)}.
        Agents without a meta-policy get (ones, 0.0, 0.0).
        """
        result = {a: (np.ones(self.n_regions, dtype=np.float32), 0.0, 0.0)
                  for a in self.agents}
        for a, mp in self.meta_policies.items():
            obs_t = torch.from_numpy(prev_state.meta_obs[a]).float().unsqueeze(0).to(mp.device)
            with torch.no_grad():
                mean, std, value = mp._forward(obs_t)
            z_t = torch.from_numpy(z_noise[a]).float().unsqueeze(0).to(mp.device)
            action = (mean + std * z_t).clamp(0.0, 2.0)
            logp = torch.distributions.Normal(mean, std).log_prob(action).sum(-1)
            result[a] = (
                action.squeeze(0).cpu().numpy(),
                float(logp.item()),
                float(value.item()),
            )
        return result
