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
        self._S_old = list(self._S_pred)
        self._day_results = []
        self._delta_history = []
        self._last_result = None
        print(f"[Picard] episode {episode_idx} begin, max_iters={self.max_iters}, tol={self.tol}")

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
        if i_day + 1 < len(self._S_pred):
            self._S_pred[i_day + 1] = next_state

    def next_iteration(self, env) -> bool:
        """Check convergence after all N days (Gauss-Seidel style).

        S_pred was mutated in-place by record_day during the for-loop, so the
        state from day d propagates immediately into day d+1 within one sweep.

        Returns True  → not yet converged; re-run the loop.
        Returns False → done; call commit() next.
        """
        delta = self._compute_delta(self._S_pred, self._S_old)
        self._delta_history.append(delta)
        print(f"[Picard] iter={self._k + 1}/{self.max_iters} delta={delta:.6f} tol={self.tol:.6f}", end="")

        if self._converged(delta) or self._k + 1 >= self.max_iters:
            tag = "CONVERGED" if self._converged(delta) else "MAX_ITERS"
            print(f" -> {tag} (K={self._k + 1})")
            self._last_result = EpisodeResult(
                day_results=list(self._day_results),
                K_used=self._k + 1,
                final_delta=delta,
                converged=self._converged(delta),
                delta_history=list(self._delta_history),
            )
            return False

        print(" -> continue")
        S_old_copy = self._S_old
        self._S_pred = self._update_guess(S_old_copy, self._S_pred, self._k, self._delta_history)
        self._S_old = list(self._S_pred)
        self._k += 1
        self._day_results = []
        return True

    def commit(self, meta_policies) -> EpisodeResult:
        """Populate meta-policy PPO buffers from the converged day results.

        Call once after next_iteration() returns False, before meta_policy.update().
        Returns the EpisodeResult for WandB logging and diagnostics.
        """
        if self._last_result is None:
            raise RuntimeError("commit() called before an episode completed.")
        res = self._last_result
        print(f"[Picard] commit: K_used={res.K_used} converged={res.converged} final_delta={res.final_delta:.6f} delta_history={[float(f'{d:.6f}') for d in res.delta_history]}")
        for result in res.day_results:
            for a in self.meta_agents:
                alpha, logp, value = result.meta_out[a]
                meta_policies[a].append_transition(
                    obs=result.meta_obs_in[a],
                    act=alpha,
                    logp=logp,
                    value=value,
                    reward=result.meta_reward.get(a, 0.0),
                )
        return res

    # ── strategies (override to experiment) ───────────────────────────────────

    def _pick_initial_guess(self, episode_idx: int) -> list:
        """BASIC: zero-init. M=0.5, meta_obs zeros for every day."""
        return [self._zero_state() for _ in range(self.num_days + 1)]

    def _update_guess(self, S_old, S_new, iter_idx: int, delta_history: list) -> list:
        """BASIC: identity pass (Gauss-Seidel already updated S_pred in-place).

        S_last is the pre-iteration snapshot; S_cur is the post-sweep state.
        Override to add damping: return [(1-ω)*S_old[d] + ω*S_new[d]].
        """
        return list(S_new)

    def _compute_delta(self, S_new, S_old) -> float:
        """Max-norm over brand_momentum on days 1..N. Skips d=0 (fixed episode-start)."""
        if len(S_new) <= 1:
            return 0.0
        return max(
            max(abs(S_new[d].brand_momentum[a] - S_old[d].brand_momentum[a])
                for a in self.agents)
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
