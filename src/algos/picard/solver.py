"""Picard fixed-point iteration coordinator for multi-day episode rollouts.

Pure Jacobi: all N days read from the same S_pred snapshot, run independently
(parallelizable), then S_pred is replaced with S_new = f(S_old).  Correctness
propagates one day per iteration, guaranteeing convergence in ≤ N iterations.

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


@dataclass
class WorkerDayPayload:
    """Self-contained per-day data shipped to a parallel worker.

    Avoids sending the full PicardSolver across the process boundary. The
    worker uses these fields via a small shim that quacks like the solver
    (see src/algos/picard/parallel.py::_WorkerShim).
    """
    i_day: int
    bm_in: dict             # {agent: float} → injected as env.brand_momentum
    day_seed: int           # seeds np / env._shuffle_rng / torch
    meta_alpha: dict        # {agent: float} → returned by shim.prepare_day
    num_days: int           # for shim.record_day → accumulator.daily_state
    reward_scalar: float    # ditto
    agents: list


@dataclass
class WorkerDayCapture:
    """Per-day outcome shipped back from a parallel worker."""
    bm_out: dict            # {agent: float} env.brand_momentum after the day
    meta_obs: dict          # {agent: ndarray[7]}
    meta_reward: dict       # {agent: float}


class PicardSolver:
    """Picard fixed-point iteration coordinator.

    Thin wrapper around the existing day loop. Does not own the simulation.
    See module docstring for the integration pattern.

    Strategies are plain overridable methods.
    Defaults: zero-init, full-replace update (pure Picard).
    """

    def __init__(self, env, meta_policies, model_agents, args,
                 max_iters: int = 6, tol: float = 1e-3,
                 episode_seed_base: int = 12345,
                 update_strategy: str = 'analytic',
                 omega: float = 1.0,
                 anderson_m: int = 5):
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
        self._update_strategy = update_strategy
        self._omega = omega
        self._anderson_m = anderson_m

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
        self._prev_S: Optional[list] = None
        self._anderson_history: list = []   # list of (x_flat, Gx_flat) per iteration
        # meta_out cache keyed by i_day; populated by make_worker_payload,
        # consumed by merge_worker_capture. Resets per Picard iteration so
        # earlier iterations' meta-policy noise outputs don't leak in.
        self._meta_out_by_day: dict = {}

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
        self._anderson_history = []
        self._meta_out_by_day = {}

    def prepare_day(self, i_day: int, env) -> dict:
        """Inject predicted state; return meta multipliers for this day.

        Call at day start — after reset_day() (if i_day > 0), before the
        while-not-done loop.

        Side effects:
          - Sets env.brand_momentum from the current Picard prediction.
          - Seeds np.random, env._shuffle_rng, and torch with the day's seed
            so the simulation is deterministic w.r.t. the iteration index.

        Returns {agent_id: alpha_scalar}.
        """
        seed = int(self._day_seeds[i_day])
        np.random.seed(seed)
        env._shuffle_rng = _random.Random(seed)
        torch.manual_seed(seed)

        self._current_prev_state = self._S_pred[i_day]
        env.brand_momentum = dict(self._S_pred[i_day].brand_momentum)

        meta_out = self._meta_forward(self._S_pred[i_day], self._z_noise[i_day])
        self._current_meta_out = meta_out
        return {a: float(meta_out[a][0]) for a in self.agents}

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
        """Check convergence after all N days (Jacobi).

        All days were computed from the same S_old snapshot.  Build S_new from
        day results, compare brand_momentum, replace S_pred if not converged.

        Returns True  → not yet converged; re-run the loop.
        Returns False → done; call commit() next.
        """
        S_new = [self._S_pred[0]] + [r.next_state for r in self._day_results]
        delta = self._compute_delta(S_new, self._S_old)
        self._delta_history.append(delta)

        if self._converged(delta) or self._k + 1 >= self.max_iters:
            self._prev_S = list(S_new)
            self._last_result = EpisodeResult(
                day_results=list(self._day_results),
                K_used=self._k + 1,
                final_delta=delta,
                converged=self._converged(delta),
                delta_history=list(self._delta_history),
            )
            return False

        S_old_copy = self._S_old
        self._S_pred = self._update_guess(S_old_copy, S_new, self._k, self._delta_history)
        self._S_old = list(self._S_pred)
        self._k += 1
        self._day_results = []
        self._meta_out_by_day = {}
        return True

    # ── parallel-worker bridge ────────────────────────────────────────────────

    def make_worker_payload(self, i_day: int) -> WorkerDayPayload:
        """Build a self-contained payload for a worker to run day ``i_day``.

        Side effect: caches the day's meta_out (alpha, logp, value tuples) so
        ``merge_worker_capture`` can reconstruct the DayResult without
        round-tripping the torch outputs back through pickle (the meta-policy
        networks only live in the central process).
        """
        meta_out = self._meta_forward(self._S_pred[i_day], self._z_noise[i_day])
        self._meta_out_by_day[i_day] = meta_out
        return WorkerDayPayload(
            i_day=i_day,
            bm_in=dict(self._S_pred[i_day].brand_momentum),
            day_seed=int(self._day_seeds[i_day]),
            meta_alpha={a: float(meta_out[a][0]) for a in self.agents},
            num_days=self.num_days,
            reward_scalar=self.args.reward_scalar,
            agents=list(self.agents),
        )

    def merge_worker_capture(self, i_day: int, capture: WorkerDayCapture) -> None:
        """Replay a worker's per-day capture into central state.

        Mirrors what ``record_day`` would have done, but using ``capture``
        (filled by the worker) instead of reading live env/accumulator state.
        Call in i_day order so ``_day_results`` stays day-ordered.

        ``meta_obs_in`` is read directly from ``self._S_pred[i_day]`` rather
        than from ``self._current_prev_state`` (which only ``prepare_day``
        on the central solver sets — workers run ``prepare_day`` on the shim,
        so the central solver's per-call state is irrelevant here).
        """
        next_state = DayState(
            brand_momentum=dict(capture.bm_out),
            meta_obs=dict(capture.meta_obs),
        )
        self._day_results.append(DayResult(
            day_idx=i_day,
            next_state=next_state,
            meta_obs_in={a: self._S_pred[i_day].meta_obs[a] for a in self.agents},
            meta_out=dict(self._meta_out_by_day[i_day]),
            meta_reward=dict(capture.meta_reward),
        ))

    def commit(self, meta_policies) -> EpisodeResult:
        """Populate meta-policy PPO buffers from the converged day results.

        Call once after next_iteration() returns False, before meta_policy.update().
        Returns the EpisodeResult for WandB logging and diagnostics.
        """
        if self._last_result is None:
            raise RuntimeError("commit() called before an episode completed.")
        res = self._last_result
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
        """Warm-start from previous episode's converged state, or zero-init if none."""
        if self._prev_S is not None:
            return list(self._prev_S)
        return [self._zero_state() for _ in range(self.num_days + 1)]

    def _update_guess(self, S_old, S_new, iter_idx: int, delta_history: list) -> list:
        """Dispatch to the configured update strategy.

        S_old: previous S_pred (len N+1); BM_in[d] = S_old[d].brand_momentum.
        S_new: result of running all days with S_old (len N+1); BM_out[d] = S_new[d+1].brand_momentum.

        Add new strategies as _<name>_update_guess methods and set update_strategy accordingly.
        """
        if self._update_strategy == 'jacobi':
            return self._jacobi_update_guess(S_old, S_new, iter_idx, delta_history)
        elif self._update_strategy == 'analytic':
            return self._analytic_update_guess(S_old, S_new, iter_idx, delta_history)
        elif self._update_strategy == 'anderson':
            return self._anderson_update_guess(S_old, S_new, iter_idx, delta_history)
        else:
            raise ValueError(f"[Picard] Unknown update_strategy: {self._update_strategy!r}")

    def _jacobi_update_guess(self, S_old, S_new, _iter_idx: int, _delta_history: list) -> list:
        """Pure Jacobi: full replace — S^(k+1) = S_new.

        Converges in exactly N+1 iterations for a causal chain of N days (cold start).
        """
        return list(S_new)

    def _analytic_update_guess(self, S_old, S_new, _iter_idx: int, _delta_history: list) -> list:
        """Tier 1: analytically propagate BM forward using observed market shares.

        Assumption: market_share[d] is approximately independent of BM_in[d].
        This holds when bm_gamma (the BM coefficient in the MNL utility) is small,
        i.e. BM has weak direct effect on passenger choice within a day.

        After one parallel run we have observed market_share[d] for all d. Instead
        of propagating these one step at a time (Jacobi), we apply the EMA formula
        sequentially with the observed market shares to get a corrected BM trajectory
        in a single pass — fixing all N days at once rather than one per iteration.

        meta_obs is carried unchanged from S_new (computed under the old BM_in; it
        lags by one Picard iteration but converges quickly in 1-2 extra iterations).

        omega < 1 damps the update: BM_final = (1-ω)·BM_old + ω·BM_analytic, which
        prevents oscillation when the BM→market_share feedback is non-negligible.
        """
        lam = self.env.bm_lambda
        market_shares = self._compute_market_shares(S_old, S_new)

        result = [S_new[0]]  # S[0] is always fixed (true episode start)
        bm_current = dict(S_new[0].brand_momentum)

        for d in range(self.num_days):
            ms = market_shares[d]
            bm_next = {a: lam * bm_current[a] + (1 - lam) * ms[a] for a in self.agents}
            bm_next = self._apply_damping(bm_next, S_old[d + 1].brand_momentum)
            result.append(DayState(
                brand_momentum=bm_next,
                meta_obs=dict(S_new[d + 1].meta_obs),
            ))
            bm_current = bm_next

        return result

    def _anderson_update_guess(self, S_old, S_new, iter_idx: int, delta_history: list) -> list:
        """Anderson acceleration on top of the analytic update.

        Uses the last anderson_m iterates to find the optimal linear combination
        of G(x_j) values that minimises the residual norm, breaking the slow
        contraction / oscillation of the plain fixed-point iteration.

        Each iteration:
          1. Apply analytic update → G(x_k)
          2. Store (x_k, G(x_k)) in a rolling window of length anderson_m
          3. Solve: min ||F θ||²  s.t.  sum(θ) = 1,  F[:,j] = G(x_j) - x_j
          4. Return G_history @ θ  (mixed over BM dimensions only;
             meta_obs is taken from G(x_k) unchanged)
        """
        S_g = self._analytic_update_guess(S_old, S_new, iter_idx, delta_history)

        x_flat  = self._bm_to_flat(S_old)
        gx_flat = self._bm_to_flat(S_g)

        self._anderson_history.append((x_flat, gx_flat))
        if len(self._anderson_history) > self._anderson_m:
            self._anderson_history.pop(0)

        if len(self._anderson_history) < 2:
            return S_g   # not enough history yet; use plain analytic result

        bm_mixed = self._anderson_mix(self._anderson_history)

        # Reconstruct DayState list: mixed BM, meta_obs from G(x_k)
        result = [S_g[0]]   # day 0 is always fixed
        n_a = len(self.agents)
        for d in range(self.num_days):
            bm_slice = bm_mixed[d * n_a: (d + 1) * n_a]
            result.append(DayState(
                brand_momentum={a: float(bm_slice[i]) for i, a in enumerate(self.agents)},
                meta_obs=dict(S_g[d + 1].meta_obs),
            ))
        return result

    def _bm_to_flat(self, S: list) -> np.ndarray:
        """Flatten brand_momentum for days 1..N into a 1-D float64 array."""
        return np.array(
            [S[d].brand_momentum[a] for d in range(1, len(S)) for a in self.agents],
            dtype=np.float64,
        )

    def _anderson_mix(self, history: list) -> np.ndarray:
        """Solve the Anderson least-squares problem and return the mixed G(x).

        min ||F θ||²  s.t.  sum(θ) = 1
        where F[:,j] = gx_j - x_j  (residuals).

        Uses the unconstrained reformulation (eliminate one θ via the constraint)
        with Tikhonov regularisation (λ = 1e-10 · ||ΔF||²_F) for stability.
        Falls back to the most recent G(x) on singular systems.
        """
        xs  = np.stack([h[0] for h in history], axis=1)   # (dim, m)
        gxs = np.stack([h[1] for h in history], axis=1)   # (dim, m)
        R   = gxs - xs                                     # residuals (dim, m)
        m   = len(history)

        # Unconstrained reformulation:
        #   dR[:,j] = R[:,j] - R[:,-1]   (shape: dim × m-1)
        #   solve:  min ||dR c + R[:,-1]||²   →   c = -(dR^T dR)^{-1} dR^T R[:,-1]
        #   recover: θ = [c; 1 - sum(c)]
        dR  = R[:, :-1] - R[:, -1:]                        # (dim, m-1)
        lam = 1e-10 * float((dR * dR).sum())
        A   = dR.T @ dR + lam * np.eye(m - 1)
        b   = -(dR.T @ R[:, -1])
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return gxs[:, -1]   # fallback: most recent G(x)

        theta = np.empty(m)
        theta[:-1] = c
        theta[-1]  = 1.0 - c.sum()
        return gxs @ theta

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

    def _apply_damping(self, bm_new: dict, bm_old: dict) -> dict:
        """Mix bm_new toward bm_old: BM_final = (1-ω)·BM_old + ω·BM_new.

        With omega=1.0 (default) this is a no-op. Set omega < 1 to dampen
        oscillation when the BM→market_share feedback causes cycling.
        """
        if self._omega >= 1.0:
            return bm_new
        return {a: (1.0 - self._omega) * bm_old[a] + self._omega * bm_new[a]
                for a in self.agents}

    def _compute_market_shares(self, S_old, S_new) -> list:
        """Back-calculate observed market share for each day from BM in/out.

        Uses the EMA formula: BM[d+1] = λ·BM[d] + (1-λ)·ms[d], solved for ms:
            ms[d][a] = (BM_out[a] - λ·BM_in[a]) / (1-λ)

        where BM_in[d] = S_old[d].brand_momentum and BM_out[d] = S_new[d+1].brand_momentum.

        Returns list of length num_days: [{agent_id: float}, ...].
        """
        lam = self.env.bm_lambda
        one_minus_lam = 1.0 - lam
        market_shares = []
        for d in range(self.num_days):
            bm_in = S_old[d].brand_momentum
            bm_out = S_new[d + 1].brand_momentum
            market_shares.append({
                a: (bm_out[a] - lam * bm_in[a]) / one_minus_lam
                for a in self.agents
            })
        return market_shares

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
            {a: np.array(rng.standard_normal(), dtype=np.float32)
             for a in self.meta_agents}
            for _ in range(self.num_days)
        ]
        return day_seeds, z_noise

    def _meta_forward(self, prev_state: DayState, z_noise: dict) -> dict:
        """Deterministic meta-policy forward: alpha = clamp(mean(obs) + std*z, 0, 2).

        Pre-sampled z makes alpha a continuous deterministic function of state
        (same distribution as select_action(), just with a fixed noise draw).
        Returns {agent_id: (alpha_scalar, logp_scalar, value_scalar)}.
        Agents without a meta-policy get (1.0, 0.0, 0.0).
        """
        result = {a: (1.0, 0.0, 0.0)
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
