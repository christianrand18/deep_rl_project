# Picard Solver

Fixed-point iteration coordinator for multi-day episode rollouts. Opt-in via `--parallel_days`.

## Problem

In a multi-day episode, brand momentum (BM) carries over between days via an EMA:

```
BM[d+1] = λ · BM[d] + (1-λ) · market_share[d]
```

Without Picard, day `d` is initialised with whatever BM the previous day produced sequentially. With Picard, days run with a *predicted* BM trajectory and iterate until the predictions are self-consistent — so day 3 genuinely sees the BM consequence of day 2's actions, not a stale initialised value.

## Architecture

The solver is a **thin coordinator** — it injects state at day start and captures it at day end. The day simulation loop in `main_a2c_multi_agent.py` is untouched. Four injection points:

| Method | When to call | What it does |
|---|---|---|
| `begin_episode(i, env)` | After `env.reset()` | Pre-samples seeds & noise, initialises `S_pred` from warm start |
| `prepare_day(i_day, env)` | Before day loop | Injects `brand_momentum` into env, seeds RNGs, returns `{agent: alpha}` |
| `record_day(i_day, env, acc, reward)` | After `update_brand_momentum()` | Captures `DayState` (BM + meta_obs) for this day |
| `next_iteration(env) → bool` | After all N days | Convergence check; `True` = re-run days with updated `S_pred` |
| `commit(meta_policies) → EpisodeResult` | After convergence | Populates meta PPO buffers; returns diagnostics |

## Update strategies (`--picard_update_strategy`)

### `analytic` (default)
Back-calculates observed market shares from the EMA formula and re-propagates the full BM trajectory analytically in one sequential pass:

```
ms[d] = (BM_out[d] - λ · BM_in[d]) / (1-λ)
BM[d+1] = λ · BM[d] + (1-λ) · ms[d]   for d = 0..N-1
```

Fixes all N days simultaneously rather than one per iteration (pure Jacobi). Converges in K≈2–3 on warm-started episodes. Optional damping via `--picard_omega` (default 1.0 = no damping).

### `anderson` (recommended)
Applies the analytic update as `G(x_k)`, then mixes the last `m` outputs using Anderson acceleration: finds the convex combination `θ` of `{G(x_j)}` that minimises the residual norm. Breaks oscillation that the plain analytic iteration can exhibit when the BM→market_share feedback is strong (`bm_gamma` > 1). Converges in K=1–3 with `tol=1e-3`. Controlled by `--picard_anderson_m` (default 5, window size).

### `jacobi`
Pure full-replace: `S^(k+1) = S_new`. Converges in N+1 iterations for N days. Included as a baseline; `analytic` is strictly better.

## Key design decisions

- **Fixed seeds per episode** — `_presample_noise` draws day seeds and meta-policy noise once at episode start. Held constant across Picard iterations so the fixed point is well-defined (same demand, same alpha draws).
- **Warm start** — converged `S_pred` from the previous episode is reused as the initial guess. Keeps K=1 convergence common once training stabilises.
- **Delta on BM only** — convergence is measured as max-norm over `brand_momentum` (days 1..N). `meta_obs` lags by one iteration but stabilises quickly.
- **Anderson history reset per episode** — `_anderson_history` is cleared in `begin_episode` so mixing never leaks across episodes.
- **Days stay independent** — all update logic runs *after* all N days finish. Individual day simulations are never modified and remain fully parallelisable.

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--parallel_days` | False | Enable Picard |
| `--picard_max_iters` | 6 | Max iterations per episode |
| `--picard_tol` | 1e-3 | Convergence tolerance (max-norm on BM) |
| `--picard_update_strategy` | `analytic` | `analytic`, `anderson`, or `jacobi` |
| `--picard_omega` | 1.0 | Damping factor for analytic/anderson (1.0 = off) |
| `--picard_anderson_m` | 5 | Window size for Anderson acceleration |

## WandB diagnostics (`debug/` panel)

| Metric | What to watch for |
|---|---|
| `debug/picard_K_used` | Should stay 1–3 with Anderson; spikes to max_iters indicate oscillation |
| `debug/picard_converged` | Frequent 0s mean tol is too tight or max_iters too low |
| `debug/picard_delta_i1` | First-iteration residual; trends down as warm start improves |
| `debug/picard_final_delta` | Residual on exit; hovering at ~4.5e-4 is the discrete demand noise floor |
