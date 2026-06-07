# Picard Solver

Fixed-point iteration coordinator for multi-day episode rollouts. Opt-in via `--parallel_days`. Optionally runs the N days of an iteration in parallel via `--picard_parallel_workers > 1`.

## Problem

In a multi-day episode, brand momentum (BM) carries over between days via an EMA:

```
BM[d+1] = λ · BM[d] + (1-λ) · market_share[d]
```

Without Picard, day `d` is initialised with whatever BM the previous day produced sequentially. With Picard, days run with a *predicted* BM trajectory and iterate until the predictions are self-consistent — so day 3 genuinely sees the BM consequence of day 2's actions, not a stale initialised value.

## Architecture

The solver is a **thin coordinator** — it injects state at day start and captures it at day end. The day-simulation logic lives in `src/misc/day_runner.py::run_day`; the solver never modifies it. Five sequential-path injection points:

| Method | When to call | What it does |
|---|---|---|
| `begin_episode(i, env)` | After `env.reset()` | Pre-samples seeds & noise, initialises `S_pred` from warm start |
| `prepare_day(i_day, env)` | Before day loop | Injects `brand_momentum` into env, seeds RNGs, returns `{agent: alpha}` |
| `record_day(i_day, env, acc, reward)` | After `update_brand_momentum()` | Captures `DayState` (BM + meta_obs) for this day |
| `next_iteration(env) → bool` | After all N days | Convergence check; `True` = re-run days with updated `S_pred` |
| `commit(meta_policies) → EpisodeResult` | After convergence | Populates meta PPO buffers; returns diagnostics |

Two extra methods exist for the parallel path (see [Parallel days](#parallel-days)):

| Method | What it does |
|---|---|
| `make_worker_payload(i_day) → WorkerDayPayload` | Runs `_meta_forward` in main (workers can't hold the meta-policy nets), caches the output, returns a small pickleable payload (BM, day_seed, alpha) for a worker. |
| `merge_worker_capture(i_day, capture)` | Replays a worker's `WorkerDayCapture` into `_day_results` exactly as `record_day` would. Call in `i_day` order. |

## Update strategies (`--picard_update_strategy`)

### `anderson` (default, recommended)

Applies the analytic update as `G(x_k)`, then mixes the last `m` outputs using Anderson acceleration: finds the convex combination `θ` of `{G(x_j)}` that minimises the residual norm. Breaks oscillation that the plain analytic iteration can exhibit when the BM→market_share feedback is strong (`bm_gamma` > 1). Converges in K=1–3 at `tol=5e-3`. Controlled by `--picard_anderson_m` (default 5, window size).

### `analytic`

Back-calculates observed market shares from the EMA formula and re-propagates the full BM trajectory analytically in one sequential pass:

```
ms[d] = (BM_out[d] - λ · BM_in[d]) / (1-λ)
BM[d+1] = λ · BM[d] + (1-λ) · ms[d]   for d = 0..N-1
```

Fixes all N days simultaneously rather than one per iteration (pure Jacobi). Converges in K≈2–3 on warm-started episodes. Susceptible to oscillation at high `bm_gamma`; switch to `anderson` or add damping via `--picard_omega`.

### `jacobi`

Pure full-replace: `S^(k+1) = S_new`. Converges in N+1 iterations for N days. Included as a baseline; `analytic` is strictly better.

## Parallel days

When `--picard_parallel_workers > 1`, the N days of each Picard iteration are dispatched to a fork-based `multiprocessing.Pool` instead of running serially.

### What the workers carry

Pool is forked **after** `set_worker_state(...)` populates module-level references, so each worker inherits `env`, `model_agents`, `meta_policies`, `args`, and `solveRebFlow` via copy-on-write. Per-call payload (`WorkerDayPayload`) is just the per-day data: BM to inject, day_seed, pre-sampled meta-alpha.

Workers don't carry a `PicardSolver`. A small `_WorkerShim` quacks like one: its `prepare_day` seeds RNGs + injects BM from the payload; its `record_day` captures a `WorkerDayCapture` (BM-out, meta_obs, meta_reward) that the wrapper feeds into the central solver's `merge_worker_capture` in i_day order.

### Gradient flow

The hard part. PyTorch autograd graphs don't survive pickling, so a worker can't ship `SavedAction(log_prob, value)` tensors back — `backward()` in main would see leaf tensors and silently produce zero gradients.

Instead, each worker runs the math of `A2C.training_step` on its own day's buffer **as a sum** (not a mean), calls `backward()` locally, and ships per-parameter detached CPU grad tensors back. Main sums across days and divides by total transitions:

```
param.grad = (Σ_d worker_grad_d) / total_steps
```

By linearity, this is identical to `∇(mean ℓ)` from sequential `training_step`, to within floating-point reduction order.

### Per-episode weight broadcast

After main's `optimizer.step()`, workers' copies of `model_agents` are stale. Before each episode's parallel work, `broadcast_state_dicts(pool, N, model_agents, fix_agent)` pushes the fresh state_dicts out (one apply per worker via `chunksize=1`).

### Determinism

Workers seed `np.random` with `payload.day_seed` *before* calling `env.reset()` so the demand realisation is deterministic across Picard iterations within an episode — without this, two iterations would see different demand for the same `i_day` and the fixed point wouldn't exist. `day_runner.run_day` skips its `env.reset_day()` for `i_day > 0` when `ctx.env_pre_reset` is set, so the worker's seeded reset is the only one.

### BLAS thread limits

The pool initializer pins each worker to a single thread (`torch.set_num_threads(1)` + `OMP/OPENBLAS/MKL_NUM_THREADS=1`). Without this, N workers × N BLAS threads oversubscribe the CPU and speedup collapses.

## Key design decisions

- **Fixed seeds per episode** — `_presample_noise` draws day seeds and meta-policy noise once at episode start. Held constant across Picard iterations so the fixed point is well-defined (same demand, same alpha draws).
- **Warm start** — converged `S_pred` from the previous episode is reused as the initial guess. Keeps K=1 convergence common once training stabilises.
- **Delta on BM only** — convergence is measured as max-norm over `brand_momentum` (days 1..N). `meta_obs` lags by one iteration but stabilises quickly.
- **Anderson history reset per episode** — `_anderson_history` is cleared in `begin_episode` so mixing never leaks across episodes.
- **Days stay independent** — all update logic runs *after* all N days finish. Individual day simulations are never modified, which is what makes the parallel dispatch a drop-in.
- **Only the final iteration trains** — workers compute gradients on every Picard iteration; main keeps overwriting `last_parallel_results`. After convergence, only the surviving (final) iteration's grads reach `apply_aggregated_gradients`.

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--parallel_days` | False | Enable Picard fixed-point iteration. Requires `--num_days > 1`. |
| `--picard_max_iters` | 10 | Maximum Picard iterations per episode. With `anderson` at `tol=5e-3`, K is usually 1–3; this is the safety cap. |
| `--picard_tol` | 5e-3 | Convergence tolerance (max-norm on BM, days 1..N). 5e-3 sits just above the discrete demand noise floor (~4.5e-4) — tighter tolerances waste iterations without changing learning. |
| `--picard_update_strategy` | `anderson` | `anderson` (default), `analytic`, or `jacobi`. See [Update strategies](#update-strategies---picard_update_strategy). |
| `--picard_anderson_m` | 5 | Window size for Anderson acceleration. Larger = more history, but diminishing returns past ~5. |
| `--picard_omega` | 1.0 | Damping factor for analytic/anderson updates: `BM_new = (1-ω)·BM_old + ω·BM_proposed`. 1.0 = no damping; set to 0.5–0.8 only if you observe oscillation with `anderson`. |
| `--picard_parallel_workers` | 1 | Number of worker processes for parallel day execution within a Picard iteration. 1 = sequential. Only effective when `--parallel_days` is set. |

## WandB diagnostics (`debug/` panel)

| Metric | What to watch for |
|---|---|
| `debug/picard_K_used` | Should stay 1–3 with Anderson; spikes to `picard_max_iters` indicate oscillation — switch strategy or add damping. |
| `debug/picard_converged` | Frequent 0s mean tol is too tight or max_iters too low. |
| `debug/picard_delta_i1` | First-iteration residual; trends down as warm start improves. |
| `debug/picard_final_delta` | Residual on exit; hovering at ~4.5e-4 is the discrete demand noise floor. |
