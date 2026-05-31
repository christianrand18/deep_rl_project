# Parallel days — implementation plan

Goal: dispatch the N days of a Picard iteration across worker processes so the
~0.1–0.2 s per day overlaps. Only meaningful under `--parallel_days` (Picard
already pre-decouples the days; sequential mode is causally chained).

## Architecture

```
main process                          worker processes (× N_workers)
─────────────                          ─────────────────────────────
PicardSolver           ── payload ──►  _WorkerShim (quacks like PicardSolver)
model_agents (truth)   ── state_dict ► model_agents (synced once/episode)
                                        run_day(i_day, ctx)
                                        local backward() → grad tensors
result aggregator      ◄── capture ──   WorkerDayCapture (per-day state)
                       ◄── ctx delta ── episode_* + actions_* deltas
                       ◄── grad pack ── per-parameter CPU grad tensors
                       ◄── day_logs ──  list of wandb dicts
optimizer.step()
```

## File layout

| File | Change |
|---|---|
| `src/algos/picard/solver.py` | Add `WorkerDayPayload`, `WorkerDayCapture` dataclasses + `make_worker_payload(i_day)` and `merge_worker_capture(i_day, capture)` methods on `PicardSolver`. Reset a new `_meta_out_by_day` cache in `begin_episode` and `next_iteration`. |
| `src/algos/picard/parallel.py` | New. Owns the pool, the shim, the worker entry point, gradient aggregation, ctx-delta merge. |
| `src/algos/picard/__init__.py` | Re-export the new public names. |
| `src/misc/day_runner.py` | Two one-liners: `wandb.log(day_log)` → `ctx.day_logs.append(day_log)`; gate the `if i_day > 0: env.reset_day()` block on `not getattr(ctx, 'env_pre_reset', False)`. Remove the now-unused `import wandb`. |
| `main_a2c_multi_agent.py` | Create pool once, broadcast weights per episode, branch sequential vs parallel inside the rerun loop, drain `ctx.day_logs` to wandb after the day work, branch `training_step` vs `apply_aggregated_gradients` after the rerun loop, close pool on exit. |
| `src/arguments.py` | Add `--picard_parallel_workers` (int, default 1). |

## Steps

### 1. Solver bridge

```python
# solver.py — alongside DayState / DayResult
@dataclass
class WorkerDayPayload:
    i_day: int
    bm_in: dict          # {agent: float}
    day_seed: int
    meta_alpha: dict     # {agent: float}
    num_days: int
    reward_scalar: float
    agents: list

@dataclass
class WorkerDayCapture:
    bm_out: dict
    meta_obs: dict
    meta_reward: dict
```

`make_worker_payload(i_day)` calls `_meta_forward` (needs the meta policies →
can't run in worker), caches the result in `self._meta_out_by_day[i_day]`, and
returns the payload. `merge_worker_capture(i_day, capture)` appends to
`self._day_results` exactly like `record_day` would, reading `meta_obs_in`
from `self._S_pred[i_day]` and `meta_out` from the cache.

### 2. The worker shim

```python
class _WorkerShim:
    """Per-day stand-in for PicardSolver inside a worker."""
    def prepare_day(self, i_day, env):
        env.brand_momentum = dict(self._payload.bm_in)
        np.random.seed(self._payload.day_seed)
        env._shuffle_rng = random.Random(self._payload.day_seed)
        torch.manual_seed(self._payload.day_seed)
        return dict(self._payload.meta_alpha)
    def record_day(self, i_day, env, accumulator, meta_reward):
        # Capture the WorkerDayCapture into self.captured; do not mutate central.
```

`run_day` inside a worker calls these via `ctx.picard_solver`. Workers never
touch the real `PicardSolver`.

### 3. Worker pool

- `multiprocessing.get_context('fork').Pool(N, initializer=…)`. Fork is fast
  and inherits `model_agents` / `env` / etc. via copy-on-write, so the per-call
  payload stays small.
- Module-level dict `_WORKER_STATE` populated **before** the pool is created
  (workers inherit it). Holds `env`, `model_agents`, `args`, `meta_policies`,
  `solveRebFlow`.
- Initializer: `torch.set_num_threads(1)` so N workers don't oversubscribe.

### 4. Worker entry point

```python
def _worker(task):
    i_episode, payload = task
    np.random.seed(payload.day_seed)          # deterministic env.reset demand
    env.reset()
    for a in agents_not_fix:
        model_agents[a].rewards = []
        model_agents[a].saved_actions = []
    ctx = build_worker_ctx(env, payload, i_episode)   # zero accumulators
    run_day(payload.i_day, ctx)
    grad_packets = {a: compute_partial_gradients(model_agents[a])
                    for a in agents_not_fix}
    return WorkerResult(i_day, shim.captured, strip(ctx), grad_packets, ctx.day_logs)
```

### 5. Gradient flow (the hard bit — see gotchas)

Each worker runs `training_step`'s math on its own day's `saved_actions` /
`rewards` but as a **sum** (not mean) and calls `backward()` locally. It snapshots
each parameter's `.grad.detach().cpu().clone()` into a `GradPacket`.

Main:
- sums grad tensors across days
- divides by `total_steps = Σ n_steps` (recovers the `mean()` from
  sequential `training_step`)
- assigns to `param.grad`, calls `clip_grad_norm_`, then `optimizer.step()`
- `del agent.rewards[:]`, `del agent.saved_actions[:]`

This is correct because returns are already day-bounded in sequential code (the
`% max_steps == 0` terminal reset), so per-day partial losses sum to the same
total loss.

### 6. Main glue

```python
# once, before the episode loop
set_worker_state(env=env, model_agents=model_agents, args=args, …)
pool = make_pool(args.picard_parallel_workers) if parallel else None

for i_episode in epochs:
    env.reset()
    picard_solver.begin_episode(i_episode, env)
    if pool: broadcast_state_dicts(pool, N, model_agents, fix_agent)

    last_results = None
    rerun = True
    while rerun:
        # … existing per-iteration accumulator init …
        ctx = SimpleNamespace(…, day_logs=[])
        if pool:
            last_results = run_days_parallel(ctx, picard_solver, pool)
        else:
            for i_day in range(args.num_days):
                run_day(i_day, ctx)
        for log in ctx.day_logs: wandb.log(log)
        ctx.day_logs = []
        # … existing unpack + next_iteration … 

    # Training: replace the per-agent training_step block with:
    if pool:
        grad_norms = apply_aggregated_gradients(last_results, model_agents, args, update_actor)
    else:
        # existing sequential training_step loop
```

At program exit: `pool.close(); pool.join()`.

## Technical gotchas

| Gotcha | Why it bites | What to do |
|---|---|---|
| **Autograd graph doesn't survive pickling.** | Workers can't ship `SavedAction(log_prob, value)` back — main's `backward()` would silently see leaf tensors and produce zero gradients. | Workers backward locally, ship CPU **grad tensors**. Main sums and steps. |
| **Workers' weights go stale after main's optimizer step.** | After episode 1's `optimizer.step()`, main and workers diverge. Day-2 workers would compute gradients from old weights. | `broadcast_state_dicts(pool, N, model_agents, fix_agent)` once per episode, *before* the rerun loop. `chunksize=1, n_tasks=N` guarantees one apply per worker. |
| **`env.reset()` calls `scenario.get_random_demand()` which consumes RNG.** Without seeding, two Picard iterations sample different demand for the same day, so the fixed point doesn't exist. | The `_day_seeds` already make the *within-day* simulation deterministic via `prepare_day`, but `env.reset()` runs first. | `np.random.seed(payload.day_seed)` **before** `env.reset()` in the worker. |
| **`run_day` calls `env.reset_day()` again for `i_day > 0`.** | After the worker already did `env.reset()`, the inner reset would consume more RNG and shift state. | Gate the inner reset on `not getattr(ctx, 'env_pre_reset', False)`. Set `ctx.env_pre_reset = True` in the worker. |
| **`wandb.log` from worker processes is broken / races.** | Children don't have the parent's wandb run context. | Replace the single `wandb.log(day_log)` in `day_runner.py` with `ctx.day_logs.append(day_log)`. Caller drains. |
| **N workers each spawn N torch threads → oversubscription.** | Default thread pool fights for cores. | Pool initializer: `torch.set_num_threads(1)`. |
| **macOS + `fork` after CUDA init.** | Known to deadlock. | We're CPU-only here (`args.cuda` defaults to False). Fork is fine. HPC target is Linux anyway. |
| **Which iteration's gradients get applied?** | Picard runs K iterations; only the *final* (converged) one should train. Workers compute grads every iteration, but we only want the last. | Have `run_days_parallel` return the results list. Main keeps overwriting `last_results = run_days_parallel(...)` inside the rerun loop. Only `last_results` reaches `apply_aggregated_gradients` after the loop exits. |
| **Per-day deltas vs in-place mutation.** | Workers can't share Python objects with main, so the "central ctx accumulators get mutated in place" trick from the sequential refactor doesn't work. | Workers initialise their ctx with **zeroed** accumulators. Final ctx values **are** the per-day deltas. Wrapper merges by summing scalars, extending lists in day order, and reducing min/max. |
| **`fix_agent` cleanup.** | Sequential code clears its buffers without training. Workers do the same on their copy, but main's `fix_agent` buffers need clearing too for parity. | `apply_aggregated_gradients` clears `model_agents[fix_agent].rewards / saved_actions` and returns zero metrics for that agent. |
| **Sort worker results by `i_day` before merging.** | `pool.map` preserves order, but be defensive. | `results.sort(key=lambda r: r.i_day)` before iterating. |

## Verification

Before training a real run, smoke-test:
1. `python -c "from src.algos.picard import run_days_parallel; print('import ok')"`
2. Run with `--picard_parallel_workers 1` first — exercises the new code path but with a single worker; results should be ~identical to sequential modulo RNG ordering.
3. Run with `--picard_parallel_workers 4`, 10 episodes. Confirm wandb shows the same metric shapes (no nans, sensible reward curves) and that `debug/picard_K_used` still converges.

If things look right, scale up to a full run.
