# Parallel days — implementation plan

## Goal

Dispatch the N days of a Picard iteration across worker processes so the
~0.1–0.2 s per day overlap. Only meaningful under `--parallel_days` — Picard
already pre-decouples the days (BM trajectory, day seeds, and meta-policy noise
are all sampled up front). Sequential mode is causally chained and can't be
parallelised.

## Architecture in one picture

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

The main process stays the source of truth for everything stateful
(`PicardSolver`, `model_agents`, optimizers, wandb). Workers do the heavy
day simulation and ship results back as plain data.

## File layout

| File | Change |
|---|---|
| `src/misc/day_runner.py` | **(Step 1)** New. Single callable `run_day(i_day, ctx)` containing the body of the for-day loop. |
| `src/algos/picard/solver.py` | Add `WorkerDayPayload`, `WorkerDayCapture` dataclasses + `make_worker_payload(i_day)` and `merge_worker_capture(i_day, capture)` methods. Reset a new `_meta_out_by_day` cache in `begin_episode` and `next_iteration`. |
| `src/algos/picard/parallel.py` | New. Owns the pool, the shim, the worker entry point, gradient aggregation, and ctx-delta merge. |
| `src/algos/picard/__init__.py` | Re-export the new public names. |
| `src/misc/day_runner.py` (touch-up) | Two one-liners after the initial extract: `wandb.log(day_log)` → `ctx.day_logs.append(day_log)`; gate the `if i_day > 0: env.reset_day()` block on `not getattr(ctx, 'env_pre_reset', False)`. |
| `main_a2c_multi_agent.py` | Create pool once, broadcast weights per episode, branch sequential vs parallel inside the rerun loop, drain `ctx.day_logs` to wandb, branch `training_step` vs `apply_aggregated_gradients`, close pool on exit. |
| `src/arguments.py` | Add `--picard_parallel_workers` (int, default 1). |

---

## Step 1 — Extract the day loop into a callable

Everything else hangs off this. Get this right and the rest is plumbing.

### What you're moving

In `main_a2c_multi_agent.py` find the block `for i_day in range(args.num_days):`
inside the `while rerun:` loop. The *body* of that for-loop is the day simulation
— roughly 430 lines covering all four `env.mode` branches, the inner step loop,
the `update_brand_momentum` call, and the per-day Picard hooks (`prepare_day`,
`record_day`) plus the `wandb.log(day_log)` at the end. That entire body is what
moves into `src/misc/day_runner.py::run_day(i_day, ctx)`.

The for-loop body is at four levels of indent (16 spaces) in main. In the new
function it lives at one level of indent (4 spaces). So **dedent the entire body
by 12 spaces** when copying.

### Why `src/misc` and not `src/algos/picard`

The day-sim loop has nothing to do with Picard — sequential runs go through it
too. Picard just *uses* it as a sub-call. `src/misc/` is the right neighbour for
generic training utilities.

### Why `ctx` and not many positional args

The body reads and writes ~25 distinct accumulators plus `env`, `args`,
`model_agents`, `meta_policies`, `accumulator`, `picard_solver`, `solveRebFlow`,
`i_episode`. Passing all of those as parameters would be miserable. A single
`SimpleNamespace` carries them through.

There's a subtle distinction that drives the design:

- **Reassigned** values (`obs = env.reset_day()`, `episode_reward = {a: ...}`,
  `meta_multipliers = picard.prepare_day(...)`, `episode_rejected_demand += ...`):
  the function's local name gets rebound, so the caller would never see the
  change unless we write it back. There are exactly **seven** of these:
  `obs`, `action_rl`, `episode_reward`, `meta_obs`, `meta_multipliers`,
  `episode_rejected_demand`, `episode_total_demand`.
- **In-place mutations** (`episode_served_demand[a] += info[a][...]`,
  `actions_price[a].append(...)`, `episode_min_concentration_alpha[a] = min(...)`):
  the dict/list object is shared by reference; mutation is visible to the
  caller without write-back.

### The pattern

```python
# src/misc/day_runner.py
import numpy as np
from src.misc.utils import dictsum

def run_day(i_day, ctx):
    """Run one day. Reads/writes ctx in place.

    Reassigned scalars/dicts are written back to ctx at the end; everything
    else is mutated in place via the shared dict/list references in ctx.
    """
    # --- Unpack so the body below is a verbatim copy from main ---
    env             = ctx.env
    args            = ctx.args
    model_agents    = ctx.model_agents
    meta_policies   = ctx.meta_policies
    accumulator     = ctx.accumulator
    picard_solver   = ctx.picard_solver
    solveRebFlow    = ctx.solveRebFlow
    i_episode       = ctx.i_episode

    obs              = ctx.obs
    action_rl        = ctx.action_rl
    meta_obs         = ctx.meta_obs
    meta_multipliers = ctx.meta_multipliers
    episode_reward   = ctx.episode_reward
    # ... (every other accumulator the body references) ...

    # Mode-specific trackers may be absent in ctx; pull defensively.
    actions_concentration_alpha = getattr(ctx, 'actions_concentration_alpha', None)
    # ... (the other 8 mode-specific ones) ...

    # === BEGIN verbatim copy of the for-day body (dedented 12 spaces) ===
    if i_day > 0:
        obs = env.reset_day()
        action_rl = {a: [0.0] * env.nregion for a in [0, 1]}
    # ... body untouched ...
    # === END verbatim copy ===

    # --- Write back the seven reassigned values ---
    ctx.obs                     = obs
    ctx.action_rl               = action_rl
    ctx.meta_obs                = meta_obs
    ctx.meta_multipliers        = meta_multipliers
    ctx.episode_reward          = episode_reward
    ctx.episode_rejected_demand = episode_rejected_demand
    ctx.episode_total_demand    = episode_total_demand
```

The "verbatim copy" promise is the whole point: don't refactor the body, just
move it. That way the diff is auditable line-by-line and you can trust that
sequential behaviour is identical.

### Caller change in `main_a2c_multi_agent.py`

Right after the per-iteration accumulator init (inside `while rerun:`, just
before the existing for-day loop):

```python
ctx = SimpleNamespace(
    env=env, args=args, model_agents=model_agents,
    meta_policies=meta_policies, accumulator=accumulator,
    picard_solver=picard_solver, solveRebFlow=solveRebFlow,
    i_episode=i_episode,
    obs=obs, action_rl=action_rl,
    meta_obs=meta_obs, meta_multipliers=meta_multipliers,
    episode_reward=episode_reward,
    # ... all the other accumulators ...
)
# Attach mode-specific concentration trackers conditionally
if env.mode == 0:
    ctx.actions_concentration_dirichlet = actions_concentration_dirichlet
    # ... etc ...
elif env.mode == 1:
    # ...
else:  # mode 2
    # ...

for i_day in range(args.num_days):
    run_day(i_day, ctx)

# Pull reassigned values back so post-loop code reads unchanged
obs                     = ctx.obs
action_rl               = ctx.action_rl
meta_obs                = ctx.meta_obs
meta_multipliers        = ctx.meta_multipliers
episode_reward          = ctx.episode_reward
episode_rejected_demand = ctx.episode_rejected_demand
episode_total_demand    = ctx.episode_total_demand
```

### Things to watch while extracting

1. **`solveRebFlow` is conditionally bound** in main (ortools vs pulp at the
   top of the script). It must go through `ctx`; don't import it inside
   `day_runner.py`.
2. **`dictsum`, `np`** are referenced in the body. Import them in
   `day_runner.py` (don't try to pull `dictsum` from ctx).
3. **`wandb`** is also referenced — for the single `wandb.log(day_log)` at the
   end of the body. Import it for now; step 4 will replace that line with a
   queue.
4. The body of the for-day loop *ends* with `wandb.log(day_log)`. The comment
   line right after it (`# ── end of for i_day ──`) is at a shallower indent
   and is NOT part of the body — don't include it in the extraction.
5. **Verify**: run the existing batch jobs locally for a few episodes both
   before and after the extract. WandB curves should be bit-identical.
6. **Don't change behaviour in this step.** Even if you spot something to
   clean up in the body, defer it. The whole value of the extraction is that
   it's mechanical.

Once this is in and clean, the rest of the work just feeds different `ctx`
objects into the same callable.

---

## Step 2 — Solver bridge

Add two small dataclasses and two methods on `PicardSolver`:

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

`make_worker_payload(i_day)` runs `_meta_forward` *in the main process* (it
needs the meta-policy networks, which workers don't have), caches the result
in `self._meta_out_by_day[i_day]`, and returns the payload. The cache must be
cleared in `begin_episode` and at the end of `next_iteration` — same lifetime
as `_day_results`.

`merge_worker_capture(i_day, capture)` is the worker-side replacement for
`record_day`. It appends to `self._day_results` exactly like `record_day`
would, reading `meta_obs_in` from `self._S_pred[i_day]` and `meta_out` from
the cache. Call it **once per day in i_day order** so `_day_results` stays
day-ordered.

## Step 3 — The worker shim

A small class that quacks like `PicardSolver` from the worker's point of view:

```python
class _WorkerShim:
    """Per-day stand-in for PicardSolver inside a worker."""
    def __init__(self, payload):
        self._payload = payload
        self.captured = None

    def prepare_day(self, i_day, env):
        env.brand_momentum = dict(self._payload.bm_in)
        np.random.seed(self._payload.day_seed)
        env._shuffle_rng = random.Random(self._payload.day_seed)
        torch.manual_seed(self._payload.day_seed)
        return dict(self._payload.meta_alpha)

    def record_day(self, i_day, env, accumulator, meta_reward):
        accumulator.momentum_snapshot = dict(env.brand_momentum)
        self.captured = WorkerDayCapture(
            bm_out=dict(env.brand_momentum),
            meta_obs={a: accumulator.daily_state(a, i_day + 1,
                         self._payload.num_days, self._payload.reward_scalar)
                      for a in self._payload.agents},
            meta_reward=dict(meta_reward),
        )
```

Workers receive this in their `ctx.picard_solver`. They never touch the real
`PicardSolver`. After the day runs, the wrapper reads `shim.captured` and feeds
it to `central_picard.merge_worker_capture(...)`.

## Step 4 — Worker pool plumbing

- Use `multiprocessing.get_context('fork').Pool(N, initializer=…)`. Fork is
  fast and inherits the heavy stuff (env, model_agents, meta_policies) via
  copy-on-write, so per-call payloads stay tiny.
- Stash references the workers need (`env`, `model_agents`, `args`,
  `meta_policies`, `solveRebFlow`) into a module-level dict
  (`_WORKER_STATE`) **before** creating the pool. Workers inherit the dict
  through fork.
- Pool initializer should call `torch.set_num_threads(1)` to stop N workers ×
  N torch threads from oversubscribing the CPU.

Also flip the two day_runner.py one-liners now:

- `wandb.log(day_log)` → `ctx.day_logs.append(day_log)`. Initialise
  `ctx.day_logs = []` in the caller; drain after the day work.
- `if i_day > 0:` → `if i_day > 0 and not getattr(ctx, 'env_pre_reset', False):`
  so workers can opt out of the inner `env.reset_day()` after doing their own
  seeded reset.

## Step 5 — Worker entry point

```python
def _worker(task):
    i_episode, payload = task
    np.random.seed(payload.day_seed)        # deterministic env.reset() demand
    env.reset()
    for a in agents_not_fix:
        model_agents[a].rewards = []
        model_agents[a].saved_actions = []
    ctx = build_worker_ctx(env, payload, i_episode)   # all accumulators zeroed
    ctx.env_pre_reset = True                # tell run_day to skip its reset_day
    run_day(payload.i_day, ctx)
    grad_packets = {a: compute_partial_gradients(model_agents[a])
                    for a in agents_not_fix}
    return WorkerResult(payload.i_day, shim.captured, strip(ctx),
                        grad_packets, ctx.day_logs)
```

`build_worker_ctx` constructs a fresh `SimpleNamespace` with **zeroed**
episode_* accumulators and empty lists. After `run_day` returns, those values
are the per-day deltas (since they started at zero). `strip(ctx)` keeps only
the pickleable fields (no env, no model_agents, no shim — those are
process-local).

## Step 6 — Gradient flow (the trickiest bit)

The autograd graph doesn't survive pickling. `SavedAction(log_prob, value)`
tensors shipped from worker to main would lose `grad_fn`; main's `backward()`
would silently produce zero gradients and training would quietly break.

Solution: workers run the math of `training_step` on their own day's buffer,
call `backward()` locally, and ship the resulting **per-parameter CPU grad
tensors** — those pickle fine because they're just numbers.

Worker side:

```python
def compute_partial_gradients(agent):
    """Same math as A2C.training_step, but sum (not mean), single-day buffer."""
    agent.actor.zero_grad(); agent.critic.zero_grad()
    returns = compute_returns(agent.rewards, agent.gamma) / agent.reward_scale
    policy_losses = [-log_prob * adv for ((log_prob, _), adv) in ...]
    value_losses  = [F.smooth_l1_loss(value, R) for ...]
    torch.stack(policy_losses).sum().backward()
    torch.stack(value_losses).sum().backward()
    return GradPacket(
        n_steps=len(agent.rewards),
        actor_grads={n: p.grad.detach().cpu().clone()
                     for n, p in agent.actor.named_parameters() if p.grad is not None},
        critic_grads={...same for critic...},
        p_loss_sum=…, v_loss_sum=…, advantages=…,
    )
```

Main side, after the rerun loop converges:

```python
def apply_aggregated_gradients(packets, agent, update_actor):
    total_steps = sum(p.n_steps for p in packets)
    for name, param in agent.actor.named_parameters():
        if name in packets[0].actor_grads:
            stacked = sum(p.actor_grads[name] for p in packets)
            param.grad = (stacked / total_steps).to(param.device)
    # ... same for critic ...
    if update_actor:
        clip_grad_norm_(agent.actor.parameters(), agent.actor_clip)
        agent.optimizers['a_optimizer'].step()
    clip_grad_norm_(agent.critic.parameters(), agent.critic_clip)
    agent.optimizers['c_optimizer'].step()
    del agent.rewards[:]; del agent.saved_actions[:]
```

Why this is correct: sequential `training_step` uses `mean()` over all
transitions in the episode. Total mean loss = (Σ per-day partial losses) /
N_total. Gradients are linear, so Σ per-day grads / N_total = grad of the
total mean loss. The per-day sums on workers + central division by total
steps reproduce that exactly.

## Step 7 — Main glue

```python
# once, before the episode loop
set_worker_state(env=env, model_agents=model_agents, args=args, ...)
pool = make_pool(args.picard_parallel_workers) if parallel else None

for i_episode in epochs:
    env.reset()
    picard_solver.begin_episode(i_episode, env)
    if pool: broadcast_state_dicts(pool, N, model_agents, fix_agent)

    last_results = None
    rerun = True
    while rerun:
        # ... existing per-iteration accumulator init ...
        ctx = SimpleNamespace(..., day_logs=[])
        if pool:
            last_results = run_days_parallel(ctx, picard_solver, pool)
        else:
            for i_day in range(args.num_days):
                run_day(i_day, ctx)
        for log in ctx.day_logs:
            wandb.log(log)
        ctx.day_logs = []
        # ... existing unpack + next_iteration ...

    # Training step branches on path
    if pool:
        grad_norms = apply_aggregated_gradients(last_results, model_agents, args, update_actor)
    else:
        # existing sequential training_step loop unchanged
```

At program exit: `pool.close(); pool.join()`.

---

## Technical gotchas (keep these handy)

| Gotcha | Why it bites | What to do |
|---|---|---|
| **Autograd graph doesn't survive pickling.** | Workers can't ship `SavedAction(log_prob, value)` back — main's `backward()` would see leaf tensors and produce zero gradients. Silent training failure. | Workers `backward()` locally, ship CPU grad tensors. Main sums and steps. |
| **Workers' weights go stale after main's optimizer step.** | After episode 1's `optimizer.step()`, main and workers diverge. Day-2 workers would use old weights. | `broadcast_state_dicts(pool, N, model_agents, fix_agent)` once per episode, *before* the rerun loop. Use `chunksize=1, n_tasks=N` so each worker gets exactly one apply call. |
| **`env.reset()` consumes RNG via `scenario.get_random_demand()`.** | Two Picard iterations would sample different demand for the same day → no fixed point to converge to. | `np.random.seed(payload.day_seed)` *before* `env.reset()` in the worker. The day's `prepare_day` re-seeds for the within-day simulation. |
| **`run_day` calls `env.reset_day()` again for `i_day > 0`.** | After the worker already did `env.reset()`, the inner reset would consume more RNG and shift state. | Gate the inner reset on `not getattr(ctx, 'env_pre_reset', False)`. Set `ctx.env_pre_reset = True` in the worker. |
| **`wandb.log` from worker processes is broken.** | Children don't have the parent's wandb run context. | Replace `wandb.log(day_log)` in `day_runner.py` with `ctx.day_logs.append(day_log)`. Caller drains. |
| **N workers × N torch threads → oversubscription.** | Default thread pool fights for cores; speedup collapses. | Pool initializer: `torch.set_num_threads(1)`. |
| **`fork` after CUDA init can deadlock on macOS.** | Known PyTorch + fork interaction. | We're CPU-only here (`args.cuda` defaults to False). Fork is fine. HPC target is Linux anyway. |
| **Which Picard iteration's gradients get applied?** | Picard runs K iterations per episode; only the *final* (converged) one should train. | Workers compute grads every iteration but main only keeps `last_results = run_days_parallel(...)`. The overwrite inside the rerun loop drops earlier iterations naturally. Only `last_results` reaches `apply_aggregated_gradients`. |
| **Per-day deltas vs in-place mutation.** | Workers can't share Python objects with main, so the "ctx accumulators get mutated in place" trick from Step 1 doesn't work across the process boundary. | Workers initialise their ctx with **zeroed** accumulators. Final ctx values *are* the per-day deltas. Wrapper merges by summing scalars, extending lists in day order, and reducing min/max dicts. |
| **`fix_agent` cleanup parity.** | Sequential code clears fix_agent's buffers without training. Workers do the same on their copy, but main's fix_agent buffers also need clearing. | `apply_aggregated_gradients` clears `model_agents[fix_agent].rewards / saved_actions` and returns zero metrics for that agent. |
| **Sort worker results by `i_day` before merging.** | `pool.map` preserves order, but be defensive — and list-extends are order-sensitive. | `results.sort(key=lambda r: r.i_day)` before iterating. |
| **`meta_obs_in` for `merge_worker_capture`.** | `record_day` reads `self._current_prev_state.meta_obs[a]`, which is set inside `prepare_day` on the central solver. Workers run `prepare_day` on the shim, so the central solver's `_current_prev_state` is stale. | In `merge_worker_capture`, read `self._S_pred[i_day].meta_obs[a]` directly instead of `self._current_prev_state.meta_obs[a]`. The values are equal but the source is explicit and doesn't depend on side effects. |

## Verification path

Before scaling up:

1. **Imports work**: `python -c "from src.algos.picard import run_days_parallel; print('ok')"`.
2. **Sequential parity after step 1**: run a short experiment (say 50 episodes,
   7 days, anderson) before and after the day_runner extraction. WandB curves
   should be bit-identical.
3. **Workers=1**: run with `--picard_parallel_workers 1` for 10 episodes.
   Exercises the new code path with a single worker. Results should be
   ~identical to sequential modulo RNG ordering.
4. **Workers=4**: same run with 4 workers. `debug/picard_K_used` should still
   converge in 2–3 iterations. Reward/loss curves should look statistically
   indistinguishable (it's the same math; RNG state ordering differs).
5. Only then submit a full HPC run.
