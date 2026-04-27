# Project Extension Plan: Hierarchical Multi-Agent AMoD with Brand Momentum

**Focus:** Multi-agent (duopoly) setting — `main_a2c_multi_agent.py` / `src/algos/a2c_gnn_multi_agent.py` / `src/envs/amod_env_multi.py`

---

## Overview

We extend the existing A2C-GNN duopoly system with three major additions:

1. **Brand momentum** in the passenger utility function (persistence proxy for loyalty)
2. **Hierarchical two-level MDP** with a meta-policy operating at a coarser timescale
3. **Longer simulation horizon** to support the two-level structure

These map onto GitHub issues #1–6.

---

## Issue #7 — Get Out-of-the-Box Code to Run

**Goal:** Verify the baseline multi-agent code runs end-to-end before any extensions.

**Steps:**
1. Clone the repo locally, create `venv`, install `requirements.txt`.
2. Confirm CPLEX path is accessible.
3. Run a short smoke test:
   ```bash
   python main_a2c_multi_agent.py \
     --city nyc_man_south --mode 2 --max_steps 10 --max_episodes 3 \
     --checkpoint_path smoke_test --cplexpath <path>
   ```
4. Verify W&B logging works (`.env` has `WANDB_API_KEY`).
5. **Done when:** Training loop completes without error for 3 episodes.

---

## Issue #5 — Investigate & Fix Simulation Timespan

### Current State
- Default: `max_steps=20`, `json_tstep=3` min → **60 minutes** per episode.
- Scenario data starts at `json_hr=19` (7 PM). With 20 steps the agent only sees 1 hour.
- "~8 hours" requires `max_steps=160` (160 × 3 min = 480 min = 8 h).

### Problem
Extending `max_steps` alone makes training slow and gradient variance very high (long return horizon). The meta-policy introduces a natural two-level MDP that addresses this.

### Solution: Two-Level MDP Time Structure

```
Level 1 — Meta-policy:    runs every H low-level steps (e.g. H=20, = 1 hour)
Level 2 — Low-level A2C:  runs every timestep (3 min)
```

- Total simulation = K meta-steps × H low-level steps (e.g. K=8 × H=20 = 160 steps = 8 h)
- Each meta-step, the meta-policy emits a *price reference* scalar that the low-level policy is nudged toward.
- The low-level reward at each timestep is unchanged; the meta-policy reward is accumulated over the H low-level steps.

**Implementation changes:**
- Add `--meta_interval` argument (default: 20 steps = 1 hour) to `main_a2c_multi_agent.py`.
- Add `--num_meta_steps` argument (default: 8, giving 8 × 20 = 160 total steps).
- The outer loop iterates over meta-steps; the inner loop iterates over `meta_interval` low-level steps.
- For now, keep `max_steps = meta_interval * num_meta_steps` for backward compatibility.

**Optional:** "Only care about subset of hours" — add a `--active_hours` list flag (e.g. `[0, 1, 7]` to train only on hours 0, 1, and 7 of the 8-hour window). Non-active hours use a fixed/frozen policy to advance the simulation state without gradient updates.

---

## Issue #2 — Implement Brand Momentum in Utility Function

### Problem
There is no persistent passenger ID across timesteps, so explicit loyalty modelling is impossible. We use **brand momentum** as a population-level proxy.

### Design

**Exponential smoothing of demand share:**

At each timestep `t`, let:
- `s_a(t)` = share of total served demand captured by agent `a` in the last timestep
- `m_a(t)` = brand momentum for agent `a`

Update rule:
```
m_a(t) = λ · m_a(t-1) + (1-λ) · s_a(t)
```
where `λ ∈ [0, 1)` is a smoothing parameter (default: `λ = 0.9`).

**Modified utility function** (`amod_env_multi.py`, `match_step_simple`):

```python
# Current (before):
U_0 = choice_intercept - 0.71*wage*travel_time_h - income_effect*price_mult*pr0
U_1 = choice_intercept - 0.71*wage*travel_time_h - income_effect*price_mult*pr1

# After (brand bonus):
U_0 = choice_intercept - 0.71*wage*travel_time_h - income_effect*price_mult*pr0 + alpha_brand * m_0
U_1 = choice_intercept - 0.71*wage*travel_time_h - income_effect*price_mult*pr1 + alpha_brand * m_1
```

- `alpha_brand` is a calibration constant (start: 0.5, tune so momentum effect is ~5–15% of baseline utility range).
- Momentum is symmetric: both agents' momentum can improve or decay.
- Momentum is **zone-wise** (per origin region) if demand share can be broken out per region, otherwise city-wide.

### Implementation (`amod_env_multi.py`)
1. Add `brand_momentum` dict `{agent_id: np.zeros(nregion)}` initialized in `__init__`.
2. After each `match_step_simple`, compute per-region served share and update `brand_momentum`.
3. Apply the bonus inside the utility calculation loop.
4. Add `--alpha_brand` and `--brand_lambda` CLI args with defaults.
5. Log `brand_momentum` mean and std to W&B each episode.

### Reference
See "Mathematical psychology" paper: https://arxiv.org/pdf/2512.14713 — focus on the choice model sections on reference-dependent utility and habit formation.

---

## Issues #1, #3, #4 — Implement Meta-Policy

### Architecture Decision

**Chosen approach:** Simple Actor-Critic (A2C) meta-policy — consistent with the low-level learner, simpler than PPO for initial implementation. Can upgrade to PPO later (issue #4 leaves it open).

**Initial scope:** One meta-policy per agent (symmetric). Start with only agent 0 having a trained meta-policy; agent 1 uses a fixed reference price or its own independent meta-policy (toggle with `--fix_meta_agent` flag).

---

### Issue #3 — Meta-Policy Input (State Space)

The meta-policy observes aggregate statistics over the last `H` low-level steps:

| Feature | Description | Shape |
|---|---|---|
| `service_rate_self` | Fraction of demand served by this agent | scalar |
| `avg_price_self` | Mean price scalar used (own) | scalar |
| `avg_price_opponent` | Mean price scalar used (opponent) | scalar |
| `reb_cost_self` | Total rebalancing cost in window | scalar |
| `reb_trips_self` | Total rebalancing trips | scalar |
| `profit_self` | Total profit (revenue − reb cost) in window | scalar |
| `brand_momentum_self` | City-wide brand momentum | scalar |
| `brand_momentum_opp` | Opponent brand momentum | scalar |
| `hour_of_day` | Normalized time-of-day (0–1) | scalar |
| *(optional)* `profit_by_zone` | Per-zone profit breakdown | `nregion` scalars |

**Total input size:** ~9 scalars (+ optional per-zone extension).

This is a simple MLP (no GNN needed at the meta level — the input is already aggregated).

---

### Issue #1 — Meta-Policy Reward

The meta-policy reward is accumulated over the `H` low-level steps it governs:

```
R_meta = Σ_{t in window} r_low(t)   (sum of low-level rewards)
```

**Optional augmentation:** Add a *price-reference* regularization term to the low-level reward:

```
r_low_augmented(t) = r_low(t) + α_meta · f(price_action(t), reference_price_from_meta)
```

where `f` penalizes deviation from the meta-policy's target price. This couples the two levels.

Start without the augmentation (simpler). Add it as a second experiment.

---

### Issue #4 — Meta-Policy Algorithm

**Architecture (`src/algos/meta_policy.py` — new file):**

```python
class MetaActor(nn.Module):
    # Input: aggregate state vector (9 scalars)
    # Hidden: 2-layer MLP, 64 units each
    # Output: Beta distribution parameters → price_reference scalar ∈ (0, 1)
    #         (optionally: zone-wise price scalars → nregion outputs)

class MetaCritic(nn.Module):
    # Input: same aggregate state vector
    # Hidden: 2-layer MLP, 64 units each
    # Output: scalar value V(s_meta)

class MetaA2C(nn.Module):
    # Wraps MetaActor + MetaCritic
    # select_action(): returns price_reference scalar
    # update_policy(): standard A2C update (actor + critic losses)
```

**Output:** A scalar `price_reference ∈ (0, 1)` (same space as the low-level Beta action). This is passed to the low-level policy as a soft target.

**Synchrony:** Both meta-policies run **synchronously** (same timestep boundaries). This is simpler and avoids partial-information asymmetries. Asynchronous execution is a future extension.

**Frequency:** Every `H = meta_interval` low-level steps (default: 20 = 1 hour).

---

### Integration into Training Loop (`main_a2c_multi_agent.py`)

```
for episode in episodes:
    obs = env.reset()
    meta_obs = init_meta_obs()

    for meta_step in range(num_meta_steps):          # outer loop: H-step windows
        price_ref = meta_agent.select_action(meta_obs)   # meta decision
        window_reward = 0

        for t in range(meta_interval):               # inner loop: low-level steps
            action_rl = low_level_agent.select_action(obs, price_ref)
            obs, reward, done = env.step(action_rl)
            window_reward += reward
            low_level_agent.rewards.append(reward)

        meta_agent.rewards.append(window_reward)
        meta_obs = compute_meta_obs(env, window_stats)

    low_level_agent.update_policy()
    meta_agent.update_policy()
```

**Gradient flow:** The two policies are updated independently (no shared gradients). This avoids backpropagation through the long inner loop.

---

## Issue #8 — Parallelization for Computational Efficiency

### Options (in order of implementation complexity)

1. **Vectorized environments (recommended first step):**
   Use `torch.multiprocessing` or `concurrent.futures` to run N independent environment copies in parallel during training. Each worker collects a trajectory; gradients are averaged. This is the standard A3C / parallel A2C approach.

2. **Separate meta and low-level training processes:**
   Meta-policy updates are infrequent (once per `H` steps) so the bottleneck is CPLEX. Run CPLEX calls in a `ProcessPoolExecutor` across regions.

3. **Batch CPLEX calls:**
   Current architecture solves rebalancing LP sequentially per agent. Parallelize agent 0 and agent 1 LP solves using threads (CPLEX is thread-safe per instance).

4. **Profile first:** Before implementing parallelism, profile a 10-episode run with `cProfile` to identify where time is actually spent (CPLEX? GNN forward pass? environment stepping?).

**Recommended order:** Profile → parallelize CPLEX calls (quick win) → vectorized envs (bigger win, more code).

---

## Issue #6 — Post-Hoc Analysis

### Metrics to Compute After Training

| Metric | How to compute |
|---|---|
| **Market share convergence** | Plot each agent's share of served demand over episodes. Do shares stabilize? |
| **Nash equilibrium check** | Fix one agent's policy, optimize other → compare payoff to joint training outcome. |
| **Price dynamics** | Time series of average price per agent per meta-step. Look for oscillation vs. convergence. |
| **Brand momentum effect** | Compare episode rewards with `alpha_brand=0` vs. calibrated. |
| **Meta-policy impact** | Compare runs with/without meta-policy (ablation). |
| **Rebalancing efficiency** | Total rebalancing cost / total revenue as a function of training episodes. |
| **Service rate by zone** | Heatmaps of served demand fraction per region for each agent. |
| **Waiting time distribution** | Histogram of passenger wait times per agent. |
| **Price response curves** | Hold one agent fixed, sweep the other's price reference → demand curve. |

**Implementation:** Add a `--eval_mode post_hoc` flag that loads a checkpoint, runs `N` test episodes, and saves all of the above as CSV / figures to `logs/post_hoc/`.

---

## Implementation Sequence (Recommended Order)

```
Phase 0: Housekeeping
  [#7] Get baseline multi-agent code running end-to-end

Phase 1: Environment Extensions
  [#5] Fix simulation timespan — extend to 8h, add meta_interval/num_meta_steps args
  [#2] Brand momentum in utility function (alpha_brand=0 as default → backward compatible)

Phase 2: Meta-Policy Core
  [#3] Meta-policy state aggregation (collect window stats)
  [#1] Meta-policy reward (sum of low-level rewards over window)
  [#4] MetaA2C network + integration into training loop

Phase 3: Experiments & Analysis
  [#6] Post-hoc analysis suite
  [#8] Parallelization (after profiling)
```

---

## Key Design Decisions & Open Questions

| Question | Recommended Default | Notes |
|---|---|---|
| Meta-policy frequency H | 20 steps (1 hour) | Matches json_tstep=3 min, natural day segmentation |
| Total simulation length | 160 steps (8 hours, 7PM–3AM) | Covers evening peak through night |
| Brand smoothing λ | 0.9 | Decays to 50% after ~7 meta-steps (~7 hours) |
| Brand bonus α_brand | 0.5 | Tune so effect is ~10% of utility range |
| Meta output | Scalar price_reference | Start simple; zone-wise is an extension |
| Meta algorithm | A2C | Consistent with low-level; upgrade to PPO if unstable |
| Synchrony | Synchronous | Both agents update meta-policy at same boundary |
| Fix one meta agent? | Yes, start with fix_meta_agent=1 | Isolate effect before full competition |
| Low-level reward augmentation | Off by default | Toggle with --meta_reward_coupling flag |

---

## New Files to Create

| File | Purpose |
|---|---|
| `src/algos/meta_policy.py` | MetaActor, MetaCritic, MetaA2C classes |
| `src/misc/meta_utils.py` | `compute_meta_obs()`, `aggregate_window_stats()` helpers |
| `scripts/post_hoc_analysis.py` | Post-hoc evaluation and plotting |

## Files to Modify

| File | Changes |
|---|---|
| `src/envs/amod_env_multi.py` | Add brand_momentum state, modify utility function |
| `src/algos/a2c_gnn_multi_agent.py` | Accept price_reference from meta-policy |
| `main_a2c_multi_agent.py` | Add outer meta-loop, CLI args, meta-policy instantiation |

