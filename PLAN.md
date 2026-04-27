# Implementation Plan: HRL + Brand Momentum AMoD Extension

**Spec:** See [SPEC.md](SPEC.md) for full architecture details  
**Issues:** [GitHub Issues](../../issues)

---

## Dependency Graph

```
#7 Set up baseline & checkpoints
    └── #5 Implement multi-day episode structure
            ├── #8 Investigate parallelism          (parallel with Phase 2)
            ├── #2 Brand momentum in utility        ─┐
            ├── #3 Meta-policy input                ─┼─→ #1 Meta-policy reward ─→ #4 Meta-policy architecture
            │                                                        │
            │                                              #9 WandB meta logging
            │
            └── #10 Training Phase 1 (single meta vs fixed)
                        └── #11 Training Phase 2 (competitive meta vs meta)
                                    └── #6 Post-hoc analysis
```

**Critical path:** `#7 → #5 → #2 + #3 → #1 → #4 → #10 → #11 → #6`

---

## Phase 0 — Foundation

### Issue #7: Set up baseline and obtain pre-trained checkpoints
**Goal:** Confirm existing code runs and produce trained low-level policy checkpoints to build on.

- Get `main_a2c_multi_agent.py` running locally and on HPC
- Confirm WandB logging, CPLEX/PuLP solver, and data loading all work
- Train low-level A2C policies to convergence on NYC Man South (fastest city)
- Save checkpoints for both operators

**Done when:** A completed training run exists with converging reward curves and saved checkpoints for both operators.

---

## Phase 1 — Core Infrastructure

### Issue #5: Implement multi-day episode structure
**Goal:** Wrap the existing episode loop so an episode spans N simulated days.

- Outer day loop wraps the existing inner step loop
- Add `partial_reset()` to the environment: resets vehicles and queues, preserves brand momentum state
- Add `--num_days` flag (default 8); `num_days=1` must reproduce existing behaviour exactly
- Key files: `main_a2c_multi_agent.py` (episode loop), `src/envs/amod_env_multi.py` (`reset()`)

**Done when:** A multi-day run completes without errors and `num_days=1` produces identical results to the single-day baseline.

### Issue #8: Investigate parallelism *(runs alongside Phase 2)*
**Goal:** Determine whether episode rollouts can be parallelised, given the LP rebalancing solver.

- Profile a multi-day episode to find the main bottleneck
- Test whether PuLP can safely run in parallel processes
- Recommend a concrete strategy: vectorised envs, multiprocessing, or HPC job arrays

**Done when:** A written recommendation exists with a chosen approach.

---

## Phase 2 — Brand Momentum + Meta Inputs

### Issue #2: Brand momentum in utility function *(parallel with #3)*
**Goal:** Add EMA-based brand momentum to the passenger choice model.

- Implement update rule: `M_o(d) = λ · M_o(d-1) + (1-λ) · s_o(d)` with `M_o(0) = 0.5`
- **`s_o(d)` = `served_o / potential_demand`** (capture rate, not relative market share) — this correctly reflects passengers opting out of both operators, not just switching between them
- Add `M_o(d-1)` to the MNL utility: `U += γ · M_o(d-1)`
- Add `--brand_momentum_lambda` (default 0.9) and `--brand_momentum_gamma` (default 0.0) flags
- Key file: `src/envs/amod_env_multi.py` (`match_step_simple()`, lines 279–382)
- Extension (if time allows): investigate Rescorla-Wagner formulation from Sfeir et al. (see SPEC.md §2)

**Done when:** `gamma=0` produces identical demand splits to baseline; `gamma>0` produces measurable demand shifts; raising both operators' prices simultaneously causes both M_o values to decay.

### Issue #3: Meta-policy input definition *(parallel with #2)*
**Goal:** Implement the daily stats aggregator that feeds the meta-policy.

- Aggregate per-step info into a fixed-size 7-element daily summary vector:
  `[M_o, profit_o, avg_price_o, avg_price_opp, reb_cost_o, served_o, d/N]`
- **Excluded (not realistically observable):** opponent profit, opponent brand momentum, opponent fleet positions
- **Included from opponent:** avg price only — consistent with baseline paper (operators observe competitor prices)
- The combination of own capture rate `M_o` and opponent price lets the meta-policy distinguish "losing to competitor" from "losing to outside option" without seeing M_opp
- Key file: `src/misc/utils.py` (new helper function)

**Done when:** Aggregator runs after each simulated day and produces a correctly-shaped, normalised tensor.

### Issue #1: Meta-policy reward *(after #3)*
**Goal:** Define and implement the meta-level reward signal.

- Accumulate per-step pax and reb rewards into a daily total
- Primary: daily profit. If training is unstable, consider adding small market-share improvement bonus
- Key file: `main_hrl.py` (new) or the day-loop wrapper

**Done when:** Daily reward is logged and matches the sum of intra-day step rewards.

---

## Phase 3 — Meta-Policy

### Issue #4: Meta-policy architecture and action interface
**Goal:** Implement the PPO MLP meta-policy and wire it into the training loop.

- New file: `src/algos/meta_policy.py`
- Architecture: input (~9 scalars) → 2×128 ReLU hidden → actor head (per-region price multipliers, clamped [0.5, 2.0]) + critic head (scalar)
- Action interface: `p_effective = clamp(α_meta[i] · ρ_low[i], 0, 2) · p_ref`
- Algorithm: PPO (preferred over A2C for sparse daily rewards)
- Modelled on existing `src/algos/a2c_gnn_multi_agent.py` structure

**Done when:** Meta-policy forward pass runs without error and multipliers visibly modify prices in a test episode.

### Issue #9: Add WandB logging for meta-level metrics *(alongside #4)*
**Goal:** Log daily-frequency metrics alongside existing per-step metrics.

- Per-day: own/opponent profit, market share, brand momentum Q-values, meta-policy actor/critic loss
- Keep existing per-step logging unchanged
- Key file: `main_hrl.py`

**Done when:** A multi-day run produces a WandB run with both timestep-level and day-level metric panels.

---

## Phase 4 — Training Runs

### Issue #10: Training Phase 1 — single meta-policy vs fixed opponent
**Goal:** Validate the full HRL loop with one meta-policy learning against a frozen low-level opponent.

- Run `main_hrl.py` with one operator's meta-policy active; opponent uses frozen pre-trained low-level only
- Train for enough episodes to see convergence or a clear learning signal
- Inspect whether day-1 vs day-N pricing differs (early undercutting signal)

**Done when:** Meta-policy reward curve shows learning; day-by-day pricing trajectory is inspectable.

### Issue #11: Training Phase 2 — competitive meta-vs-meta
**Goal:** Run full competitive training with both meta-policies active.

- Both operators have meta-policies; train simultaneously
- Compare equilibrium to Phase 1 and single-day baseline

**Done when:** Both reward curves converge; equilibrium metrics are recorded for post-hoc comparison.

---

## Phase 5 — Analysis

### Issue #6: Design and run post-hoc analysis
**Goal:** Produce the empirical results that answer the core research questions.

- **Pricing trajectory:** Do operators undercut early and raise prices later?
- **Brand momentum dynamics:** How fast does momentum accumulate? Is there incumbency advantage?
- **Equilibrium comparison:** HRL vs single-day baseline (prices, profits, served demand, market share)
- **Sensitivity analysis:** Vary λ (decay speed), γ (momentum strength), N (days per episode)
- **Phase 1 vs Phase 2:** Does competitive meta-vs-meta differ from meta-vs-baseline?

**Done when:** Figures and summary statistics are generated for all key questions.

---

## Risks

| Risk | Mitigation |
|------|-----------|
| LP solver not parallelisable | Use HPC job arrays instead of shared-memory parallel envs |
| Meta-policy doesn't learn (sparse reward signal) | Add auxiliary market-share bonus; reduce N to 4 days initially |
| Training runs too slow | Front-load HPC runs; overlap Phase 4 with analysis writing |
| Rescorla-Wagner extension out of scope | EMA is the primary implementation; extension is optional if time allows |
