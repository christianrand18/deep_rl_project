# Spec: Meta/Low-Level Compensation Fix

**Date:** 2026-05-25
**Branch:** `13-fix-meta-low-level-compensation` (current)
**Status:** Spec — three candidate approaches to be tested on HPC; winner merged to `main`

---

## Problem Statement

### Mental model

The meta-policy is the **CEO**: it sets the operator's daily pricing strategy based on brand momentum, market share trajectory, and multi-day episode dynamics. The low-level is the **floor manager**: it optimizes individual rides within the day, and should respect the CEO's strategic direction while retaining room for local, per-ride optimization.

Currently the CEO's instructions are ignored — the floor manager silently undoes them.

### Mechanism

The meta-policy outputs a daily price multiplier α that is applied as:

```
effective_price = clamp(α · ρ, 0, 2) · p̄
```

The low-level has no visibility into α and is trained on step-level revenue. When the meta sets α < 1 (undercut mode), effective prices drop and the low-level's step reward falls. The low-level responds by pushing ρ upward, partially restoring effective prices. Meta intent is diluted. This is architectural, not a bug.

When both levels train jointly, the low-level's reward gradient opposes the meta's strategy. The multiplier formulation is the root cause.

### Empirical confirmation

Two diagnostic runs (`sanity_normfix_d7_g5_meta_one` and `sanity_normfix_d7_g5_meta_one_unbound`) ran to 25,000 episodes on the HPC cluster after the meta-reward normalization fix (2026-05-25). Both confirm the compensation dynamic.

**Unbounded run (no ρ clamp) — Scenario A confirmed:**

| Metric | Agent 0 (meta) | Agent 1 (fixed) |
|--------|---------------|-----------------|
| meta_multiplier | 0.55–0.75 (persistent undercut) | — |
| price_scalar ρ | 1.77 | 1.0 (frozen) |
| effective price scalar | ~0.99 | ~0.99 |
| true profit | +14.3% above agent 1 | baseline |

Meta learned to undercut (Scenario A). But the low-level's compensation almost fully absorbs it: meta must drive α down to 0.59 just to achieve a near-zero net price difference relative to the opponent. Enormous meta capacity is wasted fighting the low-level.

**Bounded run (ρ ∈ [0.4, 0.6]) — neutral meta, no strategy:**

Meta converged to α ≈ 1.0 (flat, no strategic direction). The +8.6% profit gain came from low-level fleet efficiency, not meta pricing. When compensation range is artificially limited, meta stops trying.

**The compensation quantified:** In the unbounded run, α = 0.59 × ρ = 1.77 → effective = 1.04 ≈ opponent's 0.989. Meta pushed α to its lower bound and the low-level pushed ρ to its training maximum. The two signals cancel.

---

## Plan

The compensation dynamic is empirically confirmed (see above) — this is not a theoretical concern. The meta-reward normalization fix (branch `fix/meta-reward-normalization`) is a prerequisite and must be merged to `main` before these experiments run.

Test three structurally different fixes in this branch via HPC experiments. Compare against a `multiplier` baseline (current behavior). The approach that most cleanly resolves the compensation gradient *without destroying market performance* is merged to `main`.

Each approach is opt-in via a `--meta_action_mode` flag. Default remains `multiplier` so existing functionality is preserved.

### Confirmed setup (grilling session 2026-05-27)

The experiments run in **mode 2, origin-based pricing** (no `--od_price_actions`). The low-level price scalar is therefore `action_rl[a][:, 0]` — a per-origin Beta sample **ρ ∈ (0, 1)**.

Two units are in play and the original draft conflated them:
- **Price scalar ρ ∈ (0,1)** — the Beta head's raw output, what the low-level controls.
- **Price factor `2ρ ∈ [0,2]`** — the multiple of the baseline reference price the passenger sees (the env applies a built-in `2×`: `p = 2·p̄·price_scalar`).

In `multiplier` mode, **α ∈ [0,2] is a multiplier on ρ** (α=1 is neutral). In `soft`/`goal` mode, **α is reinterpreted as a target price *factor*** in `[0,2]` — see Approach B/C. `cap` keeps multiplier-space semantics (α=1 neutral). See `CONTEXT.md` for canonical definitions.

### Where the changes go

The meta action is composed into the low-level action in **`main_a2c_multi_agent.py`**, not the env — mode 1 at lines ~661-665, mode 2 (price column) at lines ~759-770. The env receives the already-composed `action_rl` and applies `p = 2·p̄·price_scalar` unchanged. So:
- The cap / drop-multiplier logic is a **main-loop** edit; `amod_env_multi.py` is untouched.
- Reward shaping (soft/goal) is applied in main at the reward-append sites (mode 2: line ~811, `model_agents[a].rewards.append(paxreward+rebreward)`), where per-step ρ is in scope. `training_step` only sees scalar rewards and cannot compute the shaping term.
- `set_meta_target(α)` on the agent exists **only** to feed the `lin_alpha` conditioning in the forward pass.

---

## Approach A — Hard Cap (`--meta_action_mode cap`)

Meta α acts as a one-sided ceiling/floor on the low-level's price scalar. The low-level keeps full action space below the cap but cannot violate the meta's price direction.

### Mechanism — ceiling-only

`cap` keeps multiplier-space semantics (α=1 = neutral) but enforces only a one-sided **undercut ceiling**. The premium branch is dropped: because the Beta head cannot emit ρ>1, a "floor" at α>1 degenerates into `max(ρ, α) = α`, a full constant override of the low-level (and lets cap reach 4× effective while soft/goal cap at 2×). We therefore make α≥1 a no-op.

```python
# α < 1 (undercut): meta caps the low-level scalar from above
# α ≥ 1: no constraint (low-level unconstrained)

if α < 1:
    effective_ρ = min(ρ, α)
else:
    effective_ρ = ρ

# composed in main; env then applies p = 2·p̄·effective_ρ
```

### Why this could work

- Mechanically enforces meta's undercut direction
- Removes the compensation incentive when the cap binds: pushing ρ above α has zero effect on revenue when α < 1, so the gradient flattens at the constraint
- Low-level retains optimization room *within* the constrained region

### Risks

- Discontinuous gradient at the boundary
- When α binds, the low-level receives zero gradient signal in the saturated direction — could stall learning
- Low-level has no observation of α (no `lin_alpha` in cap mode), so it cannot anticipate the cap
- `corr(α, raw_ρ)` can stay negative even when cap works (the low-level keeps trying to compensate; the cap just blocks it). Judge cap on **effective** ρ — see Decision criteria.

### Flag

```
--meta_action_mode cap
```

---

## Approach B — Soft Regularization (`--meta_action_mode soft`)

The CEO/employee model: meta sets strategic price direction; low-level follows but retains room for ride-to-ride optimization. Implemented as a penalty on deviation from meta's target.

### Mechanism

The α multiplication is removed from the price composition (the low-level's ρ passes through to the env unmodified). α is reinterpreted as a **target price factor in [0,2]** — the average `2ρ` the meta wants. A regularization penalty in **factor space** is added to the low-level reward:

```python
# no α in the price composition; env applies p = 2·p̄·ρ as in baseline

penalty = λ_reg · (2·mean(ρ_t) - α)²          # factor space: 2ρ ∈ [0,2] vs α ∈ [0,2]
r_lowlevel_raw = pax_reward + reb_reward - reward_scale · penalty
```

Where `mean(ρ_t)` is the mean price scalar across origins at step t (`mean(action_rl[a][:, 0])`).

**Factor space (not raw ρ).** ρ ∈ (0,1) but α ∈ [0,2]; comparing them directly is a units error and makes targets α>1 unreachable. `2ρ ∈ [0,2]` is the passenger-facing factor and is directly comparable to α across the full range. Consequence: effective price caps at 2× baseline (vs 4× in `multiplier`) — acceptable for an undercut-focused study.

**Post-scaling λ.** The append site stores **raw** rewards; `training_step` divides returns by `reward_scale=2000`. A raw `λ·(·)²` would be ~1000× too small. Pre-multiplying by `reward_scale` makes the penalty land at exactly `−λ·(·)²` *after* scaling, so λ is defined in the same O(0.001–1.5) units the policy gradient sees, and λ's meaning is independent of `reward_scalar`.

α is also injected into **both `GNNActor` and `GNNCritic`** via an additive conditioning layer:

```python
# After lin2, before lin3, in actor AND critic:
h = h + lin_alpha(α_broadcast)  # lin_alpha: Linear(1, hidden_size), zero-initialized
```

The critic is conditioned too because the shaped return now depends on α; a critic blind to α regresses a target that shifts with the daily α → biased baseline, noisier advantages. α reaches the layer via `set_meta_target(α)` (per day); when α is `None` (baseline, `cap`, `multiplier`, the fixed opponent) the add is skipped. Zero-init keeps old checkpoints loadable.

### Why this could work

- Compensation gradient is removed at the source: low-level's extrinsic reward no longer depends on α
- Soft penalty preserves gradient flow everywhere — no discontinuities
- Low-level can deviate from α when justified by local market conditions (the ride-to-ride optimization the user described)
- λ_reg controls the strength of the "CEO mandate"

### Risks

- λ_reg requires tuning
- If λ_reg too low: low-level ignores meta. If too high: low-level becomes a pure α-tracker and loses local optimization
- Quadratic penalty may dominate extrinsic reward gradients — monitor the logged shaped-term vs scaled-extrinsic ratio

### Flags

```
--meta_action_mode soft
--meta_reg_lambda 0.1     # weight on (ρ - α)² penalty
```

---

## Approach C — Goal-Setting with Intrinsic Reward (`--meta_action_mode goal`)

Meta sets a target price level β; low-level receives an intrinsic reward for tracking it. Distinct from soft regularization in that the signal is *positive* (reward for tracking) rather than *penalty* (cost for deviating), and uses a saturating shape rather than quadratic.

### Mechanism

α is reinterpreted as a goal β, a **target price factor in [0,2]** (same factor-space reinterpretation as B). The multiplier is removed from the price composition; the low-level's ρ passes through to the env unmodified.

Low-level reward gains an intrinsic alignment term, also in factor space and post-scaling:

```python
# no α in the price composition; env applies p = 2·p̄·ρ as in baseline

r_intrinsic = λ_align · max(0, 1 - |2·mean(ρ_t) - β|)   # factor space
r_lowlevel_raw = pax_reward + reb_reward + reward_scale · r_intrinsic
```

β is injected into **both `GNNActor` and `GNNCritic`** via the same zero-init additive conditioning layer as Approach B (the architecture change is shared; see B for the critic-conditioning rationale and the post-scaling treatment).

### Why this could work

- Compensation gradient removed (same as B)
- Positive shaping signal — provides gradient toward the goal even when extrinsic reward is noisy
- Saturating reward shape (caps at λ_align) prevents the intrinsic term from dominating
- Aligns with standard hierarchical RL practice (e.g. HIRO, FeUdal): meta sets goals, low-level is rewarded for achieving them

### Risks

- Intrinsic reward magnitude vs. extrinsic reward magnitude must be calibrated (post-scaling λ, as in B)
- Saturating shape gives no gradient when `|2·mean(ρ) - β| > 1`, so far-from-goal states are unguided
- More moving parts than B (an additional hyperparameter shape choice on top of λ)

### Flags

```
--meta_action_mode goal
--meta_align_lambda 0.1   # weight on intrinsic alignment reward
```

---

## Shared implementation

All three approaches share a small set of changes:

| File | Change |
|------|--------|
| `main_a2c_multi_agent.py` | Add `--meta_action_mode {multiplier,cap,soft,goal}` and the two λ flags. Dispatch in the action-composition block (cap → `min(ρ,α)` when α<1; soft/goal → drop the multiply) and at the reward-append sites (subtract/add `reward_scale·λ·(·)` using per-step `mean(action_rl[a][:,0])`). |
| `src/algos/layers.py` | Add zero-init `lin_alpha` additive conditioning to **both `GNNActor` and `GNNCritic`**, active only when a target is set (soft/goal). Zero-init preserves default behavior and checkpoint loadability. |
| `src/algos/a2c_gnn_multi_agent.py` | `set_meta_target(α)` stores the daily target and threads it into `actor(state, α)` / `critic(state, α)`; `α=None` skips conditioning. **No reward logic in `training_step`** — shaping lives in main where ρ is in scope. |
| `batch_jobs/` | Three new scripts for the three runs (see below). |

**`src/envs/amod_env_multi.py`: untouched.** The meta action is composed in main; the env only ever sees the final `action_rl`.

`MetaPolicy`: unchanged across all three. Action space stays `[0, 2]` scalar — reinterpreted downstream (multiplier vs. factor target) but never re-clamped.

### Backward compatibility

`--meta_action_mode multiplier` (default) preserves the current price formula and reward exactly. The γ=0, no-HRL baseline is unaffected. Old low-level checkpoints remain loadable: `lin_alpha` is a new layer with zero-init, and all other shapes are unchanged.

### Picard solver scope boundary

Picard solver (`--picard`) has its own action injection path and always operates in `multiplier` mode regardless of `--meta_action_mode`. Out of scope for this spec.

---

## Experiments

All runs: `num_days=7`, `brand_momentum_gamma=5`, `meta_policy=one` (agent 0 has meta, agent 1 does not), 25k episodes, `group=compensation_fix`. **All runs unbounded** — no `--low_level_scalar_min/max` (defaults 0.0/1.0), so compensation is free to act and `--meta_action_mode` is the only structural difference between runs.

**This is a screening experiment — pick one mechanism to merge, not a sensitivity study.** That framing sets the matrix: protect the *selection* from the failure modes that would corrupt a merge decision, nothing more.

**Round 1 (12 runs): seeds `{10,20,30}` × 4 configs at λ=0.1.** Single-seed RL is too noisy for the profit/tracking criteria — a single seed can make you merge a mechanism that only won by noise — so every config gets 3 seeds. γ stays fixed at 5 (varying it is a post-winner sensitivity analysis, not part of this decision).

| Config | Mode | λ | Seeds | Runs |
|--------|------|---|-------|------|
| `compfix_multiplier_s{seed}` | multiplier | — | 10,20,30 | 3 |
| `compfix_cap_s{seed}` | cap | — | 10,20,30 | 3 |
| `compfix_soft_l0.1_s{seed}` | soft | 0.1 | 10,20,30 | 3 |
| `compfix_goal_l0.1_s{seed}` | goal | 0.1 | 10,20,30 | 3 |

**λ sweep is a conditional gate, not an up-front arm.** If soft or goal looks weak/borderline at λ=0.1, run `λ ∈ {0.03, 0.3}` for that mechanism (×3 seeds, +6) *before eliminating it* — a bad λ must not unfairly kill a good mechanism. In the likely case where a winner tracks-and-preserves-profit at λ=0.1, the sweep is never spent.

**W&B layout:** all runs share `group=compensation_fix`; `job_type` = the `(mode, λ)` variant (e.g. `soft_l0.1`); seed/mode/λ are in `config` (via `config=args`). Grouping by `job_type` then auto-aggregates the 3 seeds into mean±std bands. The existing `sanity_normfix_d7_g5_meta_one_unbound` run (seed 10) can stand in for one multiplier run if a re-run is unnecessary.

Generate via LSF job arrays / a parameterized launcher rather than hand-written scripts (see plan Task 8).

---

## Decision criteria

The criteria are **not mode-symmetric**, so the primary gate is criteria 2 + 3; criterion 1 is a per-mode diagnostic only.

**Primary gate (must win both):**

2. **Effective price tracks meta intent.** `mean_effective_price_scalar` correlates positively with `meta_multiplier` / target. Currently flat near 1.0 regardless of α.
3. **Market performance is preserved.** `mean_true_profit` (extrinsic, already separate from the shaped training reward) is not significantly lower than the multiplier baseline — the fix must not gut revenue. Compare within the undercut regime, since B/C cap effective price at 2× while multiplier reaches 4×.

**Diagnostic only (criterion 1):**

1. `corr(α_day, ρ_day)`. Interpret per mode: for `cap`, compute on **effective** ρ (raw ρ can stay negatively correlated even when cap works); for `soft`/`goal`, a positive corr is near-automatic, so a *non-positive* corr there signals λ too weak, not approach failure; for `multiplier` the negative corr is the documented symptom.

Secondary signals:

- Meta strategy emerging by episode 3000–5000: `day/agent0_meta_multiplier` shows systematic intra-episode pattern; `day/agent0_market_share` exceeds opponent's.
- Stability: no training collapse, no reward explosion.

Tie-break preference: simpler implementation wins. In rough order of structural complexity, A < B < C.

---

## Resolved (grilling session 2026-05-27)

1. **`lin_alpha` shared between B and C?** Yes — same zero-init additive layer, different reward shaping downstream.
2. **`lin_alpha` for the critic?** Yes — actor **and** critic, soft/goal only. The shaped return depends on α, so a critic blind to α has a biased baseline. cap/multiplier inject nothing.
3. **Which price scalar in the shaping term?** Confirmed mode-2 origin pricing: `mean(action_rl[a][:, 0])`, used as `2·mean(·)` (factor space).
4. **Penalty/intrinsic magnitude vs. `reward_scalar=2000`.** λ is defined in **post-scaling** units; the term is appended pre-multiplied by `reward_scale` so it lands at `λ·(·)` after `training_step`'s division. λ=0.1 is meaningful; monitor logged shaped-term vs scaled-extrinsic ratio.
5. **α units in soft/goal.** Factor space: target compares against `2ρ ∈ [0,2]`, not raw ρ. Caps effective price at 2× baseline (vs 4× multiplier).
6. **Where the code goes.** All in `main_a2c_multi_agent.py`; env untouched; no reward logic in `training_step`.
7. **cap premium branch.** Ceiling-only — α≥1 is a no-op (the floor branch degenerates to a full override since ρ≤1).
8. **Run bounds.** All four runs unbounded (no `--low_level_scalar_min/max`).
