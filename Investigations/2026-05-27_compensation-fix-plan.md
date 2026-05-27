# Implementation Plan: Meta/Low-Level Compensation Fix

Companion to `2026-05-25_compensation-fix-spec.md` (grilling-refined 2026-05-27). Implements the four `--meta_action_mode` values: `multiplier` (default, unchanged), `cap`, `soft`, `goal`.

## Overview

Add three structurally different ways for the meta-policy to express price intent without the low-level silently compensating. All changes are opt-in behind `--meta_action_mode`; `multiplier` stays byte-identical to today. Work is sliced so each mode becomes usable end-to-end as early as its dependencies allow, cheapest-first (cap → soft → goal).

## Architecture Decisions (from the grilling session)

- **All logic in `main_a2c_multi_agent.py`, `src/algos/layers.py`, `src/algos/a2c_gnn_multi_agent.py`.** `src/envs/amod_env_multi.py` is untouched — the meta action is composed in main before the env sees it.
- **Factor space.** soft/goal compare `2·mean(ρ)` (the passenger-facing factor, ∈[0,2]) against α∈[0,2], not raw ρ∈(0,1).
- **Post-scaling λ.** Shaping terms are appended pre-multiplied by `reward_scale` so they land at `λ·(·)` after `training_step`'s `/reward_scale`.
- **Conditioning.** Zero-init `lin_alpha` additive layer in **both** `GNNActor` and `GNNCritic`, active only when a target is set (soft/goal). `α=None` skips → old checkpoints load.
- **cap is ceiling-only.** `α<1 → min(ρ,α)`; `α≥1 → no-op`.
- **Scope: mode 2, origin pricing only** (the experiment path). New modes guard against mode 0/1, `od_price_actions`, and `num_days==1`. (See Open Questions.)

## Task List

### Phase 1: Foundation

#### Task 1: Flags + validation guards + W&B job_type
**Description:** Add `--meta_action_mode {multiplier,cap,soft,goal}` (default `multiplier`), `--meta_reg_lambda` (default 0.1), `--meta_align_lambda` (default 0.1) to `src/arguments.py`. Add a startup validation guard: any mode ≠ `multiplier` requires `mode==2`, not `od_price_actions`, `num_days>1`, and a meta policy active; raise a clear error otherwise. Set `wandb.init(job_type=...)` to the `(mode, λ)` variant string (e.g. `soft_l0.1`, `cap`, `multiplier`) so runs nest as group → job_type → seed and W&B aggregates seeds into bands with one group-by. No behavior change for `multiplier` (job_type is metadata only).

**Acceptance criteria:**
- [ ] Three flags parse; defaults preserve current behavior.
- [ ] Unsupported combinations raise a descriptive error at startup, not mid-run.
- [ ] `wandb.init` receives a `job_type` reflecting mode + λ; seed/mode/λ remain in `config` (already true via `config=args`).

**Verification:**
- [ ] `WANDB_MODE=disabled python -m pytest tests/test_regression.py` passes (multiplier path unchanged).
- [ ] Manual: `--meta_action_mode cap --mode 1` exits with the guard message.

**Dependencies:** None
**Files:** `src/arguments.py`, `main_a2c_multi_agent.py` (guard near setup)
**Scope:** S

### Phase 2: cap (no NN/reward change — fail-fast slice)

#### Task 2: cap composition in main
**Description:** In the mode-2 origin composition block ([main_a2c_multi_agent.py#L759-L770](../main_a2c_multi_agent.py#L759-L770)), branch on `meta_action_mode`: `cap` replaces the `α·ρ` multiply with `effective = min(ρ, α) if α<1 else ρ` on the price column. `multiplier` path unchanged.

**Acceptance criteria:**
- [ ] `cap` + heuristic `const_05`: effective price scalar ≤ 0.5 everywhere.
- [ ] `cap` + heuristic `const_2`: effective == raw low-level output (no constraint).

**Verification:**
- [ ] New smoke test `test_multi_agent_mode2_cap` (mode 2, num_days=2, meta_policy=heuristic, max_episodes=2, max_steps=5) returns 0.
- [ ] Manual: inspect `mean_effective_price_scalar` honors the cap.

**Dependencies:** Task 1
**Files:** `main_a2c_multi_agent.py`, `tests/test_regression.py`
**Scope:** S

#### Checkpoint: cap
- [ ] Regression suite green; cap mode runs end-to-end and the cap binds as expected.

### Phase 3: conditioning infrastructure (shared by soft + goal)

#### Task 3: `lin_alpha` conditioning in layers
**Description:** Add a zero-init `nn.Linear(1, hidden_size)` to both `GNNActor` and `GNNCritic`. After the `lin2` activation and before `lin3`, add `lin_alpha(target_broadcast)` when a target is provided; `target=None` skips the add entirely. `forward` gains an optional `meta_target` argument.

**Acceptance criteria:**
- [ ] With zero-init, `forward(data, meta_target=1.0)` output equals `forward(data, meta_target=None)` bit-for-bit.
- [ ] An existing low-level checkpoint loads without shape errors (new layer is fresh).

**Verification:**
- [ ] Unit test: construct `GNNActor`/`GNNCritic`, assert no-op equality at init.
- [ ] Unit test: `load_checkpoint` of a saved baseline ckpt succeeds.

**Dependencies:** Task 1
**Files:** `src/algos/layers.py`, `tests/` (new unit test file)
**Scope:** S

#### Task 4: `set_meta_target` threading + per-day wiring
**Description:** Add `A2C.set_meta_target(α)` storing the daily target; `select_action` threads it into `actor(state, meta_target)` and `critic(state, meta_target)`. In main, after the meta selects α (near [main_a2c_multi_agent.py#L524](../main_a2c_multi_agent.py#L524)), call `set_meta_target` for soft/goal agents only; otherwise the target stays `None`.

**Acceptance criteria:**
- [ ] soft/goal: agent forward runs with the target set; at zero-init the first-step training reward equals a non-conditioned run.
- [ ] cap/multiplier/fixed-opponent: target is `None`, no conditioning.

**Verification:**
- [ ] Smoke test: mode 2 + soft + heuristic, num_days=2, returns 0.
- [ ] Manual: assert `_meta_target is None` for the fixed agent.

**Dependencies:** Task 3
**Files:** `src/algos/a2c_gnn_multi_agent.py`, `main_a2c_multi_agent.py`
**Scope:** S

#### Checkpoint: conditioning infra
- [ ] Infra present and a verified no-op at zero-init; old checkpoints still load.

### Phase 4: soft

#### Task 5: soft mode — drop-multiply + factor-space penalty
**Description:** Add the `soft`/`goal` shared composition branch: drop the multiply so the env sees raw ρ. At the mode-2 reward-append site ([main_a2c_multi_agent.py#L811](../main_a2c_multi_agent.py#L811)) subtract `reward_scale · meta_reg_lambda · (2·mean(action_rl[a][:,0]) − α)²`. Log the per-step shaped term and the scaled-extrinsic reward for ratio monitoring.

**Acceptance criteria:**
- [ ] Env-submitted price for a soft agent has no multiplier applied (effective == 2ρ).
- [ ] Logged mean penalty is O(λ) post-scaling, comparable to scaled extrinsic; `episode_true_profit` is unaffected by shaping.

**Verification:**
- [ ] Smoke test `test_multi_agent_mode2_soft` returns 0.
- [ ] Manual: W&B (or stdout) shows `penalty_term` and `extrinsic_scaled` in the same order of magnitude.

**Dependencies:** Task 4
**Files:** `main_a2c_multi_agent.py`
**Scope:** M

#### Checkpoint: soft
- [ ] soft runs end-to-end; shaping magnitude sane; profit metric clean.

### Phase 5: goal

#### Task 6: goal mode — intrinsic alignment term
**Description:** Reuse the drop-multiply branch from Task 5. At the reward-append site add `reward_scale · meta_align_lambda · max(0, 1 − |2·mean(action_rl[a][:,0]) − β|)`. Log the intrinsic term.

**Acceptance criteria:**
- [ ] Intrinsic term ≥ 0 and bounded by `meta_align_lambda` post-scaling.
- [ ] Composition identical to soft (raw ρ to env).

**Verification:**
- [ ] Smoke test `test_multi_agent_mode2_goal` returns 0.
- [ ] Manual: intrinsic term peaks when `2·mean(ρ) ≈ β`.

**Dependencies:** Task 5
**Files:** `main_a2c_multi_agent.py`
**Scope:** S

### Phase 6: diagnostics + experiments

#### Task 7: per-day decision-criteria logging
**Description:** Ensure per-day logs carry what the decision criteria need: meta target (already at [main_a2c_multi_agent.py#L937](../main_a2c_multi_agent.py#L937)), per-day mean **raw** ρ and mean **effective** ρ (for `corr(α, ·)`), and the existing `mean_effective_price_scalar`. Add the two ρ aggregates if absent.

**Acceptance criteria:**
- [ ] A smoke run's day_log contains target, mean raw ρ, mean effective ρ.

**Verification:**
- [ ] Manual: inspect logged keys in a 2-day smoke run.

**Dependencies:** Task 6
**Files:** `main_a2c_multi_agent.py`
**Scope:** S

#### Task 8: batch scripts — 12 runs (3 seeds × 4 configs); λ sweep conditional
**Description:** Generate the screening matrix via the `batch-jobs` skill conventions, preferring LSF **job arrays / a parameterized launcher** over hand-written files. Common flags: `num_days=7`, `brand_momentum_gamma=5`, `meta_policy=one`, 25k episodes, `wandb_group=compensation_fix`, **unbounded** (no `--low_level_scalar_*`). Round-1 matrix (seeds `{10,20,30}`, λ=0.1 for soft/goal):

| Config | Seeds | λ | Runs |
|--------|-------|---|------|
| `multiplier` | 3 | — | 3 |
| `cap` | 3 | — | 3 |
| `soft` | 3 | 0.1 | 3 |
| `goal` | 3 | 0.1 | 3 |

The purpose is to **screen mechanisms and merge one winner**, so this is a selection matrix, not a sensitivity study. 3 seeds protect the selection from RL noise (the one failure mode that corrupts a merge decision). The λ sweep `{0.03, 0.3}` for soft/goal is a **conditional follow-up**, run only if λ=0.1 looks weak or borderline for a mechanism *before eliminating it* (so a bad λ doesn't unfairly kill a good mechanism) — +6 runs if triggered. `sanity_normfix_d7_g5_meta_one_unbound` (seed 10) can stand in for one multiplier run if no re-run is needed. Run naming: `compfix_{mode}[_l{λ}]_s{seed}`.

**Acceptance criteria:**
- [ ] All flags exist in `arguments.py`; seed varies correctly across the array.
- [ ] No `--low_level_scalar_min/max` present; `wandb_group=compensation_fix` everywhere; `job_type` set per (mode, λ).

**Verification:**
- [ ] Dry parse: array expansion produces the 12 intended (mode, λ, seed) tuples.

**Dependencies:** Tasks 2, 5, 6
**Files:** `batch_jobs/...` (array script(s) / launcher)
**Scope:** S

#### Checkpoint: complete
- [ ] All four modes run; smoke tests green; batch scripts ready for HPC submission.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Editing the shared composition block breaks `multiplier` | High | Task 1 keeps the multiplier path unchanged; regression suite gates every task |
| Conditioning layer breaks checkpoint loading | High | Zero-init + explicit load test in Task 3 |
| λ=0.1 mis-calibrated (soft/goal no-op or dominating) | Med | Post-scaling fix (Task 5) + magnitude logging (Task 7); 2nd λ round pre-budgeted in spec |
| cap's flat gradient stalls low-level learning | Med | Expected, not a bug; judged on effective ρ + profit per the spec's decision criteria |

## Open Questions

1. **Mode coverage.** ✅ Resolved (2026-05-27): mode-2 origin only; guard mode 0/1, `od_price_actions`, `num_days==1` with a startup error (Task 1).
2. **Test depth.** Add the per-mode smoke tests to `tests/test_regression.py` (they need CPLEX for mode 2), or rely on manual smoke + HPC? *Recommend adding them* — cheap regression insurance, skip-marked if CPLEX absent.
3. **Baseline run.** Reuse the existing `_unbound` multiplier run as the reference, or re-run under the post-change code? *Recommend reuse* — the multiplier path is unchanged by design (verify via Task 1's regression check).
