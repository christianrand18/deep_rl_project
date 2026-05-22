# Sanity Checks for HRL + Brand Momentum

**Author:** Bertram Hage
**Date:** 2026-05-22

## What are we investigating?

We're extending the AMoD codebase with two changes: (1) a **brand momentum** term in the MNL utility (EMA of daily capture rate that carries across days within an episode), and (2) a **meta-policy** that outputs per-day price multipliers across a 7-day episode. Together these touch the env reset logic, the choice model, daily stats aggregation, the meta-policy network, the action interface (how meta and low-level multipliers combine), and the reward signal.

Initial meta-RL runs (`experiments_round_{1,2}`) showed surprising behavior: the meta-policy was *hurting* its own operator, all γ values converged to similar outcomes, and the predicted "undercut early, exploit late" strategy never emerged. Before tuning anything, we wanted to verify each underlying component works as designed in isolation.

**What we're looking for:** that (a) the multi-day wrapper reproduces single-day baseline behavior per-day at γ=0; (b) the brand momentum EMA matches its formula and shifts demand the way MNL theory predicts; (c) the action interface routes meta multipliers through to actual prices, including the new `[0, 4]×` range; (d) the meta and low-level reward signals agree; and (e) a deliberate strategy can exploit brand momentum if the mechanism is real.

## Experiments

| Phase | Where | Runs | Setting | Purpose |
|---|---|---|---|---|
| A | Local, 200 ep | `sanity_a{1,2,3}_d{1,3,7}_g0` | γ=0, no meta, varying `num_days` | Wrapper reproducibility + intra-episode drift check |
| B | HPC, ~21k ep | `sanity_b{1,2}_d7_g{1,5}_nometa` | 7-day, γ ∈ {1, 5}, no meta | Brand momentum mechanism in isolation |
| C | Local, 100–200 ep | `sanity_c{1,2,3}_alpha{1,05,2}_g0` | 7-day, γ=0, heuristic α ∈ {1.0, 0.5, 2.0} | Action interface end-to-end |
| D | HPC, ~21k ep | `sanity_d{1,2}_d7_g{1,5}_schedule` | 7-day, γ ∈ {1, 5}, heuristic α=0.5 → 1.5 | Can brand momentum be exploited? |
| Post-fix smoke | Local, 50 ep | `validate_singleday_unchanged` | `num_days=1`, no meta | Regression check on the bug fix |
| Post-fix HPC (in flight) | HPC, 25k ep × 5 | `sanity_bugfix_d7_g{0,1,5}_{nometa,schedule}` | Re-run of B/D with fix + γ=0 control | Validate fix recovers baseline-level per-day metrics |

## Results

**Phase A — wrapper looks correct.** Per-day metrics across A1/A2/A3 match within ~5%. EMA formula verified to floating-point precision. Days 0 through 6 within an episode show no systematic drift at γ=0. Symmetry M_0 ≈ M_1 holds.

**Phase B — brand momentum mechanism works.** Higher γ retains more passengers: rejection rate drops from 0.32 (γ=0) → 0.22 (γ=5), combined demand served rises monotonically with γ, and M_0/M_1 stay symmetric (within 0.002) because neither low-level has a reason to break symmetry. Behavior matches what the +γ·M utility shift predicts.

**Phase C — action interface routes correctly.** Heuristic α=1.0 produces results indistinguishable from no-meta (verifies the meta-apply code path). α=0.5 boosts agent0 served demand by +25% and grows M_0 above M_1. α=2.0 collapses agent0 demand and the effective scalar reaches 1.78 (≈2× the low-level's pre-meta output) — confirming the new `[0, 4]×` ceiling is reachable. Reward attribution: `day/meta_reward` matches `day/step_reward_sum` exactly. One incidental observation: when the meta pushes prices one way, the **low-level partially compensates by shifting its base scalar the other way** — not a bug, but worth flagging when reading meta multipliers in isolation.

**Phase D — brand momentum is exploitable.** With a scheduled heuristic (undercut days 0–2, exploit days 3–6) at γ=5, agent0 ends with +20% more profit than agent1. Day-by-day trajectory confirms the predicted pattern: M_0 climbs above M_1 during the undercut phase, peaks at day 2, then erodes during exploitation as agent0 prices up and loses share. The strategy "spends" the brand it built.

## Plot twist: a deeper issue surfaced from absolute numbers

Comparing Phase B/D against the **converged single-day baseline** (~17.5k combined profit/day) revealed that *all* multi-day runs — including the γ=0 control — plateaued at ~7–8k profit/day. The gap was not a brand momentum issue (γ=0 was equally degraded), but a multi-day **learning** issue.

Root cause: the low-level A2C return computation discounted rewards across the entire 140-step episode buffer without resetting at day boundaries. Since `env.reset_day()` wipes fleet positions, an action at the end of day 0 cannot causally affect anything in day 1 — yet was being credited for those rewards. This inflated return variance and biased credit assignment, degrading the low-level by ~60% in per-day profit.

**The fix:** one-line change in `training_step` to reset `R = 0` at each day boundary, treating each day's last step as terminal. The actor/critic networks, environment, observations, and update cadence are all unchanged. The single-day baseline path is provably bit-identical (modulo logic is a no-op at `num_days=1`); the local smoke test confirms. A secondary change ensures the meta reward equals `profit − rebalancing_cost` so the two levels optimize the same per-step signal. Five HPC re-runs are now in flight to lock in the post-fix numbers.

## What we can say now

- **All components work as designed in isolation** (A, C).
- **Brand momentum reduces opt-out and increases combined profit** at higher γ (B).
- **A deliberate undercut-then-exploit schedule gives the brand-building operator a structural advantage at high γ** (D). This is the first concrete evidence that the mechanism is exploitable, not just present.
- **The low-level partially compensates against meta interventions** (C, D) — affects how meta multipliers should be interpreted.

These conclusions are robust to the fix because they're either env-/demand-side properties (B) or relative comparisons between same-bug runs (D).

## What waits for the post-fix HPC validation

- Whether multi-day per-day metrics actually catch up to the converged single-day baseline (the validation).
- Quantitative magnitudes — the +20% schedule advantage at γ=5 may grow or shrink.
- Whether the low-level compensation effect persists or attenuates with a cleaner training signal.
- Anything about **meta-RL learning** — rounds 1+2 trained against a degraded low-level and are effectively reset.

## Low-level vs meta interference

The compensation effect is worth flagging on its own because it could make the two policies work against each other once we launch meta-RL.

**What we saw.** When the heuristic forced agent0's meta multiplier to a constant value, the low-level shifted its base scalar the opposite way: at α=0.5 the low-level pushed up to 1.10 (vs 0.95 at neutral); at α=2.0 it dropped to 0.89. The *effective* price moves less than the multiplier alone would suggest.

**Likely reason.** The low-level observes its own state and competitor prices, but not the meta multiplier ([src/algos/a2c_gnn_multi_agent.py](src/algos/a2c_gnn_multi_agent.py)). The multiplier is applied after the low-level samples. So if α is consistently low, the gradient simply says "higher ρ_low gives more reward," and the low-level shifts up. It isn't fighting the meta — it's chasing the only gradient it can see, with no way to condition on α.

**Why this matters.** When the meta tries to undercut to build brand, the low-level partially undoes it, so the effective price moves less than intended. This could attenuate the meta's gradient and dampen the strategic pricing variation we want the meta to learn. The post-fix runs will show how much of this survives a cleaner training signal.

## Bounded-ρ ablation (in flight)

Adding α to the low-level observation would actually make compensation *worse*, not better — a per-day-greedy low-level would learn to cancel α exactly. The structural check is to bound how much the low-level can deviate. Added `--low_level_scalar_min/max` and launched one HPC run at ρ ∈ [0.4, 0.6] (factor ±20% from neutral), γ=5, meta_policy=one, 25k episodes (`sanity_bound_d7_g5_meta_one.sh`). If the meta now produces visible strategic multipliers and a profit edge, compensation was meaningfully dampening the gradient. If not, bottleneck is elsewhere (likely the per-region meta action with only 7 transitions per PPO update).

## Key takeaways

- ✅ The new HRL components are individually correct: wrapper, EMA, action interface, reward attribution.
- ✅ Brand momentum behaves as theory predicts, and a sensible heuristic strategy can exploit it.
- ⚠️ A return-computation bug degraded *all* multi-day low-level training before 2026-05-22 by ~60% per-day profit. Quantitative numbers from `experiments_round_{1,2}` and `sanity_checks` Phases B/D are not trustworthy in absolute terms.
- 🔧 Fix is minimal and structurally motivated; single-day baseline is unchanged.
- 🧭 Five HPC re-runs (`sanity_bugfix_*`) are validating the fix now. Meta-RL re-launch is the next milestone.
