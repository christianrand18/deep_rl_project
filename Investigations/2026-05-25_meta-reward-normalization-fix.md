# Handoff: Meta Reward Normalization Fix

**Date:** 2026-05-25  
**Branch:** `fix/meta-reward-normalization` (pushed, not yet merged)  
**For:** HPC diagnostic re-run to confirm meta-policy can now learn

---

## What was wrong

In `main_a2c_multi_agent.py` line 882, the meta-policy reward was stored as raw profit:

```python
# BROKEN — raw profit (~3300/day)
meta_policies[_a].store_reward(accumulator.profit[_a] - accumulator.reb_cost[_a])
```

The observation vector built in `DailyStatsAccumulator.daily_state()` normalizes all monetary quantities by `reward_scalar=2000`. The W&B log at line 904 also normalizes. The reward buffer did not — a ~2000x mismatch.

**Consequence:** `meta_critic_loss` was ~3–200M (observed 3.3M in W&B run `sanity_bound_d7_g5_meta_one`). Because `MetaPolicy` uses a shared trunk with combined loss `actor_loss + 0.5 * critic_loss`, the trunk gradient was dominated entirely by critic loss, shaping all trunk features for value prediction. The actor head sat on top of a value-optimized trunk. The meta-policy was effectively untrained for the entire 25k-episode run.

**Smoke test results (local):**

| Metric | Broken | Fixed |
|--------|--------|-------|
| `meta_critic_loss` (fresh init) | 208,000,000 | 51 |
| `meta_advantage_mean` | 12,939 | 6.5 |
| Reduction factor | ~4,000,000× | — |

---

## The fix

**File:** `main_a2c_multi_agent.py`, line 880–882  
**Branch:** `fix/meta-reward-normalization`

```python
# FIXED — normalized to match observation scale
meta_policies[_a].store_reward(
    (accumulator.profit[_a] - accumulator.reb_cost[_a]) / args.reward_scalar
)
```

This is the only change on the branch. One line, no other behavior altered.

---

## What to do on HPC

### Step 1 — Pull the fix branch

```bash
cd ~/deep_rl_project
git fetch origin
git checkout fix/meta-reward-normalization
```

### Step 2 — Submit the diagnostic re-run

```bash
bsub < batch_jobs/sanity_checks/sanity_normfix_d7_g5_meta_one.sh
```

This re-runs the exact config from `sanity_bound_d7_g5_meta_one` (num_days=7, gamma=5, meta_policy=one, bounded-ρ [0.4,0.6], 25k episodes) with the fix applied. W&B run name: `sanity_normfix_d7_g5_meta_one`, group: `sanity_checks`.

### Step 3 — What to watch in W&B

**Meta-policy learning (should now work):**
- `agent0/meta_critic_loss` — target: drops below 100 within a few hundred episodes, settles O(1–10)
- `agent0/meta_advantage_mean` — should be ~1–10, not ~1700
- `agent0/meta_actor_loss` — should be non-trivial (O(0.1–1.0))

**Compensation dynamics (the actual question):**
- `day/agent0_meta_multiplier` — does it still drift >1.0 (compensating for bounded low-level), or does the meta now learn to undercut (stay <1.0)?
- `agent0/mean_price_scalar` (low-level) — still at 0.4 floor? or does compensation attenuate?
- `agent0/mean_effective_price_scalar` vs `agent1/mean_effective_price_scalar` — does agent0 achieve lower effective price?
- `day/agent0_market_share` vs `day/agent1_market_share` — does agent0 (meta-RL) start winning share?

**Profit:**
- `agent0/true_profit` vs `agent1/true_profit` — the gap from `sanity_bound_d7_g5_meta_one` was 2.2× in agent1's favour; this should narrow or reverse if meta learns to undercut

---

## Interpreting the results

### Scenario A — Meta converges, compensation attenuates
`meta_critic_loss` drops to O(1–10), `day/agent0_meta_multiplier` stays ≤1.0, agent0 gains market share. The normalization fix was sufficient — the meta can learn and discovers undercutting. Bounded-ρ may not be necessary as a permanent fixture.

**Next:** Remove bounded-ρ and test with free low-level scalars. Check whether the undercut-exploit strategy (α<1 early days, α>1 late) emerges without the clamping.

### Scenario B — Meta converges but compensation still dominates
`meta_critic_loss` drops cleanly, but `day/agent0_meta_multiplier` still drifts >1.0 while `agent0/mean_price_scalar` stays at 0.4 floor. Clean evidence that the credit assignment gap between meta and low-level is the real bottleneck, not the reward normalization.

**Next:** Proceed to goal-conditioning architecture — meta outputs target price differential β ∈ [-0.5, +0.5], β added to low-level obs, intrinsic reward for staying near β. See background section below.

### Scenario C — Meta still doesn't converge
`meta_critic_loss` stays high (>1000 after 5k episodes). Investigate further — possible issues: obs/reward scale still mismatched, learning rate too high/low, gradient clipping too tight.

---

## Background: goal-conditioning architecture (if Scenario B)

The core problem goal-conditioning solves: the low-level policy has ~20 steps per day to implicitly undo whatever price level the meta chose. Even with bounded-ρ reducing the amplitude, the meta has no way to enforce its intent.

Proposed fix:
- Meta outputs β ∈ [-0.5, +0.5]: target price relative to competitor (negative = undercut)
- β appended to low-level obs (competitor prices already in obs, so this is natural)
- Low-level gets a small intrinsic reward `λ * (1 - |actual_diff - β|)` per step
- Meta gradient: "I set β, I observe the day-level outcome, here's the reward"

β is directly observable at each low-level step and anchored to a quantity already in obs. This is the key reason it beats alternatives like goal-conditioned value functions or auxiliary losses on α directly.

Do NOT design or implement goal-conditioning until Scenario B is confirmed with clean data from this run.

---

## Files changed

- `main_a2c_multi_agent.py` — one-line fix at line 880 (branch: `fix/meta-reward-normalization`)
- `batch_jobs/sanity_checks/sanity_normfix_d7_g5_meta_one.sh` — HPC script for diagnostic re-run

## References

- Previous broken run: W&B `sanity_bound_d7_g5_meta_one` (group: `sanity_checks`)
- Investigation doc: `Investigations/2026-05-22_multi-day-return-bugfix.md`
- Meta-policy code: `src/algos/meta_policy.py`
- Obs construction: `src/misc/utils.py` → `DailyStatsAccumulator.daily_state()`
