# Investigation: Low-Level Compensation Fix — Results & Decision

**Date:** 2026-06-06
**Branch:** `13-fix-meta-low-level-compensation`
**Status:** Concluded. Soft mode (λ=0.1) merged to main. All other approaches rejected.

---

## Problem

In the `multiplier` meta-action mode, the meta-policy outputs a daily multiplier α applied as
`effective_price = clip(α·ρ, 0, 2)`. The low-level has no visibility into α and optimizes pure step revenue. When the meta sets α < 1 (undercut strategy), the low-level responds by pushing ρ upward — partially restoring effective prices. The two levels fight each other.

**Empirically confirmed in diagnostic runs (`sanity_normfix` series):**

| Metric | Value |
|--------|-------|
| Meta multiplier α | 0.55–0.75 (persistent undercut) |
| Low-level raw ρ | 1.77 |
| Effective price | ~1.0 (same as opponent) |
| Agent 0 true profit | +14% above opponent |

The meta must drive α to its lower bound and the low-level pushes ρ to its training maximum. Enormous capacity on both sides is wasted fighting each other.

---

## What Was Tried

Four structurally different fixes were tested via HPC experiments (200k episodes each, 3 seeds per config). All approaches are opt-in via `--meta_action_mode`; default (`multiplier`) is unchanged throughout.

### Benchmark

| Run | Profit (A0) |
|-----|-------------|
| compfix2_nometa (no meta-policy at all) | ~81k |
| compfix_multiplier (current main behavior) | 60–69k (mean ~63k) |

---

### Round 1 — cap / soft λ=0.1 / goal λ=0.1

| Mode | Profit | Raw ρ | Effective ρ | Meta α | Verdict |
|------|--------|-------|-------------|--------|---------|
| multiplier | 60–69k | 1.56–1.77 | 0.92–1.04 | 0.50–0.67 | Compensation confirmed |
| cap | 55–57k | ~1.0 | ~1.0 | 1.6–2.0 | Meta escapes to α≥1 where cap is no-op — structural dead end |
| soft λ=0.1 | 66–70k | ~1.0 | ~1.0 | 0.4–2.0 (scattered) | ρ stops compensating ✓; α uncontrolled |
| goal λ=0.1 | 66–78k | ~1.0 | ~1.0 | 0.0–1.6 (scattered) | Same picture as soft |

**cap** is a structural dead end: the ceiling-only design becomes a no-op whenever α≥1, which the meta learns to exploit immediately. Ruled out.

**soft and goal** stop the ρ compensation but α becomes meaningless — the two levels are decoupled rather than coordinated. Profit is modestly better than multiplier (mean ~68k vs ~63k).

---

### Round 2 — soft/goal at higher λ (0.3 and 1.0)

| Mode | Profit | Verdict |
|------|--------|---------|
| soft λ=0.3 | -26k to +15k | Partial training collapse |
| soft λ=1.0 | -5k to +16k | α saturates at 0 or 2; severe collapse |
| goal λ=0.3 | -40k to +31k | Extremely unstable across seeds |
| goal λ=1.0 | -40k to +5k | Full training collapse |
| oracle (α fixed=1.5) | 35–64k | Fixed target underperforms no-meta |

More shaping pressure makes things worse. The equilibrium breaks under stronger penalties.

---

### Round 3 — soft + meta-reward augmentation (meta_track_lambda, v2)

Closed the loop by penalizing the meta itself when α drifted from where ρ actually landed. Inspired by HIRO-style reachability reasoning (off-policy relabeling is incompatible with PPO, so this was the closest on-policy equivalent).

| Config | Profit | Verdict |
|--------|--------|---------|
| softaug lt=0.3, lm∈{0.1, 0.3, 1.0} (9 seeds total) | 26–28k | Converges to a consistently bad equilibrium |

The loop-closing penalty produced a different failure mode — not collapse, but a mediocre stable basin. Worse than the multiplier baseline.

---

### Round 4 — conditioning only, no shaping (still running at conclusion)

Zero-initialized `lin_alpha` conditioning on both actor and critic, with no reward modification. Tests whether the architecture alone, without any shaping pressure, lets the low-level learn to follow α.

| Run | Episodes | Profit | α |
|-----|----------|--------|---|
| compfix_cond_s10 | 205k | 69.2k | 2.0 (saturated) |
| compfix_cond_s20 | 224k | 72.8k | 1.96 (saturated) |
| compfix_cond_s30 | 223k | 70.4k | 0.33 |

Two of three seeds saturate α at the upper bound. Profit (69–73k) is better than softaug but still ~10k below no-meta. Not a different picture from round 1. No evidence of meaningful α tracking.

---

## Decision

**Merged: soft mode at λ=0.1.**

It is the only mode that:
1. Does not make things worse than the current `multiplier` baseline (~63k → ~68k mean)
2. Eliminates the ρ compensation dynamic
3. Is structurally stable across all 3 seeds at 200k episodes

The improvement is modest and the mechanism is impure (the meta α is scattered, not strategic). But it is a consistent improvement over what is currently on main, adds no instability, and is the only candidate that cleared both criteria.

**Not merged: cap, goal, softaug, cond.**

| Mode | Reason |
|------|--------|
| cap | Structural dead end — meta escapes to α≥1 |
| goal | Collapses at λ>0.1; not better than soft at λ=0.1 |
| softaug | Consistently underperforms even the multiplier baseline |
| cond | Promising direction but α still saturating at conclusion; needs more investigation |

---

## What Was Merged

- `--meta_action_mode soft` with `--meta_reg_lambda` (default 0.1)
- Zero-initialized `lin_alpha` conditioning in `GNNActor` and `GNNCritic` (no-op when `meta_target=None`, so old checkpoints load unchanged)
- `set_meta_target()` on `A2C` agent
- `day/agent{a}_avg_price_raw` W&B diagnostic metric
- `agent{a}/meta_shaping_term` W&B metric
- W&B `job_type` encoding mode+λ for seed aggregation
- CONTEXT.md terminology: price scalar vs price factor, meta multiplier vs meta target

---

## Open Questions

1. **Why does the meta α not track in soft mode?** The shaping term (λ=0.1) is too weak — the meta's per-day reward is dominated by profit (O(10–100) post-scale) and the penalty for misalignment barely registers. The meta learns to ignore α and the low-level learns to ignore the conditioning.

2. **Would conditioning-only (round 4) work with more episodes?** Possibly. At 200k episodes the architecture hasn't learned to use `lin_alpha`. Could be worth revisiting with a longer run or curriculum (start conditioning only, then introduce soft shaping later).

3. **Is the multiplier architecture fundamentally wrong?** The nometa baseline at 81k suggests the low-level alone already optimizes well. The meta may need a different role — multi-day fleet pre-positioning or brand momentum strategy — rather than a pricing multiplier that the low-level can fight.
