# Picard Iterations — Final Report

**Claim:** Picard fixed-point iteration genuinely works, speeds up multi-day meta-policy
training, and produces a *better* deployed policy.

All runs: `nyc_man_south`, mode 2, `num_days 7`, `meta_policy one`, `brand_momentum_gamma 1.0`.
The claim is supported by **two experiments**; everything else was mechanism-hunting.

---

## 1. It works & it's fast — convergence in K≈1

`debug_picard_7day` (`--parallel_days --picard_update_strategy anderson --picard_tol 5e-3 --picard_max_iters 10`)

- `debug/picard_converged = 1`, `debug/picard_K_used = 1` for the overwhelming majority of episodes (rises to 2–4 only transiently with a tighter tol).
- Training is stable: meta/low-level losses well-behaved, profit climbs normally.

**Why this is the speed evidence:** the N=7 days run from one shared snapshot and the
fixed point closes in **K≈1 iteration**, so an episode costs ~K parallel passes instead of
N sequential day-rollouts → **~N/K ≈ 7× fewer sequential rollouts**, with the LP solves
across days embarrassingly parallel. Naive Jacobi would need N+1 iterations; the
analytic/Anderson update is what buys K≈1. Correctness is not traded for speed — see #2.

## 2. It learns a better policy — paired deployment eval *(the headline)*

`eval_picardtrained_7day` vs `eval_seqtrained_7day`. Both trained to the **same 20k-episode
budget** (one Picard, one plain sequential), then **frozen** and evaluated on the **same 100
plain stochastic days** — real per-day sampling, deterministic actions, natural brand
momentum, **no Picard, no denoising**, shared eval seeds (paired comparison).

| Metric (100 held-out stochastic days) | Picard-trained | Seq-trained | Δ |
|---|---|---|---|
| true profit | **138,006 ± 480** | 119,450 ± 666 | **+15.5%** |
| net return (meta objective) | **128,338 ± 624** | 109,329 ± 842 | **+17.4%** |

Eval std ≈ 0.5% of the mean, so the ~18.5k gap is **~20 sigma** — not eval noise.

**Why this is the load-bearing experiment:** it isolates exactly the thing in doubt. Both
policies face an *identical* stochastic environment they were never trained to game; the
only difference is how they were trained. Picard wins decisively. This rules out the central
worry — that Picard's clean training-time metrics were a denoising artifact that wouldn't
transfer. It transfers, and it's better.

---

## Why this subset is sufficient

- #1 establishes **works + fast** (valid, stable training that converges in K≈1).
- #2 establishes **better**, on the only metric that matters — honest stochastic deployment —
  under a controlled, paired protocol. A faster scheme that produced a worse policy would
  fail #2; a paper-only improvement would fail #2. It passes both.

Together they cover all three parts of the claim with no redundant runs.

## Caveats (what is *not* claimed)

- **n = 1 per arm** (seed 10). The eval comparison is paired and razor-tight, so the gap
  between *these two* policies is solid; "Picard reliably trains better policies" wants 2–3
  seeds. Cheap follow-up: re-run the eval pair on seeds 20/30.
- Both at 20k episodes — neither necessarily converged; the gap could shift with longer training.
- Eval measures profit; Picard trained at higher rejection (~0.40 vs ~0.32) — likely a
  "price higher, serve fewer, earn more" policy. If served-demand matters beyond profit, check it.

## Appendix — mechanism (open, but not needed for the claim)

*Why* Picard's training-time critic loss collapses (~0.007 vs ~0.67) is unresolved. We ruled
out, each with an experiment: per-day seeding, lagged obs, clamped buffer, warm-start, brand
momentum (gap persists at γ=0), update strategy (jacobi=anderson), and tolerance/K (1e-3 K=2–4
tracks 5e-3 K=1). The low critic loss is robust to all of them — something fundamental about
training on the fixed-point trajectory that single-lever ablations don't touch. **The practical
claim does not depend on resolving this**; #2 settles deployment quality directly.
