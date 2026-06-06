# msoft — multiplier_soft experiment

Tests whether keeping the multiplier composition AND adding reward shaping fixes the
compensation problem. The key difference from `soft`: α still multiplies ρ directly
(so the meta retains causal control), but the low-level is also penalised when the
effective price factor deviates from α: `-λ·(2·effective - α)²`.

Hypothesis: the meta's causal link is preserved, and the shaping discourages the
low-level from compensating — without the instability seen in pure `soft` at high λ.

## Runs

| Script | Mode | λ | Seed |
|--------|------|---|------|
| run_msoft_multiplier_s{10,20,30} | multiplier (baseline) | — | 10,20,30 |
| run_msoft_l0.1_s{10,20,30} | multiplier_soft | 0.1 | 10,20,30 |
| run_msoft_l0.3_s{10,20,30} | multiplier_soft | 0.3 | 10,20,30 |
| run_msoft_l1.0_s{10,20,30} | multiplier_soft | 1.0 | 10,20,30 |

12 runs total. Compare against compfix_multiplier and compfix_soft_l0.1 from W&B.
