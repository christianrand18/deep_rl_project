# picard_iterations

Tests whether Picard fixed-point iteration produces better learning than sequential multi-day rollouts, and how performance scales with brand momentum strength.

| Script | Description |
|---|---|
| `run_baseline_7day.sh` | Sequential 7-day, no Picard, bm_gamma=1 — baseline for direct comparison |
| `run_picard_7day_bmg1.sh` | Picard 7-day, bm_gamma=1 — moderate brand momentum coupling |
| `run_picard_7day_bmg5.sh` | Picard 7-day, bm_gamma=5 — strong brand momentum coupling |
| `run_picard_7day_bmg1_damped.sh` | Picard 7-day, bm_gamma=1, omega=0.7 — tests whether damped updates fix oscillation seen in bmg1 |
| `run_picard_7day_bmg1_anderson.sh` | Picard 7-day, bm_gamma=1, Anderson acceleration m=5, tol=1e-3 — converges in K=2-3 |
| `run_picard_7day_bmg1_anderson_tol5e3.sh` | Same as above but tol=5e-3 — tests whether looser tolerance (above noise floor) gives K=1 more often |

All runs: `nyc_man_south`, mode 2, `meta_policy one`. Picard runs use `tol=1e-3`, `max_iters=10`.

## Debug ablations (`wandb_group: picard_debug`, 200k episodes each)

Isolate why Picard outperforms sequential. Three candidate mechanisms:
per-day RNG seeding, lagged obs, trajectory continuity via warm-start.

| Script | Variable isolated | Expected result if this is the cause |
|---|---|---|
| `debug_baseline_7day.sh` | — reference | — |
| `debug_picard_7day.sh` | — full Picard | — |
| `debug_seed_days_7day.sh` | per-day RNG seeding only | matches Picard if seeding drives the gap |
| `debug_lagged_7day.sh` | previous-episode obs only | matches Picard if obs lag drives the gap |
| `debug_picard_no_warmstart_7day.sh` | Picard minus warm-start | matches baseline if trajectory continuity drives the gap |

**Findings so far:** ruled out — seeding, lagged obs, warm-start, update strategy
(jacobi=anderson), clamped buffer, and brand momentum (gap persists at gamma=0).
The gap is intrinsic to the meta-policy transition pipeline. Last candidate: the
observation representation (Picard's smoothed `S_pred` vs sequential's live `daily_state`).

### zero-obs isolation (gamma=0, meta-policy blinded)

Feed the meta-policy a constant zero observation in both paths. With constant obs the
critic can only predict the mean, so `meta_critic_loss` = pure return variance.

| Script | Result interpretation |
|---|---|
| `debug_seq_zeroobs_g0_7day.sh` | sequential, blinded |
| `debug_picard_zeroobs_g0_7day.sh` | Picard, blinded |

Gap vanishes → cause was the obs representation. Gap persists → Picard's returns are
intrinsically lower-variance (look at action/reward generation next).
