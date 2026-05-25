# picard_iterations

Tests whether Picard fixed-point iteration produces better learning than sequential multi-day rollouts, and how performance scales with brand momentum strength.

| Script | Description |
|---|---|
| `run_baseline_7day.sh` | Sequential 7-day, no Picard, bm_gamma=1 — baseline for direct comparison |
| `run_picard_7day_bmg1.sh` | Picard 7-day, bm_gamma=1 — moderate brand momentum coupling |
| `run_picard_7day_bmg5.sh` | Picard 7-day, bm_gamma=5 — strong brand momentum coupling |
| `run_picard_7day_bmg1_damped.sh` | Picard 7-day, bm_gamma=1, omega=0.7 — tests whether damped updates fix oscillation seen in bmg1 |
| `run_picard_7day_bmg1_anderson.sh` | Picard 7-day, bm_gamma=1, Anderson acceleration m=5, tol=1e-3 — converges in K=2-3 |

All runs: `nyc_man_south`, mode 2, `meta_policy one`. Picard runs use `tol=1e-3`, `max_iters=10`.
