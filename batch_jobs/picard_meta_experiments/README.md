# picard_meta_experiments — Picard + soft / multiplier_soft

Tests the meta-policy action modes combined with Picard Anderson fixed-point
iteration (parallel days, 7 workers). Compare `soft` vs `multiplier_soft` at λ=0.1.

## Config

| Flag | Value |
|------|-------|
| Picard strategy | anderson |
| Anderson window | 5 |
| Tol | 5e-3 |
| Max iter | 10 |
| Workers | 7 (days) |
| CPUs | 8 (workers + 1) |
| Days | 7 |
| Episodes | 200k |

## Runs

| Script | Mode | λ | Days | Workers | CPUs | Seed |
|--------|------|---|------|---------|------|------|
| run_picard_soft_l0.1_s10 | soft | 0.1 | 7 | 7 | 8 | 10 |
| run_picard_msoft_l0.1_s10 | multiplier_soft | 0.1 | 7 | 7 | 8 | 10 |
| run_picard_soft_d28_l0.1_s10 | soft | 0.1 | 28 | 28 | 32 | 10 |
| run_picard_msoft_d28_l0.1_s10 | multiplier_soft | 0.1 | 28 | 28 | 32 | 10 |
