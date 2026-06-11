# final_experiments_bminit — brand momentum init re-run, γ=5, 3 seeds

Follow-up to `final_experiments`: fixes brand momentum strength at
`brand_momentum_gamma=5` and changes the initial brand momentum
(`brand_momentum_init`) from the default `0.5` to `0.175`. Repeats the same
4 meta-policy configs (`mult`, `msoft_l0.3`, `msoft_l1`, `nometa`) at both
episode lengths (`num_days` 7 and 28), across seeds 10, 20, 30.

## Config (all runs)

| Flag | Value |
|------|-------|
| City | nyc_man_south |
| Mode | 2 (joint pricing + rebalancing) |
| Brand momentum γ | 5 |
| Brand momentum init | 0.175 |
| Episodes | 100k |
| Solver | Picard, parallel days (`picard_parallel_workers = num_days`) |

## Runs

| Config | Action mode | λ | Days | Workers | CPUs |
|--------|-------------|---|------|---------|------|
| mult | multiplier | – | 7 / 28 | 7 / 28 | 8 / 32 |
| msoft_l0.3 | multiplier_soft | 0.3 | 7 / 28 | 7 / 28 | 8 / 32 |
| msoft_l1 | multiplier_soft | 1 | 7 / 28 | 7 / 28 | 8 / 32 |
| nometa | none | – | 7 / 28 | 7 / 28 | 8 / 32 |

Each of the 8 (config × days) combinations is run for seeds 10, 20, 30 —
24 scripts total, named `run_bmg5_<config>_d<days>_s<seed>.sh`.

`mult` runs omit `--meta_action_mode`/`--meta_reg_lambda`. `nometa` runs use
`--meta_policy none` and omit `--meta_agent`. Picard strategy/tolerance/
Anderson params are left at their defaults (anderson, m=5, tol=5e-3,
max_iters=10).
