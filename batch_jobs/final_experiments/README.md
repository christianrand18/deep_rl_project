# final_experiments — meta-policy action mode × brand momentum × episode length

Final HRL sweep: single meta-policy (agent 0) vs. competing low-level agent 1,
run via the Picard parallel-days solver. Crosses brand momentum strength
(`brand_momentum_gamma`), meta action composition (`multiplier` vs
`multiplier_soft`, with regularisation `meta_reg_lambda` for the latter), and
episode length (`num_days`).

## Config (all runs)

| Flag | Value |
|------|-------|
| City | nyc_man_south |
| Mode | 2 (joint pricing + rebalancing) |
| Seed | 10 |
| Meta-policy | one (agent 0) |
| Episodes | 100k |
| Solver | Picard, parallel days (`picard_parallel_workers = num_days`) |

## Runs

| Script | bm γ | Action mode | λ | Days | Workers | CPUs |
|--------|------|-------------|---|------|---------|------|
| run_bmg1_mult_d7_s10 | 1 | multiplier | – | 7 | 7 | 8 |
| run_bmg1_mult_d28_s10 | 1 | multiplier | – | 28 | 28 | 32 |
| run_bmg1_msoft_l0.3_d7_s10 | 1 | multiplier_soft | 0.3 | 7 | 7 | 8 |
| run_bmg1_msoft_l1_d7_s10 | 1 | multiplier_soft | 1 | 7 | 7 | 8 |
| run_bmg1_msoft_l0.3_d28_s10 | 1 | multiplier_soft | 0.3 | 28 | 28 | 32 |
| run_bmg1_msoft_l1_d28_s10 | 1 | multiplier_soft | 1 | 28 | 28 | 32 |
| run_bmg5_mult_d7_s10 | 5 | multiplier | – | 7 | 7 | 8 |
| run_bmg5_mult_d28_s10 | 5 | multiplier | – | 28 | 28 | 32 |
| run_bmg5_msoft_l0.3_d7_s10 | 5 | multiplier_soft | 0.3 | 7 | 7 | 8 |
| run_bmg5_msoft_l1_d7_s10 | 5 | multiplier_soft | 1 | 7 | 7 | 8 |
| run_bmg5_msoft_l0.3_d28_s10 | 5 | multiplier_soft | 0.3 | 28 | 28 | 32 |
| run_bmg5_msoft_l1_d28_s10 | 5 | multiplier_soft | 1 | 28 | 28 | 32 |

`multiplier` runs omit `--meta_action_mode`/`--meta_reg_lambda` (defaults).
Picard strategy/tolerance/Anderson params are left at their defaults
(anderson, m=5, tol=5e-3, max_iters=10). Total CPU request across the group: 240.
