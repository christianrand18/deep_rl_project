# Sanity Checks
Mechanism-validation runs before further meta-RL experiments. All runs grouped in WandB as `sanity_checks`.

Scripts (HPC, 25k episodes each):
- b1: γ=1, no meta — brand momentum mechanism at moderate γ
- b2: γ=5, no meta — brand momentum mechanism at high γ
- d1: γ=1, heuristic schedule (α=0.5 days 0-2, α=1.5 days 3-6) — can momentum be exploited?
- d2: γ=5, heuristic schedule — exploitation at high γ

Phase A (multi-day wrapper reproducibility) and Phase C (action interface) are run locally — see top-level plan.

## Post-bugfix re-runs

After fixing the day-boundary terminal handling in the A2C return computation (CHANGELOG 2026-05-20), the 5 main configs are re-run to validate that per-day metrics now match the converged single-day baseline:

- bugfix_d7_g0_nometa: validation control — should match single-day baseline per-day
- bugfix_d7_g1_nometa: re-run of b1 with fix
- bugfix_d7_g5_nometa: re-run of b2 with fix
- bugfix_d7_g1_schedule: re-run of d1 with fix
- bugfix_d7_g5_schedule: re-run of d2 with fix

## Meta reward normalization re-run

After fixing the meta-policy reward normalization bug (2026-05-25, branch `fix/meta-reward-normalization`), the bounded-ρ meta run is re-run with the fix to get a clean read on whether compensation is the real bottleneck:

- normfix_d7_g5_meta_one: re-run of `sanity_bound_d7_g5_meta_one` with reward normalization fix applied
