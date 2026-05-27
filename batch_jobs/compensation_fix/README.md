# Compensation Fix

Screening experiment to pick one `--meta_action_mode` that stops the low-level policy from compensating against the meta-policy's price intent. Merge the winner to `main`. See `Investigations/2026-05-25_compensation-fix-spec.md`.

12 runs = 4 configs × 3 seeds (10/20/30), all `num_days=7`, `brand_momentum_gamma=5`, `meta_policy=one`, 25k episodes, **unbounded** (no `--low_level_scalar_*`):
- `multiplier`: current behavior (baseline / reference for compensation magnitude)
- `cap`: ceiling-only constraint `min(ρ,α)` when α<1
- `soft_l0.1`: drop multiplier, penalize `(2·mean(ρ)−α)²`, λ=0.1
- `goal_l0.1`: drop multiplier, intrinsic reward for tracking target factor, λ=0.1

All grouped in WandB as `compensation_fix`; `job_type` encodes `(mode, λ)` so the 3 seeds aggregate into bands. λ sweep `{0.03, 0.3}` is a conditional follow-up — run only if soft/goal looks weak/borderline before eliminating it.

Submit: `for f in run_compfix_*.sh; do bsub < "$f"; done`
