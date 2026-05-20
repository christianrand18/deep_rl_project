# Sanity Checks
Mechanism-validation runs before further meta-RL experiments. All runs grouped in WandB as `sanity_checks`.

Scripts (HPC, 25k episodes each):
- b1: γ=1, no meta — brand momentum mechanism at moderate γ
- b2: γ=5, no meta — brand momentum mechanism at high γ
- d1: γ=1, heuristic schedule (α=0.5 days 0-2, α=1.5 days 3-6) — can momentum be exploited?
- d2: γ=5, heuristic schedule — exploitation at high γ

Phase A (multi-day wrapper reproducibility) and Phase C (action interface) are run locally — see top-level plan.
