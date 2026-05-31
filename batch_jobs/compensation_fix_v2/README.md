# Compensation Fix v2 — Meta-Reward Augmentation

Round 3 of the compensation-fix screening: tests the v2 hypothesis that the meta-policy fails to converge because it has no cost signal for proposing targets the low-level won't deliver. v2 adds `--meta_track_lambda` (per-day penalty `λ_meta·(2·mean(ρ) − β)²` on the meta reward), closing the loop. See `Investigations/2026-05-29_goal-v2-spec.md`.

9 runs = `soft` mode × λ_track=0.3 (worker shaping) × λ_meta∈{0.1, 0.3, 1.0} (new) × seeds {10,20,30}. 100k episodes, 72h walltime, unbounded, `wandb_group=compensation_fix_v2`. `job_type` encodes `(soft_l0.3_lmX)` so seeds aggregate into bands.

Scripts:
- `softaug_lt0.3_lm0.1`: light meta augmentation
- `softaug_lt0.3_lm0.3`: medium
- `softaug_lt0.3_lm1.0`: strong

Submit: `for f in run_compfix_softaug_*.sh; do bsub < "$f"; done`

Skipped this round: `goal` mode (saturating shape is independent of the augmentation question; if soft+aug tracks, goal is redundant), and the upward λ_track sweep beyond 0.3 (one axis at a time).