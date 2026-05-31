# Compensation Fix — Round 2

Follow-up to `compensation_fix` (round 1). Round 1 found: at λ=0.1, soft/goal don't track meta intent (low-level ignores the target); `cap`'s ceiling-only design lets the meta escape to α≥1 (no-op zone). Profit "wins" for soft/goal were the no-undercut equilibrium (a1 also profited), not real meta success.

Round 2 resolves three questions in one batch:

- **Lower bound (2 seeds):** what does the system look like with no meta intervention? If soft/goal at λ=0.1 ≈ this, the round-1 profit lift was just "no price war."
- **Upper bound (2 seeds):** the spec's hypothesized winning strategy — undercut early to build brand momentum, premium late to exploit it (`schedule_undercut_exploit`). Sets the ceiling a learned meta could aspire to.
- **λ sweep (3 seeds each):** soft and goal at λ ∈ {0.3, 1.0} — does scaling λ up close the tracking gap, or is the failure structural?

16 runs total. Common flags: `--mode 2 --num_days 7 --brand_momentum_gamma 5`, **100k episodes**, unbounded. Walltime: **72h**.

| Config | Mode | Meta | λ | Seeds | Runs |
|--------|------|------|---|-------|------|
| `compfix2_nometa_s{seed}` | multiplier | heuristic `const_1` | — | 10, 20 | 2 |
| `compfix2_oracle_s{seed}` | multiplier | heuristic `schedule_undercut_exploit` | — | 10, 20 | 2 |
| `compfix2_soft_l{λ}_s{seed}` | soft | learned | 0.3, 1.0 | 10, 20, 30 | 6 |
| `compfix2_goal_l{λ}_s{seed}` | goal | learned | 0.3, 1.0 | 10, 20, 30 | 6 |

All grouped in WandB as `compensation_fix2`; `job_type` encodes `(mode, λ)` so seeds aggregate into bands. `cap` is dropped — its α≥1 escape hatch is structural, not a tuning issue.

Decision logic after this round:
1. **soft/goal at higher λ tracks meta intent and lands between lower and upper bound on profit** → pick the closer-to-upper-bound mechanism, merge, done.
2. **Tracking improves but profit collapses below the lower bound** → λ overshot; binary-search.
3. **Even at λ=1.0, meta still drifts to β=2 / low-level ignores it** → failure is structural; time to redesign (meta-tracking-error in meta reward + smooth quadratic + hindsight relabeling).

Submit: `for f in batch_jobs/compensation_fix2/run_compfix2_*.sh; do bsub < "$f"; done`