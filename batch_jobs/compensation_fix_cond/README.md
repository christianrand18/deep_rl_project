# Compensation Fix — Conditioning-Only Ablation

Round 4 of the compensation-fix screening. Tests whether `lin_alpha` conditioning alone (without any reward shaping) is sufficient for the meta-policy to communicate strategic intent to the low-level.

## Hypothesis

Rounds 1–3 all added reward-shaping terms to the low-level (soft penalty, goal intrinsic) and/or the meta (tracking penalty). Round 3 showed that meta_track_lambda stabilises the low-level at ρ≈1 but locks both levels into a suboptimal equilibrium where the meta is penalised for any deviation from α=2.

The `lin_alpha` layer (zero-init additive in both GNNActor and GNNCritic) is already wired up in soft mode. With λ_reg=0 and λ_track=0, the meta target α is fed through lin_alpha to condition the actor/critic, but no shaping reward is added anywhere. The meta learns via pure profit, and the low-level can learn to respond to α signals through the value function gradient — no explicit gradient fighting.

## Config

All 3 runs: `soft` mode, λ_reg=0.0, λ_track=0.0, 100k episodes, 72h walltime, unbounded.

| Script | Seed |
|--------|------|
| `run_compfix_cond_s10.sh` | 10 |
| `run_compfix_cond_s20.sh` | 20 |
| `run_compfix_cond_s30.sh` | 30 |

WandB group: `compensation_fix_cond`

## Decision criteria

- **Pass**: agent0 profit ≥ nometa baseline (~80k) with stable brand momentum — meta is adding value, not destroying it
- **Partial**: agent0 profit between 27k (round 3) and 80k (nometa), but improving over training — useful signal even if not a winner yet
- **Fail**: agent0 profit ≈ 27k (same as round 3 with shaping) — lin_alpha conditioning has no effect; the compensation problem requires a different architectural approach

## Submit

```bash
for f in batch_jobs/compensation_fix_cond/run_compfix_cond_*.sh; do bsub < "$f"; done
```
