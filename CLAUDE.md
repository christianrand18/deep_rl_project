# Deep RL Project

@coding_behavior.md

## Project Context

Extension of Toft et al. (2026) "Competitive Multi-Operator Reinforcement Learning for Joint Pricing and Fleet Rebalancing in AMoD Systems." The base codebase implements two competing A2C+GCN operators in an Autonomous Mobility-on-Demand environment. We are extending it with:

1. **Hierarchical RL**: a daily meta-policy (PPO MLP) that guides each operator's low-level pricing strategy across multi-day episodes
2. **Brand momentum**: an EMA of daily market share that enters the Multinomial Logit passenger utility function, persisting across days within an episode

See `SPEC.md` for the full architecture. See `litterature/` for the base paper and thesis.

## Key Files

- `main_a2c_multi_agent.py` — existing single-day training entry point (do not break)
- `main_hrl.py` — new HRL training entry point (to be created)
- `src/envs/amod_env_multi.py` — AMoD environment (multi-operator)
- `src/algos/a2c_gnn_multi_agent.py` — low-level A2C+GCN agent
- `src/algos/meta_policy.py` — meta-policy (to be created)

## Conventions

- All new HRL behavior is opt-in via flags; existing `main_a2c_multi_agent.py` behavior must remain unchanged
- `γ = 0` in the brand momentum utility term must exactly reproduce baseline choice model behavior
- Daily stats normalization: use the same `reward_scalar` as the low-level policy
- WandB logging: log both low-level (per-step) and meta-level (per-day) metrics separately
