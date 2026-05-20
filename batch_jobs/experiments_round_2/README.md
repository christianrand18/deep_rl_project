# Experiments Round 2
Second round of experiments after action-range expansion (combined scalar 0-4) and meta-policy std init bumped to 0.5 (git hash: 30504cc0638d526d12d5ef462aa5d78efef3d14a).

Scripts:
- bmg0: γ=0 control (multi-day infra check)
- bmg05: γ=0.5
- bmg1: γ=1 (rerun of round 1's failed bmg1)
- bmg25: γ=2.5 (apples-to-apples vs round 1)

All runs grouped in WandB as `experiments_round_2`.
