# Changelog

## 2026-06-06

- Add `--meta_action_mode soft` as an alternative to the default `multiplier` mode. Soft mode reinterprets the meta's daily α as a target price factor (2ρ) rather than a multiplier; ρ passes through unscaled and the low-level reward is shaped toward the target via `-λ·(2·mean(ρ) − α)²`. Additionally conditions both `GNNActor` and `GNNCritic` on the daily target via a zero-init `lin_alpha` layer (no-op when `meta_target=None`, so old checkpoints load unchanged). Across 3 seeds at 200k episodes, soft λ=0.1 yields ~68k vs multiplier's ~63k while eliminating the ρ compensation dynamic. Guarded to mode-2 origin pricing with an active meta-policy. See `Investigations/2026-06-06_compensation-fix-results.md` for the full experimental record (cap, goal, softaug, and cond were also tested and rejected).

## 2026-05-25

- Fix meta-policy reward normalization — rewards stored in the PPO buffer were raw daily profit (~3300/day) while the observation vector uses profit/reward_scalar (~1.65/day), causing a ~2000× scale mismatch. Dividing by reward_scalar before storage brings meta_critic_loss from ~3M to O(10–100) and stops the shared trunk from being dominated by critic gradients.
- Simplify meta-policy output from per-region vector to global scalar — the meta observation is fully aggregated (no per-region signal), so per-region outputs were unjustified and wasted sample efficiency with only ~8 transitions per PPO update. Scalar is broadcast to all regions. Updated SPEC.md §5 to match.
- Fix PPO log_prob bias in meta-policy — log probability is now computed on the raw (unclamped) sample; clamping to [0, 2] happens only on the value returned to the environment. Previously the log_prob was evaluated at the clamped value, giving incorrect importance ratios when actions hit the boundary.
- Add Picard fixed-point iteration solver for multi-day episodes — treats the day chain as a fixed-point system S = f(S) with pre-sampled seeds and meta-policy noise held constant per episode. Gauss-Seidel style in-place updates propagate improved brand-momentum predictions forward within each sweep, converging in K=2–3 iterations (opt-in via --parallel_days).

## 2026-05-24

- Speedup: Add or-tools min-cost-flow rebalancing solver (`solveRebFlow_ortools`, default via `--reb_solver ortools`).
- Speedup: Replace per-call `random.Random(self.seed).shuffle(...)` in `match_step_simple` with a stateful RNG cached at env init.

## 2026-05-20

- Treat each day's last step as terminal in the low-level A2C return computation — `env.reset_day()` re-initializes fleet positions, so cross-day rewards don't causally depend on prior-day actions. Previously the discounted return leaked across day boundaries, adding variance and degrading multi-day learning by ~40% per-day profit vs single-day baseline.
- Subtract rebalancing cost from meta-policy reward — aligns the meta objective with the low-level reward signal (revenue − operating_cost − rebalancing_cost) so the two levels don't optimize against each other.
- Add Claude skills for wandb-results and batch-jobs — speeds up common workflows around inspecting results and creating HPC jobs.
- Add round-2 experiment job scripts and batch jobs for LSF — needed to launch the next sweep on the HPC cluster.
- Fix reward/stats scalars — daily stats normalization was inconsistent with the low-level policy's `reward_scalar`.

## 2026-05-18

- Add WandB logging for both low-level (per-step) and meta-level (per-day) metrics — required to monitor the HRL extension separately from the baseline.
- Add meta-policy (PPO MLP) that guides each operator's low-level pricing across multi-day episodes — core HRL extension from SPEC.md.
- Add `DailyStatsAccumulator` for meta-policy input (#3) — provides the per-day aggregated state the meta-policy conditions on.
- Add brand momentum propagation across days within an episode — keeps EMA market share persistent for the MNL utility term.
- Add multi-day episode structure — prerequisite for hierarchical control and brand momentum dynamics.
- Add brand momentum state and utility term to AMoD env (#2) — extends the passenger choice model with an EMA market-share term (γ=0 reproduces baseline).
- Sum gradients across operators and apply training speedups — reduces wall-clock time per epoch.

## 2026-04-27

- Add regression test, project plan, and SPEC updates from feedback — locks in the extension scope and guards baseline behavior.
- Log competing operators to a shared WandB project — easier side-by-side comparison of runs.
- Repo cleanup — removed clutter ahead of the extension work.

## 2026-04-22 / 2026-04-13

- Initial commit from template, base paper and literature added — starting point built on Toft et al. (2026).