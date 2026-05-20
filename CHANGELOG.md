# Changelog

## 2026-05-20

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