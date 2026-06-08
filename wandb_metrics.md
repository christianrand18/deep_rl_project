# WandB Metrics Guide

What every metric logged by `main_a2c_multi_agent.py` means, and what to look for when
reading a run. Source of truth is the code — this doc explains intent; if it drifts from
`wandb.log(...)` calls in `main_a2c_multi_agent.py`, trust the code.

Metrics fall into five families by prefix: `agent{0,1}/...`, `combined/...`, `day/...`,
`meta/...` + `picard/...`, and `vehicles/...` + `training/...`. Per-agent and `combined/`
metrics are logged once per **episode** (x-axis: `episode`). `day/...` metrics are logged
once per **day** within an episode (x-axis: `meta/global_day`) — they only exist when a
meta-policy is active (`--num_days > 1`).

---

## `agent{0,1}/...` — per-agent, per-episode

### Outcome metrics (what happened)

| Metric | Meaning | Look for |
|---|---|---|
| `episode_reward` | Sum of step rewards the agent received over the episode (raw env reward, includes pricing + rebalancing steps) | Should trend up over training; compare agent0 vs agent1 to see who's winning the competition |
| `episode_served_demand` / `episode_unserved_demand` | Trips the agent served / lost (e.g. rejected, timed out) | Rising served demand without rising unserved demand = healthy growth; unserved demand creeping up can mean overpricing or fleet shortage |
| `episode_rebalancing_cost` | Cost paid moving empty vehicles between regions | High or growing cost relative to profit suggests wasteful rebalancing |
| `episode_waiting_time` | Mean wait time across served trips (`episode_waiting / episode_served_demand`) | Should stay low/stable; spikes indicate fleet mismatch with demand |
| `total_revenue` / `total_operating_cost` | Gross trip revenue / cost of running the fleet (driver wages etc.) | Revenue − operating cost ≈ the profit signal; useful to see which side of the ledger is moving |
| `true_profit` | The "real" profitability metric used to judge runs (revenue − costs, undistorted by reward shaping) | **The headline metric for comparing configs.** This is what investigations report (e.g. `compfix_multiplier` ≈ 60–69k) |
| `adjusted_profit` | Profit after some adjustment (e.g. excluding unprofitable trips) — a secondary profitability lens | Compare to `true_profit`; a large gap flags a lot of marginal/loss-making trips |
| `unprofitable_trips` | Count of trips served at a loss | Should be low; a rising count means the agent is serving demand it shouldn't (e.g. chasing market share at a loss) |

### Combined-agent rollups (`combined/...`)

| Metric | Meaning | Look for |
|---|---|---|
| `total_reward`, `total_served_demand`, `total_unserved_demand`, `total_rebalancing_cost` | Sum of the corresponding `agent0`/`agent1` metrics | System-wide health at a glance |
| `total_demand` / `rejected_demand` / `rejection_rate` | System-level demand that arrived / was rejected outright (neither operator served it) / the rate | High `rejection_rate` means the system as a whole is undersupplying — not an agent-specific problem |
| `total_true_profit`, `total_adjusted_profit`, `total_unprofitable_trips` | Sum across both agents | Use to judge whether the *system* is healthier under a config, independent of which operator wins |

### Training / optimization metrics

| Metric | Meaning | Look for |
|---|---|---|
| `actor_loss` / `critic_loss` | A2C low-level policy/value loss | Should not diverge or oscillate wildly; some noise is normal. Sudden jumps often coincide with collapse events visible elsewhere (e.g. profit crashing) |
| `actor_grad_norm` / `critic_grad_norm` | L2 norm of gradients before the optimizer step | Should stay bounded; near-zero for long stretches = vanishing gradients (agent stopped learning); huge spikes = instability/exploding gradients |
| `advantage_mean` / `advantage_std` | Mean/std of the A2C advantage estimates used for the policy update | `advantage_mean` should hover near zero (advantages are mean-centered by construction); `advantage_std` collapsing toward 0 can indicate the critic has converged (or stalled); blow-ups indicate noisy value estimates |
| `mean_log_prob` | Mean log-probability the policy assigned to its own sampled actions over the episode | Trending toward 0 (less negative) = policy becoming more confident/deterministic; very large negative spikes can flag exploration blowing up |
| `reward_scale` | The scalar dividing raw rewards before they reach the policy (`= args.reward_scalar` unless adapted) | Mostly a sanity check — confirms low-level and meta-level rewards are on a comparable scale (per `CLAUDE.md` convention) |
| `reward_scale` reported per fixed agent is dummy/0 | When `--fix_agent` is set, that agent doesn't train — its loss/grad-norm/advantage metrics are logged as `0.0` placeholders | Don't read anything into a flat `0` line for the fixed agent — it's expected, not a bug |

### Pricing metrics (modes 1 and 2 only — i.e. `od_price_actions`/origin pricing modes)

| Metric | Meaning | Look for |
|---|---|---|
| `mean_price_scalar` | Mean of the *raw* price multiplier ρ output by the low-level policy (before any meta multiplier is applied) | This is "what the low-level wants to charge." Compare against `mean_effective_price_scalar` to see how much the meta-policy is overriding it |
| `mean_effective_price_scalar` | Mean of the price multiplier actually applied after the meta multiplier: `clip(α · ρ, 0, 2)` | **Use this (not `mean_price_scalar`) when judging actual pricing behavior** — it's what passengers actually saw |
| `meta_shaping_term` | Mean per-step shaping reward λ·(target − ρ) injected into the low-level reward (only meaningful in `soft`/`multiplier_soft` meta-action modes) | Non-zero only when `--meta_action_mode soft`/`multiplier_soft`; large magnitude relative to the extrinsic reward can destabilize training (see Round 2 of the compensation-fix investigation, where λ=0.3–1.0 caused collapse) |

### Concentration metrics (mode-specific — Dirichlet/Beta action distributions)

| Metric | Meaning | Look for |
|---|---|---|
| `mean/min/max_concentration_dirichlet` (mode 0, 2) | Stats on the Dirichlet concentration parameters the policy outputs for spatial price allocation | Concentration trending toward very large values = the policy is becoming near-deterministic (low exploration); very small/near-uniform = high exploration or undertrained |
| `mean/min/max_concentration_alpha` / `..._beta` (mode 1, 2) | Same idea for the Beta-distribution price-scalar action | Same read as above — watch for collapse to extreme values early in training (premature convergence) |

### Meta-policy training metrics (only when a meta-policy is attached to that agent)

| Metric | Meaning | Look for |
|---|---|---|
| `meta_actor_loss` / `meta_critic_loss` | PPO actor/critic loss for the daily meta-policy, updated once per episode | Should generally trend down / stabilize as the meta-policy learns; persistent high variance suggests the meta-reward signal is too noisy (few days per episode = few samples per update) |
| `meta_advantage_mean` / `meta_advantage_std` | Same idea as the low-level advantage stats, but for the meta-policy's PPO update | `meta_advantage_mean` near zero is expected; watch `meta_advantage_std` for collapse (meta stuck on one action) vs. blow-up (unstable meta-reward) |

### Brand momentum (only logged when `--num_days > 1`)

| Metric | Meaning | Look for |
|---|---|---|
| `agent{0,1}/brand_momentum` | End-of-episode snapshot of the EMA of daily market share that feeds back into the passenger choice utility (`U += γ · brand_momentum`) | With `brand_momentum_gamma = 0` this should sit near its 0.5 initial value and have **no effect** (baseline-equivalence requirement in `CLAUDE.md`). With `γ > 0`, look for divergence between agent0/agent1 momentum — that's the feedback loop ("rich get richer") taking hold |

---

## `day/...` — per-day metrics (meta-policy runs only)

Logged once per simulated day; use `meta/global_day` as the x-axis (continuous across
episodes) rather than `_step`, since step counts differ between meta and non-meta runs.

| Metric | Meaning | Look for |
|---|---|---|
| `meta/global_day`, `meta/episode`, `meta/day_in_episode` | Indices: a monotonically increasing day counter across the whole run, the episode it belongs to, and the day-within-episode (0…`num_days`-1) | Use `global_day` as x-axis for continuous trends; use `day_in_episode` to look at within-episode dynamics (e.g. "does undercutting happen early in the week and recover later?") |
| `agent{0,1}_daily_profit` | That day's profit (`accumulator.profit / reward_scalar`) | Day-level granularity on profitability — look for trends across `day_in_episode` (e.g. does one operator front-load profit and coast later?) |
| `agent{0,1}_meta_reward` | The reward signal the meta-policy actually trains on for that day (`(profit − reb_cost) / reward_scalar`) | This is what the meta-policy is optimizing — compare its trajectory to `daily_profit` to confirm the meta reward tracks what you actually care about |
| `agent{0,1}_brand_momentum` | Brand momentum value at the end of that day | Day-by-day view of the EMA — use this (rather than the episode-end snapshot) to see the accumulation/decay dynamics within an episode |
| `agent{0,1}_step_reward_sum` | Sum of that day's step-level rewards (the slice of `episode_reward` accumulated during that specific day) | Cross-check against `daily_profit` — large divergence can mean reward shaping (e.g. `meta_shaping_term`) is injecting a lot of signal that doesn't show up in profit |
| `agent{0,1}_market_share` | Fraction of total system demand served by that agent that day (`served / total_demand`) | **Central metric for brand-momentum analysis.** Look at how it co-moves with `brand_momentum` and `meta_multiplier`/`avg_price` — does undercutting early grow share, and does that share persist later in the episode? |
| `agent{0,1}_avg_price` | Mean *effective* price for the day (post meta-multiplier, post-clip) | Day-level view of what passengers actually paid; compare across `day_in_episode` to see pricing strategy evolve across the week |
| `agent{0,1}_meta_multiplier` | The meta-policy's chosen daily multiplier α for that day | **Key meta-policy output.** Persistent values far from 1.0 (e.g. 0.5–0.75 "undercut") indicate an active strategy; values stuck at exactly 1.0 mean the meta-policy isn't doing anything meaningful yet. See compensation-fix investigation for the canonical "meta drives α low, low-level compensates by raising ρ" failure mode |
| `agent{0,1}_avg_price_raw` | Mean *raw* low-level price scalar ρ for the day (pre-multiplier) | Compare to `avg_price` (effective) — a large, persistent gap between raw ρ and effective price is the signature of the low-level "fighting" the meta-policy (compensation behavior) |

---

## `meta/...` and `picard/...` — meta-policy / Picard-iteration bookkeeping

| Metric | Meaning | Look for |
|---|---|---|
| `meta/global_day`, `meta/episode`, `meta/day_in_episode` | (duplicated into `day/...` logs above; same indices) | — |
| `episode` (in the `picard/...` log block) | Episode index, logged alongside Picard convergence stats | Used to align Picard diagnostics with the rest of the per-episode logs |
| `picard/K_used` | Number of fixed-point iterations the Picard solver needed to converge that episode (only logged with `--parallel_days`) | Rising `K_used` over training = the daily dynamics are getting harder to predict self-consistently (more interaction/non-stationarity); flat and low = the parallel-days approximation is cheap and stable |
| `picard/final_delta` | The residual error at the iteration where the solver stopped | Should be small and stable; growth suggests the fixed point is becoming harder to pin down (early warning that `--parallel_days` may be producing less accurate day trajectories) |
| `picard/converged` | 1 if the solver converged within its iteration budget, 0 if it hit the cap without converging | **Watch for runs of `0`s** — non-convergence means the "parallel days" approximation didn't reach a self-consistent solution that episode, so its logged day-level data is less trustworthy |

---

## `vehicles/...` — fleet bookkeeping (sanity checks, not performance)

| Metric | Meaning | Look for |
|---|---|---|
| `agent0_total` / `agent1_total` / `combined_total` | Vehicle counts per operator and combined, recomputed at end of episode | Should match expectations for the scenario; useful to confirm fleet sizes are configured as intended |
| `initial` | Vehicle count at episode start (should be constant across episodes for a given scenario) | — |
| `discrepancy` | `abs(initial - combined_total)` — vehicles "lost" or "gained" somewhere in the simulation | **Should be ~0.** Any persistent non-zero value indicates a bug in vehicle accounting (vehicles created/destroyed incorrectly) — this is a correctness check, not a performance metric |

---

## `training/...` — warmup state

| Metric | Meaning | Look for |
|---|---|---|
| `critic_warmup_active` | 1/0 flag — whether the run is still in the critic-only warmup period (`--critic_warmup_episodes`) | Use to align "before vs. after actor updates start" when reading loss/reward curves — expect a regime change at the transition |
| `warmup_progress` | Fraction of warmup completed (`min(1, i_episode / critic_warmup_episodes)`) | Mostly a convenience x-axis for the warmup phase; not meaningful once it hits 1.0 |

---

## Cross-cutting things to check on any run

1. **Use the right x-axis.** Episode-level metrics → `episode`. Day-level metrics →
   `meta/global_day` (continuous across episodes, unlike `day_in_episode`). Don't compare
   meta-policy runs to baseline runs on raw `_step` — meta runs log far more steps per
   episode (see `wandb-results` skill notes).
2. **Effective vs. raw pricing.** Always read `mean_effective_price_scalar` /
   `day/agent{}_avg_price` (what happened) rather than `mean_price_scalar` /
   `avg_price_raw` (what the low-level wanted) when judging real-world pricing behavior —
   but compare the two when diagnosing meta/low-level conflict (compensation).
3. **γ=0 baseline-equivalence.** For any brand-momentum run, confirm `brand_momentum_gamma
   = 0` configs show flat `brand_momentum` near 0.5 with no measurable effect on
   `true_profit`/`market_share` — that's the invariant `CLAUDE.md` requires.
4. **`true_profit` is the headline number**, not `episode_reward` (which can include
   shaping terms like `meta_shaping_term` that inflate the optimized signal without
   reflecting real profitability).
5. **Fixed-agent metrics are placeholders.** When `--fix_agent` is set, that agent's
   losses/grad-norms/advantages/meta-metrics are logged as `0` — don't mistake that for a
   training failure.
6. **Vehicle discrepancy should always be ~0** — treat any non-zero trend as a correctness
   bug to chase down, not a performance signal.