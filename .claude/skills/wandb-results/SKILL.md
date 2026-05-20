---
name: wandb-results
description: Read, compare, and interpret experiment results from the deep_RL_project on Weights & Biases. Use this skill whenever the user asks about experiment results, run metrics, training progress, comparing runs, checking if runs are still alive, or interpreting what happened in a training run. Also trigger when the user mentions wandb, W&B, metrics, experiment results, run comparison, or asks things like "how are the runs going", "show me results", "compare the brand momentum experiments", or "what's the latest on the training". Even casual questions like "how's training" or "any results yet" should trigger this skill.
---

# WandB Results Reader

You have access to the W&B MCP tools for querying experiment data. All runs live in:
- **Entity:** `bertram-hage-danmarks-tekniske-universitet-dtu`
- **Project:** `deep_RL_project`

## If WandB calls fail

The wandb MCP connector may not be set up. Instruct the user to run:

```bash
claude mcp add --transport http wandb https://mcp.withwandb.com/mcp \
  --header "Authorization: Bearer <your-wandb-api-key>"
```

Get the API key from https://wandb.ai/authorize.

## Experiment Structure

Runs are organized into experiment groups via batch scripts in `batch_jobs/`. Read the batch scripts to understand what each experiment group contains — they define the flags and hyperparameters for each run.

Key experiment groups:
- `replication/` — baseline replication of Toft et al. (single-agent, dual-agent without meta-policy)
- `optimization_runs/` — parallelization experiments
- `experiments_round_1/` — first round of HRL experiments varying `brand_momentum_gamma`
- `experiments_round_2/` — second round with refined parameters, adds `wandb_group` tags

Runs are named by their `checkpoint_path` which encodes the config (e.g., `dual_meta_one_bmg05_r2` = dual-operator, one meta-policy, brand momentum gamma=0.5, round 2).

## Querying Runs

Use `mcp__wandb__query_wandb_tool` with GraphQL. Common patterns:

**List runs (most recent first):**
```graphql
query ($entity: String!, $project: String!) {
  project(name: $project, entityName: $entity) {
    runs(first: 20, order: "-createdAt") {
      edges { node { name displayName state createdAt summaryMetrics config } }
    }
  }
}
```

**Filter by wandb_group:**
```graphql
query ($entity: String!, $project: String!, $filters: JSONString!) {
  project(name: $project, entityName: $entity) {
    runs(filters: $filters, first: 20) {
      edges { node { name displayName state createdAt summaryMetrics config } }
    }
  }
}
```
With variables: `{"filters": "{\"config.wandb_group\": \"experiments_round_2\"}"}`.

**Get time-series history for a specific run:** Use `mcp__wandb__get_run_history_tool` with the run ID (the 8-char `name` field, not `displayName`).

Always pass `{"entity": "bertram-hage-danmarks-tekniske-universitet-dtu", "project": "deep_RL_project"}` as variables.

## Understanding Metrics

There are many metrics logged. Rather than memorizing a fixed list (they may change), reason about which metrics matter for the question at hand.

**To discover available metrics:** Query a run's `summaryMetrics` field — it returns all current metric keys as JSON. You can also read `main_a2c_multi_agent.py` to understand exactly what gets logged and when.

**Metric categories at a glance:**

- `agent{0,1}/...` — per-agent per-step metrics (reward, profit, demand served/unserved, prices, losses, gradients)
- `combined/...` — aggregated across both agents (total demand, rejection rate, total profit)
- `day/...` — daily-level metrics (brand momentum, market share, daily profit, meta multiplier, avg price)
- `meta/...` — meta-policy episode/day counters
- `training/...` — training state (warmup, reward scale)
- `vehicles/...` — fleet counts

**Key metrics for most analyses:**
- Profitability: `agent{0,1}/true_profit`, `combined/total_true_profit`
- Demand: `agent{0,1}/episode_served_demand`, `combined/rejection_rate`
- Pricing: `agent{0,1}/mean_price_scalar`, `agent{0,1}/mean_effective_price_scalar`, `day/agent{0,1}_avg_price`
- Brand momentum: `day/agent{0,1}_brand_momentum`, `agent{0,1}/brand_momentum`
- Meta-policy: `day/agent0_meta_multiplier_mean`, `agent0/meta_actor_loss`, `agent0/meta_critic_loss`
- Training stability: `agent{0,1}/actor_loss`, `agent{0,1}/critic_loss`, gradient norms

## Critical: Comparing Runs Correctly

**Episodes vs steps vs days:** Meta-policy runs have multiple days per episode (typically 7, set by `--num_days`). Baseline runs have 1 day = 1 episode. When comparing across run types, use **episodes** as the x-axis for apples-to-apples comparison. WandB may display `_step` on the x-axis by default — this is misleading because meta-policy runs log more steps per episode.

**Which agent has the meta-policy:** Check `meta_agent` in config (typically agent 0). Only the meta-agent has `meta_*` metrics. The other agent (`fix_agent`) runs a frozen low-level policy.

**Meta multiplier effect:** The effective price for the meta-agent is `meta_multiplier * low_level_price_scalar`. Compare `mean_effective_price_scalar` (which includes the meta multiplier) rather than `mean_price_scalar` (which is just the low-level output).

## Interpreting Results

Read `SPEC.md` for the full architecture and what to look for. Key questions:

1. **Does brand momentum change behavior?** Compare γ=0 (no momentum) vs γ>0. Look at pricing trajectories across days within an episode, brand momentum accumulation, and whether the meta-policy learns to undercut early.
2. **Profitability vs demand tradeoff:** Lower prices → more demand but less revenue per trip. Check if meta-policy finds a better long-term equilibrium.
3. **Training convergence:** Are losses decreasing? Are metrics stabilizing? Check gradient norms for exploding/vanishing gradients.
4. **Meta-policy learning:** Is `meta_actor_loss` / `meta_critic_loss` decreasing? Is the meta multiplier varying meaningfully or stuck at 1.0?

## Run Status

Runs often crash due to LSF time limits on the HPC cluster — this is expected. A crashed run with thousands of episodes still has useful data. Only flag concern if a run has very few episodes (< ~100). Both `running` and `crashed` states contain interpretable results.

## Presenting Results

When comparing runs, present a summary table with the key differentiating config parameter and the most relevant metrics. Keep it concise — the user can ask for deeper dives. Example format:

| Run | γ | Episodes | Agent0 Profit | Agent1 Profit | Rejection Rate |
|-----|---|----------|---------------|---------------|----------------|

For time-series analysis, describe trends rather than dumping raw numbers. Focus on what changed, not exhaustive data.
