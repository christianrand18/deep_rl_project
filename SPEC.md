# Project Specification: Hierarchical RL and Brand Momentum in Competitive AMoD Systems

## Objective

Extend the competitive dual-operator AMoD reinforcement learning framework (Toft et al., 2026) by adding:
1. A **hierarchical meta-policy** that guides low-level pricing strategy across multiple simulated days
2. A **brand momentum mechanism** in the passenger discrete choice model that persists across days

The goal is to study whether operators learn long-term strategic behavior (e.g., early-day price undercutting to build brand loyalty) that differs from the single-day competitive equilibrium.

---

## Baseline System (Do Not Break)

The existing codebase implements:
- Two independent A2C+GCN operators competing on joint pricing and fleet rebalancing
- 3-minute discrete timesteps, ~20 steps per 1-hour peak episode
- Demand allocation via Multinomial Logit discrete choice model
- Origin-based price scalars + min-cost flow rebalancing LP (PuLP/CPLEX)
- Observations include own state + competitor prices; no shared parameters
- Three city scenarios: SF, Washington DC, NYC Manhattan South

All existing functionality must remain runnable via `main_a2c_multi_agent.py` with a flag to opt into the new HRL mode.

---

## New Components

### 1. Multi-Day Episode Structure (Issue #5)

**What changes:** An episode now spans N configurable days. Each day is one 1-hour peak-hour simulation (~20 timesteps). Brand momentum state is the only thing that carries between days; vehicle positions and queues reset each day.

```
Episode (N days, configurable via --num_days, default 8):
  For d in [1..N]:
    Reset vehicle positions and queues
    Run 1-hour intra-day simulation (existing low-level loop)
    Aggregate daily stats → feed to meta-policy
    Update brand momentum state
```

**Key design choice:** We simulate only the peak hour each day (19:00–20:00), not a full 24-hour cycle. This is computationally tractable and preserves the existing demand calibration.

---

### 2. Brand Momentum in the Utility Function (Issue #2)

**Implementation: exponential moving average (EMA) of daily capture rate.**

Passengers are modelled as having a loyalty bias toward each operator that adapts slowly across days. Brand momentum is tracked as an EMA of each operator's daily **capture rate** — the fraction of the total potential demand pool that the operator actually served:

```
M_o(d) = λ · M_o(d-1) + (1-λ) · s_o(d)      ← s_o(d) = served_o / potential_demand on day d
```

- `potential_demand` is the total passenger pool sampled before the MNL choice model runs (i.e. all passengers who *could* have taken a ride, including those who chose the outside option)
- This is distinct from relative market share (`served_o / total_served`): if both operators price too high and passengers opt out, both capture rates fall correctly, whereas relative market share would remain artificially stable
- `λ ∈ (0, 1)` is the decay parameter (high λ = slow adaptation / long memory; low λ = fast adaptation)
- `M_o(0) = 0.5` at episode start (equal prior for both operators)

Brand momentum enters the MNL utility function as:

```
U_{k,i,j,o}^t = β₀ + γ · M_o(d-1)  −  β_t · τ_{i,j}  −  (v̄/v_k) · p_{i,j,o}^t
```

- `γ` controls the strength of the loyalty effect; `γ = 0` recovers the original single-day choice model exactly
- `β₀` is the operator-specific baseline preference (initially symmetric)

**Flags:** `--brand_momentum_lambda` (default 0.9), `--brand_momentum_gamma` (default 0.0).

**Sensitivity levers for post-hoc analysis:** λ (decay speed), γ (momentum strength), N (days per episode).

---

**Potential extension — Rescorla-Wagner model (Sfeir et al., 2025):**

If the EMA implementation works and time allows, we can extend to the richer psychological model from Sfeir et al. where passengers form Q-value expectations updated via prediction errors:

```
δ_o(d)   = r_o(d) − Q_o(d)           ← prediction error: actual minus expected
Q_o(d+1) = Q_o(d) + α · δ_o(d)       ← Rescorla-Wagner update
```

The EMA above is a special case of this (r_o = market share, α = 1−λ, β absorbed into γ). Extending would add an explicit β parameter for passenger exploitation sensitivity and open up alternative feedback signals r_o(d) such as service completion rate or inverse wait time. This is out of scope for the initial implementation.

---

### 3. Meta-Policy Input (Issue #3)

After each day's simulation, the meta-policy for operator o observes a fixed-size daily summary vector:

```
daily_state_o(d) = [
  M_o(d-1),              # own capture rate momentum (own internal tracking)
  profit_o(d-1),         # own daily profit (normalised)
  avg_price_o(d-1),      # own average price scalar
  avg_price_opp(d-1),    # opponent average price (observable — check the app)
  reb_cost_o(d-1),       # own rebalancing cost (normalised)
  served_o(d-1),         # own absolute passengers served (normalised)
  d / N,                 # day progress within episode
]
```

**Information asymmetry rationale:** This input set mirrors what a real operator (e.g. Uber) would plausibly know about a competitor (e.g. Lyft) on a daily basis:
- Own figures (profit, costs, ridership, brand momentum) are known exactly from internal data
- Opponent prices are observable by checking the competitor's app — consistent with the baseline paper
- Opponent profit, fleet positions, and brand momentum are *not* observable and are excluded

**Why M_opp is excluded:** Opponent brand momentum is an internal construct not visible to a competitor. Crucially, the combination of own capture rate `M_o` and opponent price `avg_price_opp` already lets the meta-policy distinguish the two competitive scenarios:
- Own capture rate falling + opponent price low → competitor is winning passengers
- Own capture rate falling + opponent price high → passengers are choosing the outside option (neither operator)

This distinction would be lost if `s_o` were defined as relative market share (`served_o / total_served`), since passengers opting out would leave the ratio between operators unchanged.

---

### 4. Meta-Policy Reward (Issue #1)

The meta-policy reward at the end of each day:

```
R_meta(d) = total_profit_o(d)
```

Cumulative meta-policy return over the episode is the sum of daily profits. This aligns the meta-policy's objective with the operator's long-term interest and allows the agent to discover that sacrificing day-d profit for brand momentum may improve days d+1..N.

No auxiliary shaping terms initially. If training is unstable, consider adding a small bonus for market share improvement: `+α · (s_o(d) - s_o(d-1))`.

---

### 5. Meta-Policy Algorithm and Architecture (Issue #4)

**Algorithm:** PPO (preferred over A2C for the meta-level due to sparse daily rewards and longer episode horizon).

**Architecture:** MLP (no GCN — daily aggregated stats have no spatial graph structure).
- Input: daily_state vector (7 scalars)
- Hidden: 2 layers × 128 units, ReLU
- Actor head: outputs a **single global price multiplier scalar** α_o ∈ ℝ, clamped to [0, 2.0] via sigmoid scaling
- Critic head: scalar value estimate

**Why scalar, not per-region vector:** The meta observation is fully aggregated (no per-region signal), so per-region outputs have no signal to differentiate regions. With only ~8 transitions per PPO update, a high-dimensional output wastes sample efficiency. The scalar is broadcast to all regions each day.

**Action interface (direct constraints):**
The meta-policy outputs a global scalar α_o broadcast to all regions. At each low-level timestep, the effective price is:

```
p_effective_{i,j,o} = clamp(α_o · ρ_o[i], 0, 2) · p̄_{i,j}  → effective factor in [0, 4]
```

where ρ_o[i] is the low-level policy's origin-based price scalar and p̄_{i,j} is the reference price.

---

### 6. Training Strategy

**Stage 1 (Issue #7 — baseline):** Confirm existing code runs. Use existing single-day checkpoints or train fresh single-day low-level policies to convergence.

**Stage 2 (Phase 1 — single meta):** Freeze low-level weights for one operator. Train its meta-policy while the opponent uses a frozen pre-trained low-level policy with no meta. Validate the HRL loop and brand momentum mechanism.

**Stage 3 (Phase 2 — competitive meta):** Both operators have meta-policies. Train competitively using the same staged approach (both low-level policies pre-trained and frozen initially).

**Stage 4 (optional joint fine-tuning):** Unfreeze low-level weights and continue training all levels simultaneously.

---

### 7. Computational Efficiency (Issue #8)

With N-day episodes, each training episode is N× more expensive than the baseline. Mitigations:
- Parallelize episode rollouts (multiple environments in parallel) — investigate via Issue #8
- Use HPC for training runs
- Start with N=4 days to validate, scale to N=8+ for final experiments

---

### 8. Post-Hoc Analysis (Issue #6)

Key analyses to run after training:
- **Pricing trajectory across days**: Do operators systematically lower prices on early days and raise them later?
- **Brand momentum dynamics**: How fast does momentum accumulate? Does it create incumbency advantage?
- **Equilibrium comparison**: Multi-day HRL equilibrium vs. baseline single-day equilibrium (prices, profits, served demand, market share)
- **Sensitivity analysis**: Vary γ (momentum strength), λ (decay), N (days), to map the parameter space
- **Phase 1 vs Phase 2**: Does competitive meta-vs-meta differ from meta-vs-baseline?

---

## File Structure (Planned)

```
main_a2c_multi_agent.py         ← add --hrl flag and --num_days, existing code unchanged when flag off
src/
  algos/
    a2c_gnn_multi_agent.py      ← unchanged (low-level)
    meta_policy.py              ← NEW: MLP meta-policy (PPO)
  envs/
    amod_env_multi.py           ← add brand momentum state + multi-day reset logic
  misc/
    utils.py                    ← add daily aggregation helpers
main_hrl.py                     ← NEW: top-level training script for HRL mode
```

---

## Tech Stack

- Python, PyTorch, PyTorch Geometric (existing)
- PPO implementation: custom, modeled after existing A2C structure
- LP solver: PuLP / CPLEX (unchanged)
- Logging: WandB (add meta-level metrics: daily profit, market share, brand momentum, meta-policy loss)
- HPC: existing cluster setup

---

## What We Are NOT Doing

- Changing the low-level GCN architecture
- Changing the rebalancing LP formulation
- Adding more than 2 operators (for now)
- Simulating non-peak hours
- Changing the city data or demand calibration
