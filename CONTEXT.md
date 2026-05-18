# Domain Glossary

This file defines canonical terms for the HRL + Brand Momentum AMoD extension. Implementation details belong in SPEC.md, not here.

---

## Core Terms

**Episode**
A single training rollout spanning N simulated days (configurable via `--num_days`). Brand momentum state persists across days; vehicle positions and queues reset between days.

**Day**
One simulated 1-hour peak-hour period (~20 timesteps at 3-minute resolution). The atomic unit of the meta-policy: one action in, one reward out.

**Timestep**
A single 3-minute decision interval within a day. The atomic unit of the low-level A2C+GCN policy.

**Operator**
One of two competing ride-hailing firms. Each operator runs its own low-level policy and, in HRL mode, its own meta-policy.

**Capture Rate** (`s_o(d)`)
The fraction of the total daily demand pool that operator `o` actually served on day `d`.
```
s_o(d) = served_o(d) / potential_demand(d)
```

**Potential Demand** (`potential_demand(d)`)
The sum of `d_original(n,j,t)` across all (origin, destination, timestep) triples for a full day — the total number of passengers who could have taken a ride, before the MNL choice model runs. Includes passengers who ultimately rejected both operators. Does NOT filter by vehicle availability.

**Brand Momentum** (`M_o(d)`)
An exponential moving average of the operator's daily capture rate, representing accumulated passenger loyalty.
```
M_o(d) = λ · M_o(d-1) + (1-λ) · s_o(d)
M_o(0) = 0.5  (symmetric prior at episode start)
```

**MNL Utility**
The per-passenger utility used in the Multinomial Logit discrete choice model:
```
U_{k,i,j,o}^t = β₀ + γ · M_o(d-1) − β_t · τ_{i,j} − (v̄/v_k) · p_{i,j,o}^t
```
`γ = 0` recovers the original single-day baseline exactly.

**Low-Level Policy**
The per-timestep A2C+GCN agent that outputs origin-based price scalars `ρ_o[i]` and rebalancing decisions.

**Meta-Policy**
The per-day PPO MLP agent that outputs per-region price multipliers `α_o[i]`, held constant across all timesteps within a day.

**Effective Price**
The price seen by passengers after composing meta-policy and low-level policy outputs:
```
p_effective[i,j,o] = clamp(α_o[i] · ρ_o[i], 0, 2) · p̄[i,j]
```
The low-level policy runs its full forward pass every timestep (observations, price head, rebalancing) exactly as in the baseline — it is not replaced by a constant. The meta-policy's α_o is a multiplicative wrapper applied on top of the low-level's price output. The low-level's observation does not include α_o; during Phase 1 the low-level is frozen so this has no training effect.

**Partial Reset**
The environment operation between days within an episode. Resets vehicle positions and queues to their initial conditions (`accInit` distribution) — identical to episode start. Brand momentum state is NOT reset. The low-level policy was pre-trained on the `accInit` distribution, so resetting to it keeps the low-level in-distribution across all days.

**Daily State** (`daily_state_o(d)`)
The fixed-size 7-element input vector to the meta-policy, aggregated after each day's simulation:
`[M_o, profit_o, avg_price_o, avg_price_opp, reb_cost_o, served_o, d/N]`
`d/N` is 1-indexed: day 1 → `1/N`, final day → `1.0`. This gives the meta-policy a clear "last day" signal for end-game strategy.

**Daily Profit** (`R_meta(d)`)
The meta-policy reward: total operator profit (pax revenue minus rebalancing cost) accumulated over all timesteps in day `d`.
