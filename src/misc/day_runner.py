"""Day simulation loop for multi-day AMoD episodes.

Extracted from main_a2c_multi_agent.py so the same callable can be invoked
sequentially today and dispatched to parallel workers downstream. The body
is a verbatim copy of the original for-day-loop body; only the surrounding
unpack/repack of `ctx` is new — sequential behaviour must be unchanged.

Design contract:
  - The body reads roughly twenty-five accumulators plus several stable
    references (env, args, model_agents, meta_policies, accumulator,
    picard_solver, solveRebFlow, i_episode). They all live on `ctx`
    (a SimpleNamespace), which is constructed by the caller and threaded
    through every day's call.
  - Seven values get *reassigned* inside the body (the Python name gets
    rebound, so a write-back is required for the caller to see the change):
        obs, action_rl, episode_reward, meta_obs, meta_multipliers,
        episode_rejected_demand, episode_total_demand
    These are read from ctx at the top of the function and written back
    to ctx at the bottom.
  - All other accumulators are *mutated in place* (`dict[k] += ...`,
    `list.append(...)`, etc.). They're shared by reference between ctx
    and the function's local names, so mutations are visible to the
    caller without any explicit write-back.
"""

import time as _time

import numpy as np

from src.misc.utils import dictsum


def run_day(i_day, ctx):
    """Run one day of the multi-agent AMoD episode.

    ``ctx`` carries everything the day loop needs. See the module docstring
    for the unpack/repack contract.
    """
    # --- Unpack so the body below is a verbatim copy from main ---
    env = ctx.env
    args = ctx.args
    model_agents = ctx.model_agents
    meta_policies = ctx.meta_policies
    accumulator = ctx.accumulator
    picard_solver = ctx.picard_solver
    solveRebFlow = ctx.solveRebFlow
    i_episode = ctx.i_episode

    obs = ctx.obs
    action_rl = ctx.action_rl
    meta_obs = ctx.meta_obs
    meta_multipliers = ctx.meta_multipliers

    episode_reward = ctx.episode_reward
    episode_served_demand = ctx.episode_served_demand
    episode_unserved_demand = ctx.episode_unserved_demand
    episode_rebalancing_cost = ctx.episode_rebalancing_cost
    episode_total_revenue = ctx.episode_total_revenue
    episode_total_operating_cost = ctx.episode_total_operating_cost
    episode_waiting = ctx.episode_waiting
    episode_rejected_demand = ctx.episode_rejected_demand
    episode_total_demand = ctx.episode_total_demand
    episode_rejection_rates = ctx.episode_rejection_rates
    episode_true_profit = ctx.episode_true_profit
    episode_adjusted_profit = ctx.episode_adjusted_profit
    episode_unprofitable_trips = ctx.episode_unprofitable_trips
    actions_price = ctx.actions_price
    actions_effective_price = ctx.actions_effective_price
    meta_shaping_term = ctx.meta_shaping_term
    episode_logprobs = ctx.episode_logprobs

    # Mode-specific concentration trackers — present for the mode in use;
    # pulled defensively so the body's references resolve. Branches gated
    # on env.mode never touch the others.
    actions_concentration_alpha = getattr(ctx, 'actions_concentration_alpha', None)
    actions_concentration_beta = getattr(ctx, 'actions_concentration_beta', None)
    actions_concentration_dirichlet = getattr(ctx, 'actions_concentration_dirichlet', None)
    episode_min_concentration_alpha = getattr(ctx, 'episode_min_concentration_alpha', None)
    episode_max_concentration_alpha = getattr(ctx, 'episode_max_concentration_alpha', None)
    episode_min_concentration_beta = getattr(ctx, 'episode_min_concentration_beta', None)
    episode_max_concentration_beta = getattr(ctx, 'episode_max_concentration_beta', None)
    episode_min_concentration_dirichlet = getattr(ctx, 'episode_min_concentration_dirichlet', None)
    episode_max_concentration_dirichlet = getattr(ctx, 'episode_max_concentration_dirichlet', None)

    # Fine-grained day-phase timing (parallel profiling — ctx carries the dict)
    _tb = getattr(ctx, 'timing_breakdown', None)
    _t_phase = None

    def _t_start():
        nonlocal _t_phase
        if _tb is not None:
            _t_phase = _time.perf_counter()

    def _t_end(phase: str):
        if _tb is not None and _t_phase is not None:
            _tb[phase] = _tb.get(phase, 0.0) + _time.perf_counter() - _t_phase

    # === BEGIN verbatim copy from main_a2c_multi_agent.py (for i_day body) ===
    # Skip env.reset_day() when the caller (e.g. parallel workers) has
    # already put env into a fresh "start of day" state with its own seeded
    # reset. Sequential callers don't set this flag → behaviour unchanged.
    if i_day > 0 and not getattr(ctx, 'env_pre_reset', False):
        obs = env.reset_day()
        action_rl = {a: [0.0] * env.nregion for a in [0, 1]}
    day_served = {0: 0, 1: 0}
    day_total_demand = 0
    day_price_raw = {0: [], 1: []}  # per-day raw ρ (pre-composition) for corr diagnostics
    # Snapshot episode_reward at day start so we can compute per-day step-reward sum
    # for reward-attribution debug logging (compare to day/agent{a}_daily_profit).
    day_reward_start = dict(episode_reward)
    done = False
    step = 0

    # Reset daily accumulator (used for both meta-policy state and per-day WandB logs)
    if accumulator is not None:
        _t_start()
        accumulator.reset(env)
        _t_end('accumulator')
    # Meta-policy: select daily multipliers
    if args.parallel_days:
        # Picard: inject brand_momentum, seed RNGs, return deterministic alpha.
        _t_start()
        meta_multipliers = picard_solver.prepare_day(i_day, env)
        _t_end('prepare_day')
    elif meta_policies:
        for _a in meta_policies:
            meta_multipliers[_a] = meta_policies[_a].select_action(meta_obs[_a])

    while not done:
        # Capture prices about to be submitted (for accumulator tracking)
        if meta_policies and env.mode in [1, 2] and step > 0:
            if env.mode == 1:
                submitted_prices = {a: np.array(action_rl[a]) for a in [0, 1]}
            elif args.od_price_actions:
                submitted_prices = {a: np.array(action_rl[a])[:, :env.nregion] for a in [0, 1]}
            else:
                submitted_prices = {a: np.array(action_rl[a])[:, 0] for a in [0, 1]}
        else:
            submitted_prices = None
        if env.mode == 0:
            # Make Match Step
            _t_start()
            obs, paxreward, done, info, system_info, _, _ = env.match_step_simple()
            _t_end('match_step')

            # Update episode reward
            episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}

            # Get actions and concentrations
            action_rl = {}
            concentrations = {}
            for a in [0, 1]:
                if a == args.fix_agent:
                    # Fixed agent: use actual initial distribution for rebalancing
                    # Convert initial vehicle counts to proportions
                    total_vehicles = sum(env.agent_initial_acc[a].values())
                    action_rl[a] = np.array([
                        env.agent_initial_acc[a][env.region[i]] / total_vehicles 
                        for i in range(env.nregion)
                    ])
                    concentrations[a] = np.zeros((env.nregion, 1))  # Dummy for tracking
                else:
                    _t_start()
                    action_rl[a], concentrations[a], logprob = model_agents[a].select_action(obs[a], return_concentration=True)
                    _t_end('select_action')
                    episode_logprobs[a].append(logprob)

            # Track concentration (mode 0: Dirichlet concentration for rebalancing)
            for a in [0, 1]:
                if a != args.fix_agent:
                    actions_concentration_dirichlet[a].append(np.mean(concentrations[a]))
                    # Update episode-level min/max
                    episode_min_concentration_dirichlet[a] = min(episode_min_concentration_dirichlet[a], np.min(concentrations[a]))
                    episode_max_concentration_dirichlet[a] = max(episode_max_concentration_dirichlet[a], np.max(concentrations[a]))

            # Determine which agents are active (not fixed)
            # Compute desired accumulation for all agents
            desiredAcc = {}
            for a in [0, 1]:
                if a == args.fix_agent:
                    # For fixed agent, distribute vehicles uniformly across all regions
                    current_total = dictsum(env.agent_acc[a], env.time + 1)
                    base_per_region = current_total // env.nregion
                    remainder = current_total % env.nregion
                    # Distribute uniformly with remainder going to first regions
                    desiredAcc[a] = {
                        env.region[i]: base_per_region + (1 if i < remainder else 0)
                        for i in range(env.nregion)
                    }
                else:
                    # For active agent, use action to determine desired distribution
                    desiredAcc[a] = {
                        env.region[i]: int(action_rl[a][i] * dictsum(env.agent_acc[a], env.time + 1))
                        for i in range(env.nregion)
                    }

            # Compute rebalancing flows for both agents sequentially
            _t_start()
            rebAction = {a: solveRebFlow(env, desiredAcc[a], a) for a in [0, 1]}
            _t_end('solve_reb')

            _t_start()
            new_obs, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
            _t_end('reb_step')
            episode_reward = {a: episode_reward[a] + rebreward[a] for a in [0, 1]}

            for agent_id in [0, 1]:
                model_agents[agent_id].rewards.append((paxreward[agent_id] + rebreward[agent_id]))

        elif env.mode == 1:
            _t_start()
            obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
            _t_end('match_step')

            episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}

            for agent_id in [0, 1]:
                model_agents[agent_id].rewards.append(paxreward[agent_id])

            # Get actions and concentrations
            action_rl = {}
            concentrations = {}
            for a in [0, 1]:
                if a == args.fix_agent:
                    # Fixed agent: environment handles price override to 0.5
                    # Just provide any valid pricing action (will be ignored)
                    if args.od_price_actions:
                        action_rl[a] = np.full((env.nregion, env.nregion), 0.5)
                    else:
                        action_rl[a] = np.array([0.5] * env.nregion)
                    concentrations[a] = np.zeros((env.nregion, 2))  # Dummy for tracking
                else:
                    _t_start()
                    action_rl[a], concentrations[a], logprob = model_agents[a].select_action(obs[a], return_concentration=True)
                    _t_end('select_action')
                    episode_logprobs[a].append(logprob)

            # Track prices during episode (mode 1: action_rl is price scalar)
            for a in [0, 1]:
                if a == args.fix_agent:
                    # Fixed agent always uses 0.5 scalar
                    actions_price[a].append(1.0)  # 2 * 0.5 = 1.0 (base price)
                else:
                    actions_price[a].append(np.mean(2 * np.array(action_rl[a])))

            # Track concentration (mode 1: Beta distribution - alpha and beta)
            for a in [0, 1]:
                if a != args.fix_agent:
                    if args.od_price_actions:
                        # OD: concentrations shape [1, nregion, nregion, 2]
                        actions_concentration_alpha[a].append(np.mean(concentrations[a][0, :, :, 0]))
                        actions_concentration_beta[a].append(np.mean(concentrations[a][0, :, :, 1]))
                        episode_min_concentration_alpha[a] = min(episode_min_concentration_alpha[a], np.min(concentrations[a][0, :, :, 0]))
                        episode_max_concentration_alpha[a] = max(episode_max_concentration_alpha[a], np.max(concentrations[a][0, :, :, 0]))
                        episode_min_concentration_beta[a] = min(episode_min_concentration_beta[a], np.min(concentrations[a][0, :, :, 1]))
                        episode_max_concentration_beta[a] = max(episode_max_concentration_beta[a], np.max(concentrations[a][0, :, :, 1]))
                    else:
                        # Origin: concentrations shape [1, nregion, 2]
                        actions_concentration_alpha[a].append(np.mean(concentrations[a][0, :, 0]))
                        actions_concentration_beta[a].append(np.mean(concentrations[a][0, :, 1]))
                        episode_min_concentration_alpha[a] = min(episode_min_concentration_alpha[a], np.min(concentrations[a][0, :, 0]))
                        episode_max_concentration_alpha[a] = max(episode_max_concentration_alpha[a], np.max(concentrations[a][0, :, 0]))
                        episode_min_concentration_beta[a] = min(episode_min_concentration_beta[a], np.min(concentrations[a][0, :, 1]))
                        episode_max_concentration_beta[a] = max(episode_max_concentration_beta[a], np.max(concentrations[a][0, :, 1]))

            # Bound the meta-controlled agent's pre-meta scalar to [min, max]
            # (ablation: limits how much the low-level can compensate against α)
            if meta_policies and (args.low_level_scalar_min > 0.0 or args.low_level_scalar_max < 1.0):
                lo, hi = args.low_level_scalar_min, args.low_level_scalar_max
                for _a in meta_policies:
                    action_rl[_a] = lo + (hi - lo) * np.array(action_rl[_a])

            # Apply meta multipliers to mode 1 pricing actions
            # Low-level scalar ρ ∈ [0, 1] (env scales by 2 → factor [0, 2])
            # Meta multiplier α ∈ [0, 2]; combined α·ρ clipped to [0, 2] → factor [0, 4]
            if meta_policies:
                for _a in [0, 1]:
                    if _a in meta_policies:
                        arr = np.array(action_rl[_a])
                        action_rl[_a] = np.clip(meta_multipliers[_a] * arr, 0.0, 2.0)

            # Track effective prices (post-meta multiplier) during episode (mode 1)
            for a in [0, 1]:
                if a == args.fix_agent:
                    actions_effective_price[a].append(1.0)
                else:
                    actions_effective_price[a].append(np.mean(2 * np.array(action_rl[a])))

            # Matching update (global step)
            _t_start()
            env.matching_update()
            _t_end('matching_update')

        elif env.mode == 2:
            # --- Matching step ---
            _t_start()
            obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
            _t_end('match_step')

            episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}

            # Get actions and concentrations
            action_rl = {}
            concentrations = {}
            for a in [0, 1]:
                if a == args.fix_agent:
                    # Fixed agent: environment handles price override to 0.5
                    total_vehicles = sum(env.agent_initial_acc[a].values())
                    reb_action = np.array([
                        env.agent_initial_acc[a][env.region[i]] / total_vehicles 
                        for i in range(env.nregion)
                    ])
                    if args.od_price_actions:
                        # Mode 2 OD action shape: [nregion, nregion+1] where [:, :nregion] = OD prices, [:, -1] = reb
                        action_rl[a] = np.column_stack([
                            np.full((env.nregion, env.nregion), 0.5),  # OD prices (will be overridden)
                            reb_action.reshape(-1, 1)  # Rebalancing
                        ])
                    else:
                        # Mode 2 action shape: [nregion, 2] where [:, 0] = price scalar, [:, 1] = reb action
                        action_rl[a] = np.column_stack([
                            np.array([0.5] * env.nregion),  # Price (will be overridden to 0.5 by env)
                            reb_action  # Rebalancing: actual initial distribution
                        ])
                    concentrations[a] = np.zeros((env.nregion, 3))  # Dummy for tracking
                else:
                    _t_start()
                    action_rl[a], concentrations[a], logprob = model_agents[a].select_action(obs[a], return_concentration=True)
                    _t_end('select_action')
                    episode_logprobs[a].append(logprob)

            # Track prices during episode (mode 2: price part of action)
            for a in [0, 1]:
                if a == args.fix_agent:
                    # Fixed agent always uses 0.5 scalar
                    actions_price[a].append(1.0)  # 2 * 0.5 = 1.0 (base price)
                else:
                    if args.od_price_actions:
                        # OD: action shape [nregion, nregion+1], prices in [:, :nregion]
                        actions_price[a].append(np.mean(2 * np.array(action_rl[a])[:, :env.nregion]))
                    else:
                        actions_price[a].append(np.mean(2 * np.array(action_rl[a])[:, 0]))

            # Per-day raw price scalar (pre-composition) for the corr(α, raw ρ) diagnostic
            if not args.od_price_actions:
                for a in [0, 1]:
                    if a != args.fix_agent:
                        day_price_raw[a].append(float(np.mean(np.array(action_rl[a])[:, 0])))

            # Track concentration (mode 2: Beta + Dirichlet)
            for a in [0, 1]:
                if a != args.fix_agent:
                    if args.od_price_actions:
                        # OD: concentrations is dict {'beta': [1, nregion, nregion, 2], 'dirichlet': [1, nregion, 1]}
                        actions_concentration_alpha[a].append(np.mean(concentrations[a]['beta'][0, :, :, 0]))
                        actions_concentration_beta[a].append(np.mean(concentrations[a]['beta'][0, :, :, 1]))
                        actions_concentration_dirichlet[a].append(np.mean(concentrations[a]['dirichlet'][0, :, 0]))
                        episode_min_concentration_alpha[a] = min(episode_min_concentration_alpha[a], np.min(concentrations[a]['beta'][0, :, :, 0]))
                        episode_max_concentration_alpha[a] = max(episode_max_concentration_alpha[a], np.max(concentrations[a]['beta'][0, :, :, 0]))
                        episode_min_concentration_beta[a] = min(episode_min_concentration_beta[a], np.min(concentrations[a]['beta'][0, :, :, 1]))
                        episode_max_concentration_beta[a] = max(episode_max_concentration_beta[a], np.max(concentrations[a]['beta'][0, :, :, 1]))
                        episode_min_concentration_dirichlet[a] = min(episode_min_concentration_dirichlet[a], np.min(concentrations[a]['dirichlet'][0, :, 0]))
                        episode_max_concentration_dirichlet[a] = max(episode_max_concentration_dirichlet[a], np.max(concentrations[a]['dirichlet'][0, :, 0]))
                    else:
                        # Origin: concentrations[a] has shape (1, nregion, 3)
                        actions_concentration_alpha[a].append(np.mean(concentrations[a][0, :, 0]))
                        actions_concentration_beta[a].append(np.mean(concentrations[a][0, :, 1]))
                        actions_concentration_dirichlet[a].append(np.mean(concentrations[a][0, :, 2]))
                        episode_min_concentration_alpha[a] = min(episode_min_concentration_alpha[a], np.min(concentrations[a][0, :, 0]))
                        episode_max_concentration_alpha[a] = max(episode_max_concentration_alpha[a], np.max(concentrations[a][0, :, 0]))
                        episode_min_concentration_beta[a] = min(episode_min_concentration_beta[a], np.min(concentrations[a][0, :, 1]))
                        episode_max_concentration_beta[a] = max(episode_max_concentration_beta[a], np.max(concentrations[a][0, :, 1]))
                        episode_min_concentration_dirichlet[a] = min(episode_min_concentration_dirichlet[a], np.min(concentrations[a][0, :, 2]))
                        episode_max_concentration_dirichlet[a] = max(episode_max_concentration_dirichlet[a], np.max(concentrations[a][0, :, 2]))

            # Bound the meta-controlled agent's pre-meta scalar to [min, max]
            # (ablation: limits how much the low-level can compensate against α)
            if meta_policies and (args.low_level_scalar_min > 0.0 or args.low_level_scalar_max < 1.0):
                lo, hi = args.low_level_scalar_min, args.low_level_scalar_max
                for _a in meta_policies:
                    if args.od_price_actions:
                        action_rl[_a][:, :env.nregion] = lo + (hi - lo) * np.array(action_rl[_a][:, :env.nregion])
                    else:
                        action_rl[_a][:, 0] = lo + (hi - lo) * np.array(action_rl[_a][:, 0])

            # Compose the meta action into the mode-2 price column.
            # Low-level scalar ρ ∈ [0, 1] (env scales by 2 → factor [0, 2]).
            #   multiplier (default): effective = clip(α·ρ, 0, 2)  → factor [0, 4]
            #   soft:                 α is a target price factor, not a multiplier;
            #                         ρ passes through unscaled (shaping done in the reward).
            # soft is guarded to mode-2 origin pricing, so the od_price path
            # below stays multiplier-only.
            if meta_policies:
                for _a in [0, 1]:
                    if _a in meta_policies:
                        if args.od_price_actions:
                            action_rl[_a][:, :env.nregion] = np.clip(
                                meta_multipliers[_a] * action_rl[_a][:, :env.nregion], 0.0, 2.0)
                        elif args.meta_action_mode == "soft":
                            pass  # target, not multiplier — ρ passes through
                        else:  # multiplier
                            action_rl[_a][:, 0] = np.clip(
                                meta_multipliers[_a] * action_rl[_a][:, 0], 0.0, 2.0)

            # Track effective prices (post-meta multiplier) during episode (mode 2)
            for a in [0, 1]:
                if a == args.fix_agent:
                    actions_effective_price[a].append(1.0)
                else:
                    if args.od_price_actions:
                        actions_effective_price[a].append(np.mean(2 * np.array(action_rl[a])[:, :env.nregion]))
                    else:
                        actions_effective_price[a].append(np.mean(2 * np.array(action_rl[a])[:, 0]))

            # --- Desired Acc computation ---
            # Compute desired accumulation for all agents
            desiredAcc = {}
            for a in [0, 1]:
                if a == args.fix_agent:
                    # For fixed agent, distribute vehicles uniformly across all regions
                    current_total = dictsum(env.agent_acc[a], env.time + 1)
                    base_per_region = current_total // env.nregion
                    remainder = current_total % env.nregion
                    # Distribute uniformly with remainder going to first regions
                    desiredAcc[a] = {
                        env.region[i]: base_per_region + (1 if i < remainder else 0)
                        for i in range(env.nregion)
                    }
                else:
                    # For active agent, use action to determine desired distribution
                    desiredAcc[a] = {
                        env.region[i]: int(action_rl[a][i, -1] * dictsum(env.agent_acc[a], env.time + 1))
                        for i in range(env.nregion)
                    }

            # --- Rebalancing step ---
            # Compute rebalancing flows for both agents sequentially
            _t_start()
            rebAction = {a: solveRebFlow(env, desiredAcc[a], a) for a in [0, 1]}
            _t_end('solve_reb')

            _t_start()
            new_obs, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
            _t_end('reb_step')

            episode_reward = {a: episode_reward[a] + rebreward[a] for a in [0, 1]}
            for agent_id in [0, 1]:
                r = paxreward[agent_id] + rebreward[agent_id]
                # soft: shape the low-level reward toward the meta target price factor
                # (2·mean(ρ)). Pre-multiply by reward_scale so that after
                # training_step's /reward_scale the term lands at λ·(·) (post-scale units).
                if agent_id in meta_policies and args.meta_action_mode in ("soft", "multiplier_soft"):
                    # soft: rho_factor is the raw pre-multiplier price factor (2·ρ)
                    # multiplier_soft: rho_factor is the post-multiplier effective price factor (2·clip(α·ρ))
                    # In both cases target = α; penalty lands in post-scale units after /reward_scale.
                    rho_factor = 2.0 * float(np.mean(np.array(action_rl[agent_id])[:, 0]))
                    target = meta_multipliers[agent_id]
                    shaped = -args.meta_reg_lambda * (rho_factor - target) ** 2
                    r += model_agents[agent_id].reward_scale * shaped
                    meta_shaping_term[agent_id].append(shaped)
                model_agents[agent_id].rewards.append(r)

        elif env.mode == 3:
            # === BASELINE MODE: No rebalancing, fixed prices ===
            # Use fixed price (scalar = 0.5 for both agents)
            action_rl = {
                0: np.array([0.5] * env.nregion),
                1: np.array([0.5] * env.nregion)
            }

            # Matching step with fixed prices
            _t_start()
            obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
            _t_end('match_step')

            # Track rewards (no rebalancing cost in baseline)
            episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}

            # NO rebalancing step - just update vehicle arrivals from completed passenger trips
            _t_start()
            env.matching_update()
            _t_end('matching_update')

        elif env.mode == 4:
            # === BASELINE MODE 4: Uniform rebalancing, fixed prices ===
            # Use fixed price (scalar = 0.5 for both agents)
            action_rl = {
                0: np.array([0.5] * env.nregion),
                1: np.array([0.5] * env.nregion)
            }

            # Matching step with fixed prices
            _t_start()
            obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
            _t_end('match_step')

            # Track rewards
            episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}

            # UNIFORM rebalancing: distribute vehicles equally across all regions
            desiredAcc = {}
            for a in [0, 1]:
                # Calculate total available vehicles for this agent
                current_total = dictsum(env.agent_acc[a], env.time + 1)
                base_per_region = current_total // env.nregion
                remainder = current_total % env.nregion
                # Distribute uniformly with remainder going to first regions
                desiredAcc[a] = {
                    env.region[i]: base_per_region + (1 if i < remainder else 0)
                    for i in range(env.nregion)
                }

            # Compute rebalancing flows for both agents sequentially
            _t_start()
            rebAction = {a: solveRebFlow(env, desiredAcc[a], a) for a in [0, 1]}
            _t_end('solve_reb')

            _t_start()
            new_obs, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
            _t_end('reb_step')
            episode_reward = {a: episode_reward[a] + rebreward[a] for a in [0, 1]}

        else:
            raise ValueError("Only mode 0, 1, 2, 3, and 4 are allowed")

        # Track agent-specific metrics
        for a in [0, 1]:
                episode_served_demand[a] += info[a]["served_demand"]
                episode_unserved_demand[a] += info[a]["unserved_demand"]
                episode_rebalancing_cost[a] += info[a]["rebalancing_cost"]
                episode_total_revenue[a] += info[a]["revenue"]
                episode_total_operating_cost[a] += info[a]["operating_cost"]
                episode_waiting[a] += info[a]["served_waiting"]
                # Track profitability metrics
                episode_true_profit[a] += info[a].get("true_profit", 0)
                episode_adjusted_profit[a] += info[a].get("adjusted_profit", 0)
                episode_unprofitable_trips[a] += info[a].get("unprofitable_trips", 0)
                # Per-day demand for brand momentum update
                day_served[a] += info[a]["served_demand"]

        # Track system-level metrics (not agent-specific)
        episode_rejected_demand += system_info["rejected_demand"]
        episode_total_demand += system_info["total_demand"]
        episode_rejection_rates.append(system_info["rejection_rate"])
        day_total_demand += system_info["total_demand"]

        if accumulator is not None:
            _t_start()
            accumulator.update(info, system_info, submitted_prices)
            _t_end('accumulator')

        step += 1

    _t_start()
    env.update_brand_momentum(served_counts=day_served, total_demand=day_total_demand)
    _t_end('update_bm')

    if args.parallel_days:
        # Picard path: capture day state; meta buffers filled by commit() after convergence.
        meta_reward = {
            a: (accumulator.profit[a] - accumulator.reb_cost[a]) / args.reward_scalar
            for a in [0, 1]
        }
        _t_start()
        picard_solver.record_day(i_day, env, accumulator, meta_reward)
        _t_end('record_day')
    else:
        if meta_policies:
            for _a in meta_policies:
                meta_policies[_a].store_reward(
                    (accumulator.profit[_a] - accumulator.reb_cost[_a]) / args.reward_scalar
                )
        if accumulator is not None:
            accumulator.momentum_snapshot = dict(env.brand_momentum)
            meta_obs = {
                a: accumulator.daily_state(a, i_day + 1, args.num_days, args.reward_scalar)
                for a in [0, 1]
            }

            # Day-level WandB log (uses meta/global_day as x-axis in custom panels)
            if not args.test:
                global_day = i_episode * args.num_days + i_day
                day_log = {
                    "meta/global_day": global_day,
                    "meta/episode": i_episode,
                    "meta/day_in_episode": i_day,
                    "day/agent0_daily_profit": accumulator.profit[0] / args.reward_scalar,
                    "day/agent1_daily_profit": accumulator.profit[1] / args.reward_scalar,
                    "day/agent0_meta_reward": (accumulator.profit[0] - accumulator.reb_cost[0]) / args.reward_scalar,
                    "day/agent1_meta_reward": (accumulator.profit[1] - accumulator.reb_cost[1]) / args.reward_scalar,
                    "day/agent0_brand_momentum": env.brand_momentum[0],
                    "day/agent1_brand_momentum": env.brand_momentum[1],
                    "day/agent0_step_reward_sum": (episode_reward[0] - day_reward_start[0]) / args.reward_scalar,
                    "day/agent1_step_reward_sum": (episode_reward[1] - day_reward_start[1]) / args.reward_scalar,
                }
                if accumulator.total_demand > 0:
                    day_log["day/agent0_market_share"] = accumulator.served[0] / accumulator.total_demand
                    day_log["day/agent1_market_share"] = accumulator.served[1] / accumulator.total_demand
                if accumulator._price_steps > 0:
                    day_log["day/agent0_avg_price"] = accumulator._price_sum[0] / accumulator._price_steps
                    day_log["day/agent1_avg_price"] = accumulator._price_sum[1] / accumulator._price_steps
                for _a in meta_policies:
                    day_log[f"day/agent{_a}_meta_multiplier"] = float(meta_multipliers[_a])
                    if len(day_price_raw[_a]) > 0:
                        day_log[f"day/agent{_a}_avg_price_raw"] = float(np.mean(day_price_raw[_a]))
                # Queue for the caller to drain (sequential or parallel path).
                # Parallel workers can't call wandb.log directly — children
                # don't have the parent's wandb run context.
                ctx.day_logs.append(day_log)
    # === END verbatim copy ===

    # --- Write back the seven reassigned values ---
    ctx.obs = obs
    ctx.action_rl = action_rl
    ctx.meta_obs = meta_obs
    ctx.meta_multipliers = meta_multipliers
    ctx.episode_reward = episode_reward
    ctx.episode_rejected_demand = episode_rejected_demand
    ctx.episode_total_demand = episode_total_demand
