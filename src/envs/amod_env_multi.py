from collections import defaultdict
from copy import deepcopy
import json
import os
import random

import networkx as nx
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from src.envs.structures import generate_passenger

class AMoD:
    def __init__(self, scenario, mode, beta, jitter, max_wait, choice_price_mult, seed, fix_agent, choice_intercept, wage, use_dynamic_wage_man_south=False, od_price_actions=False, brand_momentum_lambda=0.9, brand_momentum_gamma=0.0):
        self.scenario = deepcopy(scenario)
        self.od_price_actions = od_price_actions
        self.mode = mode
        self.jitter = jitter
        self.max_wait = max_wait
        
        # Setting which agent to fix (0=fix agent 0, 1=fix agent 1, 2=no fixing)
        self.fix_agent = fix_agent
        
        # Choice model intercept (utility of using ridehailing service)
        self.choice_intercept = choice_intercept
        
        # Wage parameter for choice model (city-wide average)
        self.wage = wage
        
        # Region-specific wage distributions for NYC Manhattan South
        self.use_dynamic_wage_man_south = use_dynamic_wage_man_south
        self.wage_distributions = None
        self.city_avg_wage = wage  # Default to uniform wage
        
        if use_dynamic_wage_man_south:
            # Load wage distribution from JSON file
            wage_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'manhattan_wage_data.json')
            if os.path.exists(wage_data_path):
                with open(wage_data_path, 'r') as f:
                    wage_data = json.load(f)
                
                # Store wage distributions (convert probabilities from percentages to probabilities)
                self.wage_distributions = {}
                for region_str, dist_data in wage_data['wage_distribution'].items():
                    region_id = int(region_str)
                    wages = np.array(dist_data['hourly_wages'])
                    probs = np.array(dist_data['probabilities']) / 100.0  # Convert to probabilities
                    
                    # Normalize to ensure exact sum of 1.0 (handle rounding errors)
                    probs = probs / probs.sum()
                    
                    self.wage_distributions[region_id] = {
                        'wages': wages,
                        'probabilities': probs
                    }
                
                # Use the city-wide average wage already provided
                self.city_avg_wage = wage
                
                print(f"\n{'='*80}")
                print(f"DYNAMIC WAGE MODE ENABLED (NYC Manhattan South)")
                print(f"  - Loaded wage distributions for {len(self.wage_distributions)} regions")
                print(f"  - City-wide average wage: ${self.city_avg_wage:.2f}/hour")
                print(f"  - Income effect will vary per passenger: city_avg / passenger_wage")
                print(f"{'='*80}\n")
            else:
                print(f"\n{'='*80}")
                print(f"WARNING: use_dynamic_wage_man_south=True but wage data file not found:")
                print(f"  {wage_data_path}")
                print(f"Falling back to uniform wage: ${wage:.2f}/hour")
                print(f"{'='*80}\n")
                self.use_dynamic_wage_man_south = False
        
        # Track unprofitable trips for logging
        self.agent_unprofitable_trips = {agent_id: 0 for agent_id in [0, 1]}

        self.G = scenario.G

        self.demandTime = self.scenario.demandTime
        self.rebTime = self.scenario.rebTime

        self.time = 0
        self.tf = scenario.tf
        self.tstep = scenario.tstep

        self.agents = [0, 1]

        self.region = list(self.G)

        self.demand = defaultdict(dict)

        self.agent_passenger = {agent_id: dict() for agent_id in self.agents}
        self.agent_queue = {agent_id: defaultdict(list) for agent_id in self.agents}

        for agent_id in self.agents:
            for i in self.region:
                self.agent_passenger[agent_id][i] = defaultdict(list)

        self.agent_price = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_arrivals = {agent_id: 0 for agent_id in self.agents}

        # Initialize demand and pricing from scenario data
        # For each trip attribute (origin i, destination j, time t, demand d, base price p):
        # - Store O-D specific demand for matching
        # - Set initial prices for both agents (they can adjust independently later)
        # - Accumulate departure demand (total passengers leaving region i at time t)
        # - Accumulate arrival demand (total passengers arriving at region i at time t+travel_time)
        for i, j, t, d, p in scenario.tripAttr:
            self.demand[i, j][t] = d
            for agent_id in self.agents:
                self.agent_price[agent_id][i, j][t] = p

        self.agent_acc = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_dacc = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_rebFlow = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_rebFlow_ori = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_paxFlow = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_paxWait = {agent_id: defaultdict(list) for agent_id in self.agents}

        # Initialize graph structure and flow tracking
        self.edges = []
        self.nregion = len(scenario.G)

        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))

        self.nedge = [len(self.G.out_edges(n))+1 for n in self.region]

        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.rebTime[i, j][self.time]
            for agent_id in self.agents:
                self.agent_rebFlow[agent_id][i, j] = defaultdict(int)
                self.agent_rebFlow_ori[agent_id][i, j] = defaultdict(int)

        for i, j in self.demand:
            for agent_id in self.agents:
                self.agent_paxFlow[agent_id][i, j] = defaultdict(int)
                self.agent_paxWait[agent_id][i, j] = []

        # Initialize vehicle counts for each agent and region
        # Use agent-specific vehicle distributions from scenario (already split and distributed)
        # Store initial distribution for fixed agent rebalancing
        self.agent_initial_acc = {agent_id: {} for agent_id in self.agents}
        for agent_id in self.agents:
            for n in self.region:
                # Use agent-specific accInit values from scenario
                acc_key = f'accInit_agent{agent_id}'
                initial_count = self.G.nodes[n][acc_key]
                self.agent_acc[agent_id][n][0] = initial_count
                self.agent_initial_acc[agent_id][n] = initial_count  # Store for fixed agent
                self.agent_dacc[agent_id][n] = defaultdict(int)


        # scenario.tstep: number of steps as one timestep
        self.beta = beta * scenario.tstep # Cost for rebalancing per time unit in simulation time

        self.agent_demand = {agent_id: defaultdict(dict) for agent_id in self.agents}
        
        for agent_id in self.agents:
            for i, j in self.demand:
                self.agent_demand[agent_id][i, j] = defaultdict(int)

        self.agent_servedDemand = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_unservedDemand = {agent_id: defaultdict(dict) for agent_id in self.agents}


        for agent_id in self.agents:
            for i, j in self.demand:
                self.agent_servedDemand[agent_id][i, j] = defaultdict(int)
                self.agent_unservedDemand[agent_id][i, j] = defaultdict(int)

        self.N = len(self.region)

        self.agent_info = {agent_id: dict.fromkeys(['revenue', 'served_demand', 'unserved_demand',
                                    'rebalancing_cost', 'operating_cost', 'served_waiting', 
                                    'true_profit', 'adjusted_profit'], 0) 
                    for agent_id in self.agents}
        
        self.system_info = dict.fromkeys(['rejected_demand', 'total_demand', 'rejection_rate'], 0)

        self.system_wage_samples = []

        self.ext_reward_agents = {a: np.zeros(self.nregion) for a in [0, 1]}

        self.agent_obs = {agent_id: (self.agent_acc[agent_id], self.time, 
                            self.agent_dacc[agent_id], self.demand) 
            for agent_id in self.agents}
    
        self.choice_price_mult = choice_price_mult

        self.seed = seed
        self._shuffle_rng = random.Random(seed)
        self.track_trip_assignments = False
        self._match_cache = None

        self.bm_lambda = brand_momentum_lambda
        self.bm_gamma = brand_momentum_gamma
        self.brand_momentum = {0: 0.5, 1: 0.5}

        # Trip assignment tracking: stores detailed data for each trip
        self.trip_assignments = []

    def _build_match_cache(self):
        """Cache per-edge scenario arrays + per-edge inner-dict refs.

        Built lazily and invalidated by reset(). Holds references to the
        currently-live agent_price / agent_demand / demand inner dicts so
        the per-edge writeback loop can avoid tuple-key dict lookups.
        """
        edges = list(self.G.edges)
        n_edges = len(edges)
        tf = self.tf + 1
        region = list(self.region)
        region_to_idx = {r: i for i, r in enumerate(region)}
        n_regions = len(region)

        baseline_price = np.zeros((n_edges, tf))
        travel_time_arr = np.zeros((n_edges, tf))
        demand_orig = np.zeros((n_edges, tf), dtype=np.int64)
        edge_n_idx = np.zeros(n_edges, dtype=np.int64)
        edge_j = np.zeros(n_edges, dtype=np.int64)
        for k, (n, j) in enumerate(edges):
            edge_n_idx[k] = region_to_idx[n]
            edge_j[k] = j
            ap = self.agent_price[0][n, j]
            dt = self.demandTime[n, j]
            dm = self.demand[n, j]
            for tt in range(tf):
                if tt in ap:
                    baseline_price[k, tt] = ap[tt]
                if tt in dt:
                    travel_time_arr[k, tt] = dt[tt]
                if tt in dm:
                    demand_orig[k, tt] = dm[tt]

        ap0_per_edge = [self.agent_price[0][e] for e in edges]
        ap1_per_edge = [self.agent_price[1][e] for e in edges]
        ad0_per_edge = [self.agent_demand[0][e] for e in edges]
        ad1_per_edge = [self.agent_demand[1][e] for e in edges]
        demand_per_edge = [self.demand[e] for e in edges]

        edges_idx_by_origin = [[] for _ in range(n_regions)]
        for k in range(n_edges):
            edges_idx_by_origin[edge_n_idx[k]].append(k)
        edges_idx_by_origin = [np.array(lst, dtype=np.int64) for lst in edges_idx_by_origin]

        # --- Reb-step-specific cache (indexed by self.edges order, which may
        # differ from list(self.G.edges) order). rebAction arrays produced by
        # solveRebFlow_* are indexed by self.edges. ---
        reb_edges = list(self.edges)
        n_reb = len(reb_edges)
        reb_i_idx = np.zeros(n_reb, dtype=np.int64)
        reb_j_idx = np.zeros(n_reb, dtype=np.int64)
        reb_time = np.zeros((n_reb, tf), dtype=np.int64)
        for k, (i, j) in enumerate(reb_edges):
            reb_i_idx[k] = region_to_idx[i]
            reb_j_idx[k] = region_to_idx[j]
            rt_inner = self.rebTime[i, j]
            for tt in range(tf):
                if tt in rt_inner:
                    reb_time[k, tt] = rt_inner[tt]

        # Per-edge inner-dict refs in reb_edges order.
        reb_flow_pe = [
            [self.agent_rebFlow[a][e] for e in reb_edges] for a in self.agents
        ]
        reb_flow_ori_pe = [
            [self.agent_rebFlow_ori[a][e] for e in reb_edges] for a in self.agents
        ]
        # paxFlow only has entries for demand edges (no self-loops). Use .get
        # so a missing key doesn't auto-create on the defaultdict.
        pax_flow_pe_reb = [
            [self.agent_paxFlow[a].get(e, None) for e in reb_edges]
            for a in self.agents
        ]
        # Per-region refs for agent_acc / agent_dacc (region-label key).
        acc_per_region = [
            [self.agent_acc[a][r] for r in region] for a in self.agents
        ]
        dacc_per_region = [
            [self.agent_dacc[a][r] for r in region] for a in self.agents
        ]
        # NetworkX edge dict refs for the post-step G.edges['time'] update.
        g_edge_dicts = [self.G.edges[e] for e in edges]

        self._match_cache = {
            "edges": edges,
            "n_edges": n_edges,
            "region": region,
            "region_to_idx": region_to_idx,
            "n_regions": n_regions,
            "edge_n_idx": edge_n_idx,
            "edge_j": edge_j,
            "edges_idx_by_origin": edges_idx_by_origin,
            "baseline_price": baseline_price,
            "travel_time": travel_time_arr,
            "demand_orig": demand_orig,
            "ap0_per_edge": ap0_per_edge,
            "ap1_per_edge": ap1_per_edge,
            "ad0_per_edge": ad0_per_edge,
            "ad1_per_edge": ad1_per_edge,
            "demand_per_edge": demand_per_edge,
            # reb-step cache
            "reb_edges": reb_edges,
            "n_reb": n_reb,
            "reb_i_idx": reb_i_idx,
            "reb_j_idx": reb_j_idx,
            "reb_time": reb_time,
            "reb_flow_pe": reb_flow_pe,
            "reb_flow_ori_pe": reb_flow_ori_pe,
            "pax_flow_pe_reb": pax_flow_pe_reb,
            "acc_per_region": acc_per_region,
            "dacc_per_region": dacc_per_region,
            "g_edge_dicts": g_edge_dicts,
        }

    def match_step_simple(self, price = None):
        t = self.time
        paxreward = {0: 0, 1: 0}

        for agent_id in self.agents:
            self.agent_unprofitable_trips[agent_id] = 0
        for agent_id in self.agents:
            for key in self.agent_info[agent_id]:
                self.agent_info[agent_id][key] = 0
        for key in self.system_info:
            self.system_info[key] = 0
        self.system_wage_samples = []

        # Price scalar extraction (unchanged from prior code).
        price_scalars = None
        if self.mode != 0 and price is not None:
            total_price_sum = sum(np.sum(price[a]) for a in self.agents)
            if total_price_sum != 0:
                price_scalars = {}
                for agent_id in self.agents:
                    if self.fix_agent == agent_id:
                        if self.od_price_actions:
                            price_scalars[agent_id] = {(n, j): 0.5 for n in self.region for j in self.G[n]}
                        else:
                            price_scalars[agent_id] = {n: 0.5 for n in self.region}
                    else:
                        price_scalars[agent_id] = {}
                        if self.od_price_actions:
                            for n in self.region:
                                for j in self.G[n]:
                                    price_scalars[agent_id][(n, j)] = float(price[agent_id][n][j])
                        else:
                            for n in self.region:
                                scalar = price[agent_id][n]
                                if isinstance(scalar, (list, np.ndarray)):
                                    scalar = scalar[0]
                                price_scalars[agent_id][n] = scalar

        if self._match_cache is None:
            self._build_match_cache()
        cache = self._match_cache
        edges = cache["edges"]
        n_edges = cache["n_edges"]
        edge_n_idx = cache["edge_n_idx"]
        edge_j = cache["edge_j"]
        region = cache["region"]
        edges_idx_by_origin = cache["edges_idx_by_origin"]

        # --- Vectorized price application across all edges ---
        # baseline_price is the scenario reference; agent_price for both
        # agents is initialized to the same reference value per (i,j,t).
        baseline_p = cache["baseline_price"][:, t]
        travel_times = cache["travel_time"][:, t]
        demands = cache["demand_orig"][:, t].copy()

        if price_scalars is not None:
            if self.od_price_actions:
                scalars_0 = np.array([price_scalars[0][e] for e in edges])
                scalars_1 = np.array([price_scalars[1][e] for e in edges])
            else:
                sper0 = np.array([price_scalars[0][r] for r in region])
                sper1 = np.array([price_scalars[1][r] for r in region])
                scalars_0 = sper0[edge_n_idx]
                scalars_1 = sper1[edge_n_idx]
            pr0 = 2.0 * baseline_p * scalars_0
            pr1 = 2.0 * baseline_p * scalars_1
            pr0 = np.where(pr0 <= 1e-6, self.jitter, pr0)
            pr1 = np.where(pr1 <= 1e-6, self.jitter, pr1)
        else:
            pr0 = baseline_p
            pr1 = baseline_p

        # --- Choice model ---
        travel_times_in_hours = travel_times / 60.0
        # Buffers populated by either path.
        d0_arr = np.zeros(n_edges, dtype=np.int64)
        d1_arr = np.zeros(n_edges, dtype=np.int64)
        dr_arr = np.zeros(n_edges, dtype=np.int64)
        U_0_log = None  # populated only if track_trip_assignments
        U_1_log = None
        prob_log = None

        if self.use_dynamic_wage_man_south and self.wage_distributions is not None:
            # Per-passenger wage sampling — keep the per-edge loop but feed it
            # from the cached arrays.
            if self.track_trip_assignments:
                U_0_log = np.zeros(n_edges)
                U_1_log = np.zeros(n_edges)
                prob_log = np.zeros((n_edges, 3))
            for k in range(n_edges):
                d_k = int(demands[k])
                if d_k == 0:
                    continue
                n_lab = edges[k][0]
                tt_h = travel_times_in_hours[k]
                if n_lab in self.wage_distributions:
                    dist = self.wage_distributions[n_lab]
                    pw = np.random.choice(dist['wages'], size=d_k, p=dist['probabilities'])
                else:
                    pw = np.full(d_k, self.wage)
                self.system_wage_samples.extend(pw.tolist())
                ie = self.city_avg_wage / pw
                U0b = self.choice_intercept - 0.71 * pw * tt_h - ie * self.choice_price_mult * pr0[k] + self.bm_gamma * self.brand_momentum[0]
                U1b = self.choice_intercept - 0.71 * pw * tt_h - ie * self.choice_price_mult * pr1[k] + self.bm_gamma * self.brand_momentum[1]
                Urb = np.zeros(d_k)
                ee = np.column_stack([np.exp(U0b), np.exp(U1b), np.exp(Urb)])
                pp = ee / ee.sum(axis=1, keepdims=True)
                rv = np.random.rand(d_k)
                cp = np.cumsum(pp, axis=1)
                ch = np.sum(rv[:, None] > cp, axis=1)
                d0_arr[k] = int(np.sum(ch == 0))
                d1_arr[k] = int(np.sum(ch == 1))
                dr_arr[k] = int(np.sum(ch == 2))
                if self.track_trip_assignments:
                    U_0_log[k] = float(np.mean(U0b))
                    U_1_log[k] = float(np.mean(U1b))
                    prob_log[k] = pp.mean(axis=0)
        else:
            # Uniform wage: fully vectorized MNL + Multinomial-as-Binomial chain.
            base_U = self.choice_intercept - 0.71 * self.wage * travel_times_in_hours
            U_0 = base_U - self.choice_price_mult * pr0 + self.bm_gamma * self.brand_momentum[0]
            U_1 = base_U - self.choice_price_mult * pr1 + self.bm_gamma * self.brand_momentum[1]
            exp_U_0 = np.exp(U_0)
            exp_U_1 = np.exp(U_1)
            total_exp = exp_U_0 + exp_U_1 + 1.0  # exp(0) for reject
            p0_arr = exp_U_0 / total_exp
            p1_arr = exp_U_1 / total_exp
            p_r_arr = 1.0 / total_exp

            d0_arr = np.random.binomial(demands, p0_arr).astype(np.int64)
            rem = demands - d0_arr
            denom = p1_arr + p_r_arr
            # Safe divide: where denom==0 nothing to split anyway.
            p_split = np.where(denom > 0, p1_arr / np.where(denom > 0, denom, 1.0), 0.0)
            d1_arr = np.random.binomial(rem, p_split).astype(np.int64)
            dr_arr = (rem - d1_arr).astype(np.int64)

            total_for_wages = int(demands.sum())
            if total_for_wages > 0:
                self.system_wage_samples.extend([self.wage] * total_for_wages)

            if self.track_trip_assignments:
                U_0_log = U_0
                U_1_log = U_1
                prob_log = np.column_stack([p0_arr, p1_arr, p_r_arr])

        # --- Per-edge writebacks via cached refs ---
        ap0_per_edge = cache["ap0_per_edge"]
        ap1_per_edge = cache["ap1_per_edge"]
        ad0_per_edge = cache["ad0_per_edge"]
        ad1_per_edge = cache["ad1_per_edge"]
        demand_per_edge = cache["demand_per_edge"]
        pr0_list = pr0.tolist()
        pr1_list = pr1.tolist()
        d0_list = d0_arr.tolist()
        d1_list = d1_arr.tolist()
        dr_list = dr_arr.tolist()
        demands_list = demands.tolist()
        if price_scalars is not None:
            for k in range(n_edges):
                ap0_per_edge[k][t] = pr0_list[k]
                ap1_per_edge[k][t] = pr1_list[k]
        for k in range(n_edges):
            d0_k = d0_list[k]
            d1_k = d1_list[k]
            ad0_per_edge[k][t] += d0_k
            ad1_per_edge[k][t] += d1_k
            demand_per_edge[k][t] = d0_k + d1_k

        total_original_demand = int(demands.sum())
        total_rejected_demand = int(dr_arr.sum())

        if self.track_trip_assignments:
            travel_times_list = travel_times.tolist()
            for k in range(n_edges):
                d_k = demands_list[k]
                if d_k == 0:
                    continue
                n_lab, j_lab = edges[k]
                self.trip_assignments.append({
                    'time': t,
                    'origin': n_lab,
                    'destination': j_lab,
                    'travel_time': travel_times_list[k],
                    'price_agent0': pr0_list[k],
                    'price_agent1': pr1_list[k],
                    'utility_agent0': float(U_0_log[k]),
                    'utility_agent1': float(U_1_log[k]),
                    'utility_reject': 0,
                    'prob_agent0': float(prob_log[k, 0]),
                    'prob_agent1': float(prob_log[k, 1]),
                    'prob_reject': float(prob_log[k, 2]),
                    'demand_agent0': d0_list[k],
                    'demand_agent1': d1_list[k],
                    'demand_rejected': dr_list[k],
                    'total_demand': d_k,
                })

        # --- Build new passenger lists per (agent, origin) ---
        # Tuple format: (destination, price, wait_time=0).
        arr0 = self.agent_arrivals[0]
        arr1 = self.agent_arrivals[1]
        for n_idx, n_label in enumerate(region):
            e_idx_arr = edges_idx_by_origin[n_idx]
            if e_idx_arr.size == 0:
                self.agent_passenger[0][n_label][t] = []
                self.agent_passenger[1][n_label][t] = []
                continue
            dsts = edge_j[e_idx_arr]
            c0 = d0_arr[e_idx_arr]
            c1 = d1_arr[e_idx_arr]
            p0 = pr0[e_idx_arr]
            p1 = pr1[e_idx_arr]
            if c0.sum() > 0:
                dsts0 = np.repeat(dsts, c0).tolist()
                prices0 = np.repeat(p0, c0).tolist()
                pax0 = list(zip(dsts0, prices0, [0] * len(dsts0)))
                arr0 += len(pax0)
            else:
                pax0 = []
            if c1.sum() > 0:
                dsts1 = np.repeat(dsts, c1).tolist()
                prices1 = np.repeat(p1, c1).tolist()
                pax1 = list(zip(dsts1, prices1, [0] * len(dsts1)))
                arr1 += len(pax1)
            else:
                pax1 = []
            self.agent_passenger[0][n_label][t] = pax0
            self.agent_passenger[1][n_label][t] = pax1
        self.agent_arrivals[0] = arr0
        self.agent_arrivals[1] = arr1

        # --- Per-region match loop (tuple-based; shuffle once per region) ---
        max_wait = self.max_wait
        beta = self.beta
        for n in self.region:
            self._shuffle_rng.shuffle(self.agent_passenger[0][n][t])
            self._shuffle_rng.shuffle(self.agent_passenger[1][n][t])

            for agent_id in [0, 1]:
                accCurrent = self.agent_acc[agent_id][n][t]
                new_enterq = self.agent_passenger[agent_id][n][t]
                queueCurrent = self.agent_queue[agent_id][n] + new_enterq
                self.agent_queue[agent_id][n] = queueCurrent
                num_q = len(queueCurrent)
                num_served = num_q if accCurrent >= num_q else accCurrent

                ainfo = self.agent_info[agent_id]
                ext_reward_n = self.ext_reward_agents[agent_id]
                paxFlow_a = self.agent_paxFlow[agent_id]
                paxWait_a = self.agent_paxWait[agent_id]
                dacc_a = self.agent_dacc[agent_id]
                servedDemand_a = self.agent_servedDemand[agent_id]
                unservedDemand_a = self.agent_unservedDemand[agent_id]
                demandTime_t = self.demandTime
                sum_revenue = 0.0
                sum_op_cost = 0.0
                sum_wait = 0

                for k in range(num_served):
                    dst, price_p, wt = queueCurrent[k]
                    tt = demandTime_t[n, dst][t]
                    arr_t = t + tt
                    paxFlow_a[n, dst][arr_t] += 1
                    paxWait_a[n, dst].append(wt)
                    dacc_a[dst][arr_t] += 1
                    servedDemand_a[n, dst][t] += 1
                    trip_cost = tt * beta
                    sum_revenue += price_p
                    sum_op_cost += trip_cost
                    sum_wait += wt
                    ext_reward_n[n] += trip_cost
                paxreward[agent_id] += sum_revenue - sum_op_cost
                ainfo['revenue'] += sum_revenue
                ainfo['operating_cost'] += sum_op_cost
                ainfo['served_waiting'] += sum_wait
                ainfo['true_profit'] += sum_revenue - sum_op_cost
                ainfo['served_demand'] += num_served

                new_queue = []
                unserved_count = 0
                for k in range(num_served, num_q):
                    dst, price_p, wt = queueCurrent[k]
                    new_wt = wt + 1
                    if new_wt >= max_wait:
                        unservedDemand_a[n, dst][t] += 1
                        unserved_count += 1
                    else:
                        new_queue.append((dst, price_p, new_wt))
                ainfo['unserved_demand'] += unserved_count
                self.agent_queue[agent_id][n] = new_queue
                self.agent_acc[agent_id][n][t + 1] = accCurrent - num_served

        done = (self.tf == t+1)
        ext_done = [done]*self.nregion

        self.obs = {
            0: (self.agent_acc[0], self.time, self.agent_dacc[0], self.agent_demand[0]),
            1: (self.agent_acc[1], self.time, self.agent_dacc[1], self.agent_demand[1])
        }

        # Update system-level info
        self.system_info['rejected_demand'] = total_rejected_demand
        self.system_info['total_demand'] = total_original_demand
        self.system_info['rejection_rate'] = (
            total_rejected_demand / total_original_demand if total_original_demand > 0 else 0
        )
        
        # Calculate average wage (always track, regardless of dynamic wage flag)
        if len(self.system_wage_samples) > 0:
            self.system_info['avg_wage'] = np.mean(self.system_wage_samples)
        else:
            self.system_info['avg_wage'] = None

        # Add unprofitable trips count to agent info
        for agent_id in [0, 1]:
            self.agent_info[agent_id]['unprofitable_trips'] = self.agent_unprofitable_trips[agent_id]

        return self.obs, paxreward, done, self.agent_info, self.system_info, self.ext_reward_agents, ext_done

    def matching_update(self):
        """Update properties if there is no rebalancing after matching"""
        t = self.time
        # Update acc. Assuming arriving vehicle will only be availbe for the next timestamp.
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            for agent_id in [0, 1]:
                if (i, j) in self.agent_paxFlow[agent_id] and t in self.agent_paxFlow[agent_id][i, j]:
                    self.agent_acc[agent_id][j][t+1] += self.agent_paxFlow[agent_id][i, j][t]
        
        # For fixed agents, reset vehicle distribution to initial state
        if self.fix_agent in [0, 1]:
            fixed_agent_id = self.fix_agent
            for n in self.region:
                self.agent_acc[fixed_agent_id][n][t+1] = self.agent_initial_acc[fixed_agent_id][n]
        
        self.time += 1

    def reb_step(self, rebAction_agents):
        t = self.time
        rebreward = {0: 0, 1: 0}
        nregion = self.nregion
        self.ext_reward_agents = {a: np.zeros(nregion) for a in [0, 1]}
        for agent_id in [0, 1]:
            self.agent_info[agent_id]['rebalancing_cost'] = 0

        if self._match_cache is None:
            self._build_match_cache()
        cache = self._match_cache
        n_reb = cache["n_reb"]
        reb_i_idx = cache["reb_i_idx"]
        reb_j_idx = cache["reb_j_idx"]
        reb_time_at_t = cache["reb_time"][:, t]  # int array per edge
        beta = self.beta

        for agent_id in [0, 1]:
            rebAction = rebAction_agents[agent_id]
            rebAction_arr = np.asarray(rebAction, dtype=np.int64)

            # --- Vectorized cost ---
            cost_arr = reb_time_at_t * beta * rebAction_arr
            total_cost = float(cost_arr.sum())
            rebreward[agent_id] -= total_cost
            self.agent_info[agent_id]['rebalancing_cost'] += total_cost
            # ext_reward[a][i] -= cost: scatter-sub by origin idx.
            np.subtract.at(self.ext_reward_agents[agent_id], reb_i_idx, cost_arr)

            # --- Per-edge dict writes (rebFlow / rebFlow_ori) ---
            # Skip zero values: writes to defaultdict(int) of 0 are observationally
            # no-ops downstream (the inner dict's missing-key default is 0, and
            # nothing checks the presence of a key as a side-channel for these).
            rf_pe = cache["reb_flow_pe"][agent_id]
            rfo_pe = cache["reb_flow_ori_pe"][agent_id]
            reb_list = rebAction_arr.tolist()
            reb_time_list = reb_time_at_t.tolist()
            for k in range(n_reb):
                v = reb_list[k]
                if v == 0:
                    continue
                rt = reb_time_list[k]
                rf_pe[k][t + rt] = v
                rfo_pe[k][t] = v

            # --- agent_acc[a][i][t+1] -= outflow (grouped by origin) ---
            acc_pr = cache["acc_per_region"][agent_id]
            outflow_per_origin = np.bincount(
                reb_i_idx, weights=rebAction_arr, minlength=nregion
            ).astype(np.int64)
            for n_idx in range(nregion):
                v = int(outflow_per_origin[n_idx])
                if v != 0:
                    acc_pr[n_idx][t + 1] -= v

            # --- agent_dacc[a][j][t + reb_time] += rebAction (varying arr_t) ---
            dacc_pr = cache["dacc_per_region"][agent_id]
            j_idx_list = reb_j_idx.tolist()
            for k in range(n_reb):
                v = reb_list[k]
                if v == 0:
                    continue
                arr_t = t + reb_time_list[k]
                dacc_pr[j_idx_list[k]][arr_t] += v

        # --- Second pass: arrivals at j at current t from rebFlow + paxFlow ---
        # For each agent: gather inner-dict values at t (if present), accumulate
        # by destination, scatter-add into agent_acc[a][j][t+1].
        for agent_id in [0, 1]:
            rf_pe = cache["reb_flow_pe"][agent_id]
            pf_pe = cache["pax_flow_pe_reb"][agent_id]
            acc_pr = cache["acc_per_region"][agent_id]
            arrivals_per_dst = np.zeros(nregion, dtype=np.int64)
            for k in range(n_reb):
                inner = rf_pe[k]
                v_reb = inner.get(t, 0)
                inner_p = pf_pe[k]
                v_pax = inner_p.get(t, 0) if inner_p is not None else 0
                total = v_reb + v_pax
                if total != 0:
                    arrivals_per_dst[reb_j_idx[k]] += total
            for n_idx in range(nregion):
                v = int(arrivals_per_dst[n_idx])
                if v != 0:
                    acc_pr[n_idx][t + 1] += v

        # For fixed agents, reset vehicle distribution to initial state
        if self.fix_agent in [0, 1]:
            fixed_agent_id = self.fix_agent
            initial = self.agent_initial_acc[fixed_agent_id]
            acc_pr = cache["acc_per_region"][fixed_agent_id]
            region = cache["region"]
            for n_idx, n_label in enumerate(region):
                acc_pr[n_idx][t + 1] = initial[n_label]

        self.time += 1

        self.obs = {
            0: (self.agent_acc[0], self.time, self.agent_dacc[0], self.agent_demand[0]),
            1: (self.agent_acc[1], self.time, self.agent_dacc[1], self.agent_demand[1])
        }

        # Update G.edges['time'] via cached edge-dict refs (avoids the
        # self.G.edges[i, j]['time'] networkx attribute lookup).
        new_time = self.time
        g_edge_dicts = cache["g_edge_dicts"]
        edges_g = cache["edges"]
        for k, (i, j) in enumerate(edges_g):
            rt_inner = self.rebTime[i, j]
            if new_time in rt_inner:
                g_edge_dicts[k]['time'] = rt_inner[new_time]

        done = (self.tf == t + 1)
        ext_done = [done] * nregion

        return self.obs, rebreward, done, self.agent_info, self.system_info, self.ext_reward_agents, ext_done

    def get_total_vehicles(self, agent_id=None):
        """
        Calculate total number of vehicles in the system at current time for each agent.
        Includes: available vehicles + vehicles with passengers + rebalancing vehicles
        
        Args:
            agent_id: If provided, return total for specific agent. If None, return dict with totals for all agents.
        
        Returns:
            If agent_id is None: dict with {agent_id: total_vehicles}
            If agent_id is provided: int with total vehicles for that agent
        """
        t = self.time
        
        if agent_id is not None:
            # Calculate total vehicles for the agent
            total = 0
            
            # Count available vehicles at all regions for CURRENT time
            for region in self.region:
                # Try current time first, then fallback to t+1
                if t in self.agent_acc[agent_id][region]:
                    total += self.agent_acc[agent_id][region][t]
                elif t+1 in self.agent_acc[agent_id][region]:
                    total += self.agent_acc[agent_id][region][t+1]
            
            # Count vehicles with passengers (all current and future arrivals)
            for (i, j), time_dict in self.agent_paxFlow[agent_id].items():
                for time_step, flow in time_dict.items():
                    if time_step >= t:  # Current and future arrivals (vehicles in transit)
                        total += flow
            
            # Count rebalancing vehicles (all current and future arrivals)
            for (i, j), time_dict in self.agent_rebFlow[agent_id].items():
                for time_step, flow in time_dict.items():
                    if time_step >= t:  # Current and future arrivals (vehicles in transit)
                        total += flow
            
            return total
        else:
            # Calculate totals for all agents
            totals = {}
            for agent_id in self.agents:
                # For fixed agents, just return the total from initial distribution
                if self.fix_agent == agent_id:
                    totals[agent_id] = sum(self.agent_initial_acc[agent_id].values())
                    continue
                
                # Calculate total for active (non-fixed) agent
                total = 0
                
                # Count available vehicles at all regions for CURRENT time
                for region in self.region:
                    # Try current time first, then fallback to t+1
                    if t in self.agent_acc[agent_id][region]:
                        total += self.agent_acc[agent_id][region][t]
                    elif t+1 in self.agent_acc[agent_id][region]:
                        total += self.agent_acc[agent_id][region][t+1]
                
                # Count vehicles with passengers (all current and future arrivals)
                for (i, j), time_dict in self.agent_paxFlow[agent_id].items():
                    for time_step, flow in time_dict.items():
                        if time_step >= t:  # Current and future arrivals (vehicles in transit)
                            total += flow
                
                # Count rebalancing vehicles (all current and future arrivals)
                for (i, j), time_dict in self.agent_rebFlow[agent_id].items():
                    for time_step, flow in time_dict.items():
                        if time_step >= t:  # Current and future arrivals (vehicles in transit)
                            total += flow
                
                totals[agent_id] = total
            
            return totals

    def get_initial_vehicles(self):
        """Get the initial number of vehicles in the system (total across both agents)"""
        return sum(
            self.G.nodes[n]['accInit_agent0'] + self.G.nodes[n]['accInit_agent1']
            for n in self.G.nodes
        )

    def get_trip_assignments(self):
        """Get and clear the trip assignments log"""
        trips = self.trip_assignments.copy()
        self.trip_assignments = []
        return trips
    
    def update_brand_momentum(self, served_counts, total_demand):
        """Update EMA brand momentum at the end of a simulated day.

        Called by the day loop (issue #5). served_counts: {agent_id: int},
        total_demand: int (all passengers that arrived this day, across both operators).
        """
        for agent_id in self.agents:
            s = served_counts[agent_id] / total_demand if total_demand > 0 else 0.0
            self.brand_momentum[agent_id] = (
                self.bm_lambda * self.brand_momentum[agent_id]
                + (1 - self.bm_lambda) * s
            )

    def reset(self):
        """Reset the episode for multi-agent environment"""

        self.brand_momentum = {0: 0.5, 1: 0.5}
        self.trip_assignments = []
        self._match_cache = None
        
        self.agent_acc = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_dacc = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_rebFlow = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_rebFlow_ori = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_paxFlow = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_paxWait = {agent_id: defaultdict(list) for agent_id in self.agents}
        self.agent_passenger = {agent_id: dict() for agent_id in self.agents}
        self.agent_queue = {agent_id: defaultdict(list) for agent_id in self.agents}
        
        for agent_id in self.agents:
            for i in self.region:
                self.agent_passenger[agent_id][i] = defaultdict(list)
        
        self.edges = []
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        
        self.demand = defaultdict(dict)
        self.agent_demand = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_price = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_arrivals = {agent_id: 0 for agent_id in self.agents}
        
        tripAttr = self.scenario.get_random_demand(reset=True)
        self.regionDemand = defaultdict(dict)
        
        for i, j, t, d, p in tripAttr:
            self.demand[i, j][t] = d
            for agent_id in self.agents:
                self.agent_price[agent_id][i, j][t] = p
                if (i, j) not in self.agent_demand[agent_id]:
                    self.agent_demand[agent_id][i, j] = defaultdict(int)
            
            if t not in self.regionDemand[i]:
                self.regionDemand[i][t] = 0
            self.regionDemand[i][t] += d
        
        self.time = 0
        
        for i, j in self.G.edges:
            for agent_id in self.agents:
                self.agent_rebFlow[agent_id][i, j] = defaultdict(int)
                self.agent_rebFlow_ori[agent_id][i, j] = defaultdict(int)
                self.agent_paxFlow[agent_id][i, j] = defaultdict(int)
                self.agent_paxWait[agent_id][i, j] = []
        
        for agent_id in self.agents:
            for n in self.G:
                acc_key = f'accInit_agent{agent_id}'
                self.agent_acc[agent_id][n][0] = self.G.nodes[n][acc_key]
                self.agent_dacc[agent_id][n] = defaultdict(int)
        
        self.agent_servedDemand = {agent_id: defaultdict(dict) for agent_id in self.agents}
        self.agent_unservedDemand = {agent_id: defaultdict(dict) for agent_id in self.agents}
        
        for agent_id in self.agents:
            for i, j in self.demand:
                self.agent_servedDemand[agent_id][i, j] = defaultdict(int)
                self.agent_unservedDemand[agent_id][i, j] = defaultdict(int)
        
        self.agent_info = {agent_id: dict.fromkeys(['revenue', 'served_demand', 'unserved_demand',
                                    'rebalancing_cost', 'operating_cost', 'served_waiting', 
                                    'true_profit', 'adjusted_profit'], 0) 
                    for agent_id in self.agents}
        
        self.system_info = dict.fromkeys(['rejected_demand', 'total_demand', 'rejection_rate'], 0)
        
        self.system_wage_samples = []
        
        self.agent_obs = {agent_id: (self.agent_acc[agent_id], self.time,
                            self.agent_dacc[agent_id], self.demand)
            for agent_id in self.agents}

        return self.agent_obs

    def reset_day(self):
        """Partial reset between days in a multi-day episode.
        Resets vehicle positions, queues, and demand while preserving brand momentum.
        """
        saved_momentum = dict(self.brand_momentum)
        obs = self.reset()
        self.brand_momentum = saved_momentum
        return obs



class Scenario:
    def __init__(self, N1=2, N2=4, tf=60, sd=None, ninit=5, tripAttr=None, demand_input=None, demand_ratio=None, supply_ratio=1,
                 trip_length_preference=0.25, grid_travel_time=1, fix_price=True, alpha=0.0, json_file=None, json_hr=19, json_tstep=3, varying_time=False, json_regions=None, impute=False, agent0_vehicle_ratio=0.5, total_vehicles=None):
        # trip_length_preference: positive - more shorter trips, negative - more longer trips
        # grid_travel_time: travel time between grids
        # demand_input： list - total demand out of each region,
        #          float/int - total demand out of each region satisfies uniform distribution on [0, demand_input]
        #          dict/defaultdict - total demand between pairs of regions
        # demand_input will be converted to a variable static_demand to represent the demand between each pair of nodes
        # static_demand will then be sampled according to a Poisson distributionjson_tstep
        # alpha: parameter for uniform distribution of demand levels - [1-alpha, 1+alpha] * demand_input
        self.sd = sd
        self.agent0_vehicle_ratio = agent0_vehicle_ratio
        if sd != None:
            np.random.seed(self.sd)
        if json_file == None:
            self.varying_time = varying_time
            self.is_json = False
            self.alpha = alpha
            self.trip_length_preference = trip_length_preference
            self.grid_travel_time = grid_travel_time
            self.demand_input = demand_input
            self.fix_price = fix_price
            self.N1 = N1
            self.N2 = N2
            self.G = nx.complete_graph(N1*N2)
            self.G = self.G.to_directed()
            # Add self-loops to the graph for within-region trips
            self.G.add_edges_from([(i, i) for i in self.G.nodes])
           
            # Self-loops are now part of G.edges, no need to add them separately
            self.edges = list(self.G.edges)
            self.tstep = json_tstep
            for i, j in self.edges:
                for t in range(tf*2):
                    self.demandTime[i, j][t] = (
                        (abs(i//N1-j//N1) + abs(i % N1-j % N1))*grid_travel_time)
                    self.rebTime[i, j][t] = (
                        (abs(i//N1-j//N1) + abs(i % N1-j % N1))*grid_travel_time)

            # Total fleet = ninit vehicles per node
            total_fleet = ninit * len(self.G.nodes)
            
            # Split fleet between two agents using agent0_vehicle_ratio
            fleet_agent0 = int(total_fleet * self.agent0_vehicle_ratio)
            fleet_agent1 = total_fleet - fleet_agent0
            
            # Create list of nodes and shuffle for random remainder assignment
            nodes_list = list(self.G.nodes)
            random.seed(sd)
            random.shuffle(nodes_list)
            num_nodes = len(nodes_list)
            
            # Distribute agent 0's fleet
            base_vehicles_agent0 = fleet_agent0 // num_nodes
            remainder_agent0 = fleet_agent0 % num_nodes
            
            # Distribute agent 1's fleet
            base_vehicles_agent1 = fleet_agent1 // num_nodes
            remainder_agent1 = fleet_agent1 % num_nodes
            
            # Assign vehicles to each node for both agents
            for idx, n in enumerate(nodes_list):
                vehicles_agent0 = base_vehicles_agent0 + (1 if idx < remainder_agent0 else 0)
                vehicles_agent1 = base_vehicles_agent1 + (1 if idx < remainder_agent1 else 0)
                self.G.nodes[n]['accInit_agent0'] = vehicles_agent0
                self.G.nodes[n]['accInit_agent1'] = vehicles_agent1
            
            self.tf = tf
            self.demand_ratio = defaultdict(list)

            # demand mutiplier over time
            if demand_ratio == None or type(demand_ratio) == list or type(demand_ratio) == dict:
                for i, j in self.edges:
                    if type(demand_ratio) == list:
                        self.demand_ratio[i, j] = list(np.interp(range(0, tf), np.arange(
                            0, tf+1, tf/(len(demand_ratio)-1)), demand_ratio))+[demand_ratio[-1]]*tf
                    elif type(demand_ratio) == dict:
                        self.demand_ratio[i, j] = list(np.interp(range(0, tf), np.arange(0, tf+1, tf/(len(demand_ratio[i]) - 1)), demand_ratio[i]))+[demand_ratio[i][-1]]*tf
                    else:
                        self.demand_ratio[i, j] = [1]*(tf+tf)
            else:
                for i, j in self.edges:
                    if (i, j) in demand_ratio:
                        self.demand_ratio[i, j] = list(np.interp(range(0, tf), np.arange(
                            0, tf+1, tf/(len(demand_ratio[i, j])-1)), demand_ratio[i, j]))+[1]*tf
                    else:
                        self.demand_ratio[i, j] = list(np.interp(range(0, tf), np.arange(
                            0, tf+1, tf/(len(demand_ratio['default'])-1)), demand_ratio['default']))+[1]*tf
            if self.fix_price:  # fix price
                self.p = defaultdict(dict)
                for i, j in self.edges:
                    self.p[i, j] = (np.random.rand()*2+1) * \
                        (self.demandTime[i, j][0]+1)
            if tripAttr != None:  # given demand as a defaultdict(dict)
                self.tripAttr = deepcopy(tripAttr)
            else:
                self.tripAttr = self.get_random_demand()  # randomly generated demand
        else:
            self.varying_time = varying_time
            
            self.is_json = True

            with open(json_file, "r") as file:
                data = json.load(file)

            self.tstep = json_tstep

            self.N1 = data["nlat"]
            self.N2 = data["nlon"]

            self.demand_input = defaultdict(dict)

            self.json_regions = json_regions

            # Create a directed graph representing the regions and their connections.
            if json_regions != None:
                self.G = nx.complete_graph(json_regions)
            elif 'region' in data:
                self.G = nx.complete_graph(data['region'])
            else:
                self.G = nx.complete_graph(self.N1*self.N2)
            
            self.G = self.G.to_directed()
            # Add self-loops to the graph for within-region trips
            self.G.add_edges_from([(i, i) for i in self.G.nodes])

            # Will hold aggregated/averaged prices per OD per time bin (p[(o,d)][t])
            self.p = defaultdict(dict)

            # No randomness is added to demand input. Hence demand is fixed. If alpha = 0.2 demand_input will fluctuate within [0.8, 1.2] * demand_input 
            self.alpha = alpha

            # Creates stucture for travel time per OD per time bin (demandTime[(o,d)][t])
            self.demandTime = defaultdict(dict)

            # Creates structure for rebalancing time per OD per time bin (rebTime[(o,d)][t])
            self.rebTime = defaultdict(dict)

            self.json_start = json_hr * 60

            self.tf = tf

            self.edges = list(self.G.edges)

            self.nregion = len(self.G)

            for i, j in self.demand_input:
                self.demandTime[i, j] = defaultdict(float)
                self.rebTime[i, j] = 1
           
            matrix_demand = defaultdict(lambda: np.zeros((self.nregion,self.nregion)))
            matrix_price_ori = defaultdict(lambda: np.zeros((self.nregion,self.nregion)))
            for item in data["demand"]:
                t, o, d, v, tt, p = item["time_stamp"], item["origin"], item[
                    "destination"], item["demand"], item["travel_time"], item["price"]
                
                if json_regions != None and (o not in json_regions or d not in json_regions):
                    continue
                
                if (o, d) not in self.demand_input:
                    self.demand_input[o, d], self.p[o, d], self.demandTime[o, d] = defaultdict(
                        float), defaultdict(float), defaultdict(float)

                self.demand_input[o, d][(
                    t-self.json_start)//json_tstep] += v*demand_ratio

                self.p[o, d][(t-self.json_start) //
                             json_tstep] += p*v*demand_ratio

                self.demandTime[o, d][(t-self.json_start) //
                                      json_tstep] += tt*v*demand_ratio/json_tstep

                matrix_demand[(t-self.json_start) //
                                      json_tstep][o,d] += v*demand_ratio

                matrix_price_ori[(t-self.json_start) //
                                      json_tstep][o,d] += p*v*demand_ratio

            
            for o, d in self.edges:
                for t in range(0, tf*2):

                    if t in self.demand_input[o, d]:
                        self.p[o, d][t] /= self.demand_input[o, d][t]

                        self.demandTime[o, d][t] /= self.demand_input[o, d][t]
                        self.demandTime[o, d][t] = max(
                            int(round(self.demandTime[o, d][t])), 1)

                        matrix_price_ori[t][o,d] /= matrix_demand[t][o,d]
                    
                    else:
                        self.demand_input[o, d][t] = 0
                        self.p[o, d][t] = 0
                        self.demandTime[o, d][t] = 0

            matrix_reb = np.zeros((self.nregion,self.nregion))

            for item in data["rebTime"]:

                hr, o, d, rt = item["time_stamp"], item["origin"], item["destination"], item["reb_time"]

                if json_regions != None and (o not in json_regions or d not in json_regions):
                    continue

                # If varying time is true (default False
                # Each JSON rebTime record with hour hr is mapped to the time bins that cover that hour (a sliding window). 
                # Effect: rebalancing time is written only into the bins that correspond to the actual hour hr in the JSON 
                # (so rebTime varies across the timeline according to the timestamps in the file). 
                if varying_time:
                    t0 = int((hr*60 - self.json_start)//json_tstep)
                    t1 = int((hr*60 + 60 - self.json_start)//json_tstep)
                    for t in range(t0, t1):
                        self.rebTime[o, d][t] = max(
                            int(round(rt/json_tstep)), 1)
                else:
                    if hr == json_hr:
                        for t in range(0, tf+1):
                            self.rebTime[o, d][t] = max(
                                int(round(rt/json_tstep)), 1)
                            matrix_reb[o,d] = rt/json_tstep
            
            # KNN regression for each time step
            if impute:
                knn = defaultdict(lambda: KNeighborsRegressor(n_neighbors=3))
                for t in matrix_price_ori.keys():
                    reb = matrix_reb
                    price = matrix_price_ori[t]
                    X = []
                    y = []
                    for i in range(self.nregion):
                        for j in range(self.nregion):
                            if price[i,j] != 0:
                                X.append(reb[i,j])
                                y.append(price[i,j])
                    X_train = np.array(X).reshape(-1, 1)
                    y_train = np.array(y)

                    knn[t].fit(X_train, y_train)

                for o, d in self.edges:
                    for t in range(0, tf*2):
                        if self.p[o,d][t]==0 and t in knn.keys():
                            
                            knn_regressor = knn[t]

                            X_test = np.array([[matrix_reb[o,d]]])

                            y_pred = knn_regressor.predict(X_test)[0]
                            self.p[o,d][t] = float(y_pred)

            # Initial vehicle distribution
            # Data contains hour and total number of vehicles in network
            # Use total_vehicles if provided, otherwise read from data
            if total_vehicles is not None:
                # Use the provided total_vehicles value
                total_fleet = total_vehicles
                fleet_agent0 = int(total_fleet * self.agent0_vehicle_ratio)
                fleet_agent1 = total_fleet - fleet_agent0
            else:
                # Read from data file
                for item in data["totalAcc"]:
                    hr, acc = item["hour"], item["acc"]
                    if hr == json_hr+int(round(json_tstep/2*tf/60)):
                        # Total fleet with supply ratio applied
                        total_fleet = int(supply_ratio * acc)
                        
                        # Split fleet between two agents using agent0_vehicle_ratio
                        fleet_agent0 = int(total_fleet * self.agent0_vehicle_ratio)
                        fleet_agent1 = total_fleet - fleet_agent0
                    
            # Create list of nodes and shuffle for random remainder assignment
            nodes_list = list(self.G.nodes)
            random.seed(sd)  # Use scenario seed for reproducibility
            random.shuffle(nodes_list)
            num_nodes = len(nodes_list)
            
            # Distribute agent 0's fleet
            base_vehicles_agent0 = fleet_agent0 // num_nodes
            remainder_agent0 = fleet_agent0 % num_nodes
            
            # Distribute agent 1's fleet
            base_vehicles_agent1 = fleet_agent1 // num_nodes
            remainder_agent1 = fleet_agent1 % num_nodes
            
            # Assign vehicles to each node for both agents
            for idx, n in enumerate(nodes_list):
                vehicles_agent0 = base_vehicles_agent0 + (1 if idx < remainder_agent0 else 0)
                vehicles_agent1 = base_vehicles_agent1 + (1 if idx < remainder_agent1 else 0)
                self.G.nodes[n]['accInit_agent0'] = vehicles_agent0
                self.G.nodes[n]['accInit_agent1'] = vehicles_agent1


            self.tripAttr = self.get_random_demand()

    def get_random_demand(self, reset=False):
        # generate demand and price
        # reset = True means that the function is called in the reset() method of AMoD enviroment,
        # assuming static demand is already generated
        # reset = False means that the function is called when initializing the demand

        demand = defaultdict(dict)
        price = defaultdict(dict)
        tripAttr = []

        # converting demand_input to static_demand
        # skip this when resetting the demand
        if self.is_json:
            for t in range(0, self.tf*2):
                for i, j in self.edges:
                    if (i, j) in self.demand_input and t in self.demand_input[i, j]:
                        demand[i, j][t] = np.random.poisson(
                            self.demand_input[i, j][t])
                        price[i, j][t] = self.p[i, j][t]
                    else:
                        demand[i, j][t] = 0
                        price[i, j][t] = 0
                    tripAttr.append((i, j, t, demand[i, j][t], price[i, j][t]))
        else:
            self.static_demand = dict()
            region_rand = (np.random.rand(len(self.G))*self.alpha *
                            2+1-self.alpha)  # multiplyer of demand
            if type(self.demand_input) in [float, int, list, np.array]:

                if type(self.demand_input) in [float, int]:
                    self.region_demand = region_rand * self.demand_input
                else:  # demand in the format of each region
                    self.region_demand = region_rand * \
                        np.array(self.demand_input)
                for i in self.G.nodes:
                    J = [j for _, j in self.G.out_edges(i)]
                    prob = np.array(
                        [np.exp(-self.rebTime[i, j][0]*self.trip_length_preference) for j in J])
                    prob = prob/sum(prob)
                    for idx in range(len(J)):
                        # allocation of demand to OD pairs
                        self.static_demand[i, J[idx]
                                            ] = self.region_demand[i] * prob[idx]
            elif type(self.demand_input) in [dict, defaultdict]:
                for i, j in self.edges:
                    self.static_demand[i, j] = self.demand_input[i, j] if (
                        i, j) in self.demand_input else self.demand_input['default']

                    self.static_demand[i, j] *= region_rand[i]
            else:
                raise Exception(
                    "demand_input should be number, array-like, or dictionary-like values")

            # generating demand and prices
            if self.fix_price:
                p = self.p
            for t in range(0, self.tf*2):
                for i, j in self.edges:
                    demand[i, j][t] = np.random.poisson(
                        self.static_demand[i, j]*self.demand_ratio[i, j][t])
                    if self.fix_price:
                        price[i, j][t] = p[i, j]
                    else:
                        price[i, j][t] = min(3, np.random.exponential(
                            2)+1)*self.demandTime[i, j][t]
                    tripAttr.append((i, j, t, demand[i, j][t], price[i, j][t]))

        return tripAttr