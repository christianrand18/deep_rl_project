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
    def __init__(self, scenario, mode, beta, jitter, max_wait, choice_price_mult, seed, fix_agent, choice_intercept, wage, use_dynamic_wage_man_south=False, od_price_actions=False):
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
        
        # Trip assignment tracking: stores detailed data for each trip
        self.trip_assignments = []
    
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

        total_original_demand = 0
        total_rejected_demand = 0

        # OPTIMIZED: Move condition check and scalar extraction outside nested loops (8.47x speedup)
        # Check condition once instead of for every (n,j) edge pair
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
                            # OD-based: extract per (origin, destination) price scalars
                            for n in self.region:
                                for j in self.G[n]:
                                    # price[agent_id][n][j] gives the OD price scalar
                                    # Works for both mode 1 ([nregion, nregion]) and mode 2 ([nregion, nregion+1])
                                    scalar = float(price[agent_id][n][j])
                                    price_scalars[agent_id][(n, j)] = scalar
                        else:
                            # Origin-based: extract per-region price scalars
                            for n in self.region:
                                scalar = price[agent_id][n]
                                if isinstance(scalar, (list, np.ndarray)):
                                    scalar = scalar[0]
                                price_scalars[agent_id][n] = scalar

        for n in self.region:
            # Update current queue
            for j in self.G[n]:
                d = self.demand[n, j][t]
                
                # Apply pre-extracted price scalars (if available)
                # Inner loop now just does cheap dict lookup instead of condition checks
                if price_scalars is not None:
                    for agent_id in self.agents:
                        baseline_price = self.agent_price[agent_id][n, j][t]
                        if self.od_price_actions:
                            price_scalar = price_scalars[agent_id][(n, j)]
                        else:
                            price_scalar = price_scalars[agent_id][n]
                        
                        # Calculate proposed price (multiply by 2 to allow range [0, 2×baseline])
                        p = 2 * baseline_price * price_scalar
                        
                        # Ensure absolute minimum price (avoid zero prices)
                        if p <= 1e-6:
                            p = self.jitter
                        
                        self.agent_price[agent_id][n, j][t] = p

                ####################### Choice Model Implementation #################
                d_original = d  # before applying choice model

                #--Choice Model--
                
                pr0 = self.agent_price[0][n, j][t]
                pr1 = self.agent_price[1][n, j][t]
                
                travel_time = self.demandTime[n, j][t]
                
                travel_time_in_hours = travel_time / 60
                U_reject = 0 
                
                d0 = d1 = dr = 0

                # Use choice model with appropriate choice set
                if d_original > 0:
                    # Batched wage sampling for performance
                    if self.use_dynamic_wage_man_south and self.wage_distributions is not None:
                        # Sample all passenger wages at once from region-specific distribution
                        if n in self.wage_distributions:
                            dist = self.wage_distributions[n]
                            passenger_wages = np.random.choice(
                                dist['wages'], 
                                size=int(d_original), 
                                p=dist['probabilities']
                            )
                        else:
                            # Fallback to uniform wage if region not in distribution
                            passenger_wages = np.full(int(d_original), self.wage)
                        
                        # Track sampled wages for system-wide average
                        self.system_wage_samples.extend(passenger_wages.tolist())
                        
                        # Vectorized income effect calculation: city_avg_wage / passenger_wage
                        income_effects = self.city_avg_wage / passenger_wages
                        
                        # Vectorized utility calculations for all passengers
                        U_0_batch = (
                            self.choice_intercept 
                            - 0.71 * passenger_wages * travel_time_in_hours 
                            - income_effects * self.choice_price_mult * pr0
                        )
                        U_1_batch = (
                            self.choice_intercept 
                            - 0.71 * passenger_wages * travel_time_in_hours 
                            - income_effects * self.choice_price_mult * pr1
                        )
                        U_reject_batch = np.full(int(d_original), U_reject)
                        
                        # Vectorized probability calculation
                        exp_utilities_batch = np.column_stack([
                            np.exp(U_0_batch), 
                            np.exp(U_1_batch), 
                            np.exp(U_reject_batch)
                        ])
                        probabilities_batch = exp_utilities_batch / exp_utilities_batch.sum(axis=1, keepdims=True)
                        
                        # Fully vectorized choice sampling (no loops)
                        random_values = np.random.rand(int(d_original))
                        cumsum_probs = np.cumsum(probabilities_batch, axis=1)
                        choices = np.sum(random_values[:, None] > cumsum_probs, axis=1)
                        d0 = np.sum(choices == 0)
                        d1 = np.sum(choices == 1)
                        dr = np.sum(choices == 2)
                        
                        # For logging: use average utilities across all passengers
                        U_0 = np.mean(U_0_batch)
                        U_1 = np.mean(U_1_batch)
                        U_reject_mean = U_reject
                        avg_probabilities = probabilities_batch.mean(axis=0)
                        
                    else:
                        # Original uniform wage approach (track wages too)
                        passenger_wages = np.full(int(d_original), self.wage)
                        self.system_wage_samples.extend(passenger_wages.tolist())
                        
                        income_effect = self.wage / self.wage  # Always 1.0
                        
                        # Compute utilities for all agents (same for all passengers)
                        U_0 = self.choice_intercept - 0.71 * self.wage * travel_time_in_hours - income_effect * self.choice_price_mult * pr0
                        U_1 = self.choice_intercept - 0.71 * self.wage * travel_time_in_hours - income_effect * self.choice_price_mult * pr1
                        U_reject_mean = U_reject
                        
                        # Build choice set
                        exp_utilities = []
                        labels = []
                        
                        # Always include both agents in the choice set
                        # (Fixed agent will use base price due to scalar 0.5)
                        exp_utilities.append(np.exp(U_0))
                        labels.append("agent0")
                        exp_utilities.append(np.exp(U_1))
                        labels.append("agent1")
                        
                        # Always include reject option
                        exp_utilities.append(np.exp(U_reject))
                        labels.append("reject")

                        Probabilities = np.array(exp_utilities) / np.sum(exp_utilities)
                        labels_array = np.array(labels)
                        
                        # Batch sample all choices at once with uniform wage
                        choices = np.random.choice(labels_array, size=int(d_original), p=Probabilities)
                        d0 = np.sum(choices == "agent0")
                        d1 = np.sum(choices == "agent1")
                        dr = np.sum(choices == "reject")
                        
                        avg_probabilities = Probabilities
                    
                    # Log trip assignment details (use average values for dynamic wage case)
                    self.trip_assignments.append({
                        'time': t,
                        'origin': n,
                        'destination': j,
                        'travel_time': travel_time,
                        'price_agent0': pr0,
                        'price_agent1': pr1,
                        'utility_agent0': U_0,
                        'utility_agent1': U_1,
                        'utility_reject': U_reject_mean,
                        'prob_agent0': avg_probabilities[0],
                        'prob_agent1': avg_probabilities[1],
                        'prob_reject': avg_probabilities[2],
                        'demand_agent0': d0,
                        'demand_agent1': d1,
                        'demand_rejected': dr,
                        'total_demand': d_original
                    })

                self.agent_demand[0][(n, j)][t] += d0
                self.agent_demand[1][(n, j)][t] += d1

                pax0, self.agent_arrivals[0] = generate_passenger((n, j, t, d0, pr0), self.max_wait, self.agent_arrivals[0])
                pax1, self.agent_arrivals[1] = generate_passenger((n, j, t, d1, pr1), self.max_wait, self.agent_arrivals[1])

                self.agent_passenger[0][n][t].extend(pax0)
                self.agent_passenger[1][n][t].extend(pax1)

                random.Random(self.seed).shuffle(self.agent_passenger[0][n][t])
                random.Random(self.seed).shuffle(self.agent_passenger[1][n][t])

                total_original_demand += d_original
                total_rejected_demand += dr

                self.demand[n, j][t] = d0 + d1
            
            for agent_id in [0, 1]:
                accCurrent = self.agent_acc[agent_id][n][t]

                new_enterq = [pax for pax in self.agent_passenger[agent_id][n][t] if pax.enter()]
                queueCurrent = self.agent_queue[agent_id][n] + new_enterq
                self.agent_queue[agent_id][n] = queueCurrent

                matched_leave_index = []

                for i, pax in enumerate(queueCurrent):
                    if accCurrent > 0:
                        accept = pax.match(t)
                        if accept:
                            matched_leave_index.append(i)
                            accCurrent -= 1
                            
                            arr_t = t + self.demandTime[pax.origin, pax.destination][t]
                            self.agent_paxFlow[agent_id][pax.origin, pax.destination][arr_t] += 1
                            
                            wait_t = pax.wait_time
                            self.agent_paxWait[agent_id][pax.origin, pax.destination].append(wait_t)

                            self.agent_dacc[agent_id][pax.destination][arr_t] += 1

                            self.agent_servedDemand[agent_id][pax.origin, pax.destination][t] += 1

                            trip_cost = self.demandTime[pax.origin, pax.destination][t] * self.beta
                            trip_revenue = pax.price
                            
                            # Calculate profitability-aware reward
                            base_reward = trip_revenue - trip_cost
                            
                            paxreward[agent_id] += base_reward

                            self.ext_reward_agents[agent_id][n] += max(0, trip_cost)

                            self.agent_info[agent_id]['revenue'] += trip_revenue
                            self.agent_info[agent_id]['served_demand'] += 1
                            self.agent_info[agent_id]['operating_cost'] += trip_cost
                            self.agent_info[agent_id]['served_waiting'] += wait_t
                            self.agent_info[agent_id]['true_profit'] += base_reward
                        else:
                            if pax.unmatched_update():
                                matched_leave_index.append(i)
                                self.agent_unservedDemand[agent_id][pax.origin, pax.destination][t] += 1
                                self.agent_info[agent_id]['unserved_demand'] += 1
                    else:
                        if pax.unmatched_update():
                            matched_leave_index.append(i)
                            self.agent_unservedDemand[agent_id][pax.origin, pax.destination][t] += 1
                            self.agent_info[agent_id]['unserved_demand'] += 1

                self.agent_queue[agent_id][n] = [
                    queueCurrent[i] for i in range(len(queueCurrent)) if i not in matched_leave_index
                ]
                self.agent_acc[agent_id][n][t+1] = accCurrent
            
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
        self.ext_reward_agents = {a: np.zeros(self.nregion) for a in [0, 1]}
    
        for agent_id in [0, 1]:
            self.agent_info[agent_id]['rebalancing_cost'] = 0

        # OPTIMIZED rebalancing loop: pre-calculate reb_time and cost once per edge
        for agent_id in [0, 1]:
            
            rebAction = rebAction_agents[agent_id]
    
            for k in range(len(self.edges)):
                i, j = self.edges[k]

                # OPTIMIZED: Pre-calculate reb_time once (was looked up 2 times)
                reb_time = self.rebTime[i, j][t]
                # OPTIMIZED: Pre-calculate cost once (eliminates redundant lookup and multiplications)
                rebalancing_cost = reb_time * self.beta * rebAction[k]

                self.agent_rebFlow[agent_id][i, j][t + reb_time] = rebAction[k]
                self.agent_rebFlow_ori[agent_id][i, j][t] = rebAction[k]
    
                self.agent_acc[agent_id][i][t+1] -= rebAction[k]
                self.agent_dacc[agent_id][j][t + reb_time] += rebAction[k]

                # Use pre-calculated rebalancing cost (instead of recalculating)
                rebreward[agent_id] -= rebalancing_cost
                self.ext_reward_agents[agent_id][i] -= rebalancing_cost

                self.agent_info[agent_id]['rebalancing_cost'] += rebalancing_cost
    
        for agent_id in [0, 1]:
            for k in range(len(self.edges)):
                i, j = self.edges[k]
                if (i, j) in self.agent_rebFlow[agent_id] and t in self.agent_rebFlow[agent_id][i, j]:
                    self.agent_acc[agent_id][j][t+1] += self.agent_rebFlow[agent_id][i, j][t]
                if (i, j) in self.agent_paxFlow[agent_id] and t in self.agent_paxFlow[agent_id][i, j]:
                    self.agent_acc[agent_id][j][t+1] += self.agent_paxFlow[agent_id][i, j][t]
        
        # For fixed agents, reset vehicle distribution to initial state
        if self.fix_agent in [0, 1]:
            fixed_agent_id = self.fix_agent
            for n in self.region:
                self.agent_acc[fixed_agent_id][n][t+1] = self.agent_initial_acc[fixed_agent_id][n]
    
        self.time += 1
    
        self.obs = {
            0: (self.agent_acc[0], self.time, self.agent_dacc[0], self.agent_demand[0]),
            1: (self.agent_acc[1], self.time, self.agent_dacc[1], self.agent_demand[1])
        }
    
        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.rebTime[i, j][self.time]
    
        done = (self.tf == t + 1)
        ext_done = [done] * self.nregion
    
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
    
    def reset(self):
        """Reset the episode for multi-agent environment"""
        
        self.trip_assignments = []
        
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