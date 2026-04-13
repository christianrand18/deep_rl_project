"""
Autonomous Mobility-on-Demand Environment
-----------------------------------------
This file contains the specifications for the AMoD system simulator.
"""
from collections import defaultdict
from copy import deepcopy
import json
import random

import networkx as nx
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from src.envs.structures import generate_passenger


class AMoD:
    def __init__(self, scenario, mode, beta, jitter, max_wait, choice_price_mult, seed, choice_intercept, fix_baseline=False, wage=25, od_price_actions=False):
        self.scenario = deepcopy(scenario)
        self.mode = mode  # Mode of rebalancing (0:manul, 1:pricing, 2:both. default 1)
        self.jitter = jitter # Jitter for zero demand
        self.max_wait = max_wait # Maximum passenger waiting time
        self.fix_baseline = fix_baseline  # Fix baseline behavior (base price + initial distribution)
        self.od_price_actions = od_price_actions  # Use OD-based price scalars (N×N) instead of origin-based (N)
        # Choice model intercept (utility of using ridehailing service)
        self.choice_intercept = choice_intercept
        # Wage parameter for choice model
        self.wage = wage
        self.G = scenario.G  # Road Graph: node - regiocon'dn, edge - connection of regions, node attr: 'accInit', edge attr: 'time'
        self.demandTime = self.scenario.demandTime
        self.rebTime = self.scenario.rebTime
        self.time = 0  # current time
        self.tf = scenario.tf  # final time
        self.tstep = scenario.tstep
        self.passenger = dict()  # passenger arrivals
        self.queue = defaultdict(list)
        self.demand = defaultdict(dict)
        self.region = list(self.G)
        for i in self.region:
            self.passenger[i] = defaultdict(list)


        self.price = defaultdict(dict)
        self.arrivals = 0  # total number of added passengers
        # trip attribute (origin, destination, time of request, demand, price)
        for i, j, t, d, p in scenario.tripAttr:
            self.demand[i, j][t] = d
            self.price[i, j][t] = p
            
        # number of vehicles within each region, key: i - region, t - time
        self.acc = defaultdict(dict)
        # number of vehicles arriving at each region, key: i - region, t - time
        self.dacc = defaultdict(dict)
        # number of rebalancing vehicles, key: (i,j) - (origin, destination), t - time
        self.rebFlow = defaultdict(dict)
        self.rebFlow_ori = defaultdict(dict)
        # number of vehicles with passengers, key: (i,j) - (origin, destination), t - time
        self.paxFlow = defaultdict(dict)
        self.paxWait = defaultdict(list)
        self.edges = []  # set of rebalancing edges
        self.nregion = len(scenario.G)  # number of regions
        # Set all edges
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.nedge = [len(self.G.out_edges(n))+1 for n in self.region]
        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.rebTime[i, j][self.time]
            self.rebFlow[i, j] = defaultdict(int)
            self.rebFlow_ori[i, j] = defaultdict(int)
        for i, j in self.demand:
            self.paxFlow[i, j] = defaultdict(int)
            self.paxWait[i, j] = []
        # Store initial vehicle distribution for fixed baseline mode
        self.initial_acc = {}
        for n in self.region:
            initial_count = self.G.nodes[n]['accInit']
            self.acc[n][0] = initial_count
            self.initial_acc[n] = initial_count  # Store for fixed baseline
            self.dacc[n] = defaultdict(int)
        # scenario.tstep: number of steps as one timestep
        self.beta = beta * scenario.tstep
        t = self.time
        self.servedDemand = defaultdict(dict)
        self.unservedDemand = defaultdict(dict)
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(int)
            self.unservedDemand[i, j] = defaultdict(int)

        self.N = len(self.region)

        self.info = dict.fromkeys(['revenue', 'served_demand', 'unserved_demand',
                                    'rebalancing_cost', 'operating_cost', 'served_waiting', 
                                    'true_profit', 'adjusted_profit'], 0)
        
        # System-level info tracking (not agent-specific)
        # Tracks metrics at the system level:
        # - rejected_demand: number of passengers who rejected the agent via choice model
        # - total_demand: total demand generated in the system
        # - rejection_rate: ratio of rejected demand to total demand
        self.system_info = dict.fromkeys(['rejected_demand', 'total_demand', 'rejection_rate'], 0)

        self.reward = 0
        self.choice_price_mult = choice_price_mult
        self.seed = seed
        # observation: current vehicle distribution, time, future arrivals, demand
        self.obs = (self.acc, self.time, self.dacc, self.demand)

    def match_step_simple(self, price=None):
        """
        A simple version of matching. Match vehicle and passenger in a first-come-first-serve manner. 

        price: list of price for eacj region. Default None.
        """
        t = self.time
        self.reward = 0
        self.ext_reward = np.zeros(self.nregion)
        self.agent_unprofitable_trips = 0
        
        # Reset agent-specific info
        for key in self.info:
            self.info[key] = 0
        
        for key in self.system_info:
            self.system_info[key] = 0

        total_original_demand = 0
        total_rejected_demand = 0
        
        # OPTIMIZED: Move condition check and scalar extraction outside nested loops (8.47x speedup)
        # Check condition once instead of for every (n,j) edge pair
        price_scalars = None
        if price is not None and np.sum(price) != 0:
            # Pre-extract price scalars
            price_scalars = {}
            if self.od_price_actions:
                # OD-based: extract per (origin, destination) price scalars
                for n in self.region:
                    for j in self.G[n]:
                        if self.fix_baseline:
                            price_scalars[(n, j)] = 0.5
                        else:
                            price_scalars[(n, j)] = float(price[n][j])
            else:
                # Origin-based: extract per-region price scalars
                for n in self.region:
                    if self.fix_baseline:
                        price_scalars[n] = 0.5
                    else:
                        scalar = price[n]
                        if isinstance(scalar, (list, np.ndarray)):
                            scalar = scalar[0]
                        price_scalars[n] = scalar
        
        for n in self.region:
            for j in self.G[n]:
                d = self.demand[n, j][t]
                
                # Apply pre-extracted price scalars (if available)
                # Inner loop now just does cheap dict lookup instead of condition checks
                if price_scalars is not None:
                    baseline_price = self.price[n, j][t]
                    if self.od_price_actions:
                        price_scalar = price_scalars[(n, j)]
                    else:
                        price_scalar = price_scalars[n]
                    
                    # Calculate proposed price (multiply by 2 to allow range [0, 2×baseline])
                    p = 2 * baseline_price * price_scalar
                        
                    # Ensure absolute minimum price (avoid zero prices)
                    if p <= 1e-6:
                        p = self.jitter
                    
                    self.price[n, j][t] = p

                ####################### Choice Model Implementation #################
                d_original = d  # before applying choice model
                
                current_price = self.price[n, j][t]
                travel_time = self.demandTime[n, j][t]
                travel_time_in_hours = travel_time / 60
                U_reject = 0

                exp_utilities = []
                labels = []

                income_effect = self.wage / self.wage

                utility_agent = self.choice_intercept - 0.71 * self.wage * travel_time_in_hours - income_effect * self.choice_price_mult * current_price

                exp_utilities.append(np.exp(utility_agent))
                labels.append("agent")
                # Always include reject option
                exp_utilities.append(np.exp(U_reject))
                labels.append("reject")
                Probabilities = np.array(exp_utilities) / np.sum(exp_utilities)
                labels_array = np.array(labels)

                d_agent=d_reject=0

                # Use choice model with appropriate choice set
                if d_original > 0:
                    for _ in range(d_original):
                        choice = np.random.choice(labels_array, p=Probabilities)
                        if choice == "agent":
                            d_agent += 1
                        elif choice == "reject":
                            d_reject += 1


                self.demand[n, j][t] = d_agent

                newp, self.arrivals = generate_passenger(
                    (n, j, t, d_agent, current_price), self.max_wait, self.arrivals)
                self.passenger[n][t].extend(newp)
                # shuffle passenger list at station so that the passengers are not served in destination order
                random.Random(42).shuffle(self.passenger[n][t])

                total_original_demand += d_original
                total_rejected_demand += d_reject

            accCurrent = self.acc[n][t]
            new_enterq = [pax for pax in self.passenger[n][t] if pax.enter()]
            queueCurrent = self.queue[n] + new_enterq
            self.queue[n] = queueCurrent
            matched_leave_index = []

            for i, pax in enumerate(queueCurrent):
                if accCurrent > 0:
                    accept = pax.match(t)
                    if accept:
                        matched_leave_index.append(i)
                        accCurrent -= 1
                        arr_t = t + self.demandTime[pax.origin, pax.destination][t]
                        self.paxFlow[pax.origin, pax.destination][arr_t] += 1

                        wait_t = pax.wait_time
                        self.paxWait[pax.origin, pax.destination].append(wait_t)

                        self.dacc[pax.destination][arr_t] += 1

                        self.servedDemand[pax.origin, pax.destination][t] += 1

                        trip_revenue = pax.price
                        trip_cost = self.demandTime[pax.origin, pax.destination][t] * self.beta

                        # Calculate profitability-aware reward
                        base_reward = trip_revenue - trip_cost

                        self.reward += base_reward

                        self.ext_reward[n] += max(0, trip_cost)

                        self.info['revenue'] += trip_revenue
                        self.info['served_demand'] += 1
                        self.info['operating_cost'] += trip_cost
                        self.info['served_waiting'] += wait_t
                        self.info['true_profit'] += base_reward

                    else:
                        if pax.unmatched_update():
                            matched_leave_index.append(i)
                            self.unservedDemand[pax.origin, pax.destination][t] += 1
                            self.info['unserved_demand'] += 1
                else:
                    if pax.unmatched_update():
                        matched_leave_index.append(i)
                        self.unservedDemand[pax.origin, pax.destination][t] += 1
                        self.info['unserved_demand'] += 1

            # Update queue
            self.queue[n] = [self.queue[n][i] for i in range(
                len(self.queue[n])) if i not in matched_leave_index]
            self.acc[n][t+1] = accCurrent
        
        done = (self.tf == t+1)
        ext_done = [done]*self.nregion

        # for acc, the time index would be t+1, but for demand, the time index would be t
        self.obs = (self.acc, self.time, self.dacc, self.demand)

        # Update system-level info
        self.system_info['rejected_demand'] = total_rejected_demand
        self.system_info['total_demand'] = total_original_demand
        self.system_info['rejection_rate'] = (
            total_rejected_demand / total_original_demand if total_original_demand > 0 else 0
        )
        
        # Add unprofitable trips count to agent info
        self.info['unprofitable_trips'] = self.unprofitable_trips

        return self.obs, self.reward, done, self.info, self.system_info, self.ext_reward, ext_done

    def matching_update(self):
        """Update properties if there is no rebalancing after matching"""
        t = self.time
        # Update acc. Assuming arriving vehicle will only be availbe for the next timestamp.
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) in self.paxFlow and t in self.paxFlow[i, j]:
                self.acc[j][t+1] += self.paxFlow[i, j][t]
        
        self.time += 1

    # reb_step
    def reb_step(self, rebAction):
        t = self.time
        self.reward = 0  # reward is calculated from before this to the next rebalancing, we may also have two rewards, one for pax matching and one for rebalancing
        self.ext_reward = np.zeros(self.nregion)
        self.rebAction = rebAction
        # OPTIMIZED rebalancing loop: removed redundant edge check, pre-calculate reb_time and cost factor
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            self.rebAction[k] = rebAction[k]
            
            # OPTIMIZED: Pre-calculate reb_time once (was looked up 4 times)
            reb_time = self.rebTime[i, j][t]
            # OPTIMIZED: Pre-calculate cost factor once (eliminates 3 redundant multiplications)
            reb_cost_factor = reb_time * self.beta * self.rebAction[k]
            
            self.rebFlow[i, j][t+reb_time] = self.rebAction[k]
            self.rebFlow_ori[i, j][t] = self.rebAction[k]
            self.acc[i][t+1] -= self.rebAction[k]
            self.dacc[j][t+reb_time] += self.rebFlow[i, j][t+reb_time]
            
            # Use pre-calculated cost factor (instead of recalculating 4 times)
            self.info['rebalancing_cost'] += reb_cost_factor
            self.info["operating_cost"] += reb_cost_factor
            self.reward -= reb_cost_factor
            self.ext_reward[i] -= reb_cost_factor
        # arrival for the next time step, executed in the last state of a time step
        # this makes the code slightly different from the previous version, where the following codes are executed between matching and rebalancing
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) in self.rebFlow and t in self.rebFlow[i, j]:
                self.acc[j][t+1] += self.rebFlow[i, j][t]
            if (i, j) in self.paxFlow and t in self.paxFlow[i, j]:
                # this means that after pax arrived, vehicles can only be rebalanced in the next time step, let me know if you have different opinion
                self.acc[j][t+1] += self.paxFlow[i, j][t]

        self.time += 1
        # use self.time to index the next time step
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.rebTime[i, j][self.time]
        done = (self.tf == t+1)  # if the episode is completed
        ext_done = [done]*self.nregion
        return self.obs, self.reward, done, self.info, self.system_info, self.ext_reward, ext_done

    def get_total_vehicles(self):
        """
        Calculate total number of vehicles in the system at current time.
        Includes: available vehicles + vehicles with passengers + rebalancing vehicles
        """
        t = self.time
        total = 0
        
        # Count available vehicles at all regions for CURRENT time
        for region in self.region:
            # Try current time first, then fallback to t+1
            if t in self.acc[region]:
                total += self.acc[region][t]
            elif t+1 in self.acc[region]:
                total += self.acc[region][t+1]
        
        # Count vehicles with passengers (all current and future arrivals)
        for (i, j), time_dict in self.paxFlow.items():
            for time_step, flow in time_dict.items():
                if time_step >= t:  # Current and future arrivals (vehicles in transit)
                    total += flow
        
        # Count rebalancing vehicles (all current and future arrivals)
        for (i, j), time_dict in self.rebFlow.items():
            for time_step, flow in time_dict.items():
                if time_step >= t:  # Current and future arrivals (vehicles in transit)
                    total += flow
        
        return total

    def get_initial_vehicles(self):
        """Get the initial number of vehicles in the system"""
        return sum(self.G.nodes[n]['accInit'] for n in self.G.nodes)

    def reset(self):
        self.acc = defaultdict(dict)
        self.dacc = defaultdict(dict)
        self.rebFlow = defaultdict(dict)
        self.rebFlow_ori = defaultdict(dict)
        self.paxFlow = defaultdict(dict)
        self.paxWait = defaultdict(list)
        self.passenger = dict()
        self.queue = defaultdict(list)
        self.edges = []

        self.reward = 0
        self.ext_reward = np.zeros(self.nregion)
        self.unprofitable_trips = 0

        self.info = {
            'revenue': 0,
            'served_demand': 0,
            'unserved_demand': 0,
            'rebalancing_cost': 0,
            'operating_cost': 0,
            'served_waiting': 0,
            'unprofitable_trips': 0,
            'true_profit': 0,
            'adjusted_profit': 0
        }
        
        self.system_info = {
            'rejected_demand': 0,
            'total_demand': 0,
            'rejection_rate': 0
        }

        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        for i in self.region:
            self.passenger[i] = defaultdict(list)
        self.edges = list(set(self.edges))
        self.demand = defaultdict(dict)
        self.price = defaultdict(dict)
        self.arrivals = 0
        tripAttr = self.scenario.get_random_demand(reset=True)
        self.regionDemand = defaultdict(dict)

        # trip attribute (origin, destination, time of request, demand, price)
        for i, j, t, d, p in tripAttr:
            self.demand[i, j][t] = d
            self.price[i, j][t] = p
            if t not in self.regionDemand[i]:
                self.regionDemand[i][t] = 0
            else:
                self.regionDemand[i][t] += d

        self.time = 0
        for i, j in self.G.edges:
            self.rebFlow[i, j] = defaultdict(int)
            self.rebFlow_ori[i,j] = defaultdict(int)
            self.paxFlow[i, j] = defaultdict(int)
            self.paxWait[i, j] = []
        for n in self.G:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(int)
        t = self.time
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(int)
            self.unservedDemand[i, j] = defaultdict(int)
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        return self.obs

class Scenario:
    def __init__(self, N1=2, N2=4, tf=60, sd=None, ninit=5, tripAttr=None, demand_input=None, demand_ratio=None, supply_ratio=1,
                 trip_length_preference=0.25, grid_travel_time=1, fix_price=True, alpha=0.0, json_file=None, json_hr=19, json_tstep=3, varying_time=False, json_regions=None, impute=False):
        # trip_length_preference: positive - more shorter trips, negative - more longer trips
        # grid_travel_time: travel time between grids
        # demand_input： list - total demand out of each region,
        #          float/int - total demand out of each region satisfies uniform distribution on [0, demand_input]
        #          dict/defaultdict - total demand between pairs of regions
        # demand_input will be converted to a variable static_demand to represent the demand between each pair of nodes
        # static_demand will then be sampled according to a Poisson distributionjson_tstep
        # alpha: parameter for uniform distribution of demand levels - [1-alpha, 1+alpha] * demand_input
        self.sd = sd
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
            self.demandTime = defaultdict(dict)  # traveling time between nodes
            self.rebTime = defaultdict(dict)
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
            
            # Distribute fleet evenly across nodes with remainder handling
            num_nodes = len(self.G.nodes)
            base_vehicles_per_node = total_fleet // num_nodes
            remainder = total_fleet % num_nodes
            
            # Create list of nodes and shuffle for random remainder assignment
            nodes_list = list(self.G.nodes)
            random.seed(sd)
            random.shuffle(nodes_list)
            
            # Assign vehicles to each node
            for idx, n in enumerate(nodes_list):
                vehicles = base_vehicles_per_node + (1 if idx < remainder else 0)
                self.G.nodes[n]['accInit'] = vehicles
            
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
            self.alpha = 0

            # Creates stucture for travel time per OD per time bin (demandTime[(o,d)][t])
            self.demandTime = defaultdict(dict)

            # Creates structure for rebalancing time per OD per time bin (rebTime[(o,d)][t])
            self.rebTime = defaultdict(dict)

            self.json_start = json_hr * 60

            self.tf = tf

            # Self-loops are now part of G.edges, no need to add them separately
            self.edges = list(self.G.edges)

            self.nregion = len(self.G)

            for i, j in self.demand_input:
                self.demandTime[i, j] = defaultdict(int)
                self.rebTime[i, j] = 1
        
            # Creates nregion × nregion numpy array of zeros (default) for time index t. The code will accumulate demand volumes into array cells [o,d]
            matrix_demand = defaultdict(lambda: np.zeros((self.nregion,self.nregion)))

            # Similarly stores aggregated (volume-weighted) price sums for each time index.
            matrix_price_ori = defaultdict(lambda: np.zeros((self.nregion,self.nregion)))
            for item in data["demand"]:
                t, o, d, v, tt, p = item["time_stamp"], item["origin"], item[
                    "destination"], item["demand"], item["travel_time"], item["price"]
                
                if json_regions != None and (o not in json_regions or d not in json_regions):
                    continue
                
                if (o, d) not in self.demand_input:
                    self.demand_input[o, d], self.p[o, d], self.demandTime[o, d] = defaultdict(
                        float), defaultdict(float), defaultdict(float)

                # Set ALL demand, price, and traveling time for OD. 

                # First a time index is calculated by subtracting the starting time (self.json_start) from the timestamp t
                # and dividing by the time step size (json_tstep). This bins the timestamp into discrete time indices.

                # The demand is incremented for specific OD and time index by the demand volume v, scaled by a demand_ratio factor.
                self.demand_input[o, d][(
                    t-self.json_start)//json_tstep] += v*demand_ratio

                # The price p is accumulated in a volume-weighted manner (p*v) for the same OD and time index. 
                # This is not just summing prices: it's building a demand-weighted sum. Later we divide by total demand to get average price.
                self.p[o, d][(t-self.json_start) //
                             json_tstep] += p*v*demand_ratio

                # Same as price. Accumulates travel times weighted by demand volume. 
                # Also divide by json_tstep (normalizing since demand is aggregated over that bin length) to get average travel time per unit time.
                self.demandTime[o, d][(t-self.json_start) //
                                      json_tstep] += tt*v*demand_ratio/json_tstep


                # At time bin k, matrix_demand[k][o,d] = total number of demand from o to d
                matrix_demand[(t-self.json_start) //
                                      json_tstep][o,d] += v*demand_ratio

                # At time bin k, matrix_price_ori[k][o,d] = total price (weighted by demand) from o to d
                # Later divided by matrix_demand to get average observed price per OD/time bin             
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
                        # If there is no price for specific time point, and a regressor exists for that time point, use it 
                        # to impute the price at that time point.s
                        if self.p[o,d][t]==0 and t in knn.keys():
                            
                            knn_regressor = knn[t]

                            X_test = np.array([[matrix_reb[o,d]]])

                            y_pred = knn_regressor.predict(X_test)[0]
                            self.p[o,d][t] = float(y_pred)

            # Initial vehicle distribution
            # Data contains hour and total number of vehicles in network
            for item in data["totalAcc"]:
                hr, acc = item["hour"], item["acc"]
                if hr == json_hr+int(round(json_tstep/2*tf/60)):
                    # Total fleet with supply ratio applied
                    total_fleet = int(supply_ratio * acc)
                    
                    # Distribute fleet evenly across nodes with remainder handling
                    num_nodes = len(self.G)
                    base_vehicles_per_node = total_fleet // num_nodes
                    remainder = total_fleet % num_nodes
                    
                    # Create list of nodes and shuffle for random remainder assignment
                    nodes_list = list(self.G.nodes)
                    random.seed(sd)  # Use scenario seed for reproducibility
                    random.shuffle(nodes_list)
                    
                    # Assign vehicles to each node
                    for idx, n in enumerate(nodes_list):
                        vehicles = base_vehicles_per_node + (1 if idx < remainder else 0)
                        self.G.nodes[n]['accInit'] = vehicles


            self.tripAttr = self.get_random_demand()

    def get_random_demand(self, reset=False):
        # generate demand and price
        # reset = True means that the function is called in the reset() method of AMoD enviroment,
        #   assuming static demand is already generated
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
                        [np.math.exp(-self.rebTime[i, j][0]*self.trip_length_preference) for j in J])
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