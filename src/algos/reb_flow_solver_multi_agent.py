"""
Minimal Rebalancing Cost
------------------------
This file contains the specifications for the Min Reb Cost problem.
"""
from collections import defaultdict
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, CPLEX_PY
from ortools.graph.python import min_cost_flow as _ortools_mcf


# Scale factor for converting float edge times to integer costs (or-tools requires ints).
# 1e6 keeps 6 decimal digits of precision — plenty for travel-time-like values.
_COST_SCALE = 1_000_000


def solveRebFlow_ortools(env, desiredAcc, agent_id):
    """Min-cost-flow formulation of the rebalancing LP, solved via or-tools."""

    t = env.time
    regions = list(env.agent_acc[agent_id].keys())
    n = len(regions)
    idx = {r: i for i, r in enumerate(regions)}

    acc_init = np.array(
        [int(env.agent_acc[agent_id][r][t + 1]) for r in regions], dtype=np.int64
    )
    desired = np.array(
        [int(round(desiredAcc[r])) for r in regions], dtype=np.int64
    )

    # Node ids: region i -> arrival node 2i, dispatch node 2i+1, sink = 2n
    sink = 2 * n
    total_supply = int(acc_init.sum())

    g_edges = [(i, j) for i, j in env.G.edges if i != j]
    n_reb = len(g_edges)
    n_arcs = n + n_reb + n  # internal arrival->dispatch + reb + slack-to-sink

    tails = np.empty(n_arcs, dtype=np.int64)
    heads = np.empty(n_arcs, dtype=np.int64)
    caps = np.empty(n_arcs, dtype=np.int64)
    costs = np.empty(n_arcs, dtype=np.int64)

    # Block A: k_a -> k_d, cap = acc_init[k], cost = 0  (outflow cap)
    tails[:n] = 2 * np.arange(n, dtype=np.int64)
    heads[:n] = 2 * np.arange(n, dtype=np.int64) + 1
    caps[:n] = acc_init
    costs[:n] = 0

    # Block B: i_d -> j_a for each reb edge, cap = total_supply, cost = scaled time
    off = n
    for k, (i, j) in enumerate(g_edges):
        tails[off + k] = 2 * idx[i] + 1
        heads[off + k] = 2 * idx[j]
        caps[off + k] = total_supply
        costs[off + k] = int(round(env.G.edges[i, j]['time'] * _COST_SCALE))

    # Block C: k_a -> sink (slack), cap = total_supply, cost = 0
    off = n + n_reb
    tails[off:] = 2 * np.arange(n, dtype=np.int64)
    heads[off:] = sink
    caps[off:] = total_supply
    costs[off:] = 0

    smcf = _ortools_mcf.SimpleMinCostFlow()
    arc_ids = smcf.add_arcs_with_capacity_and_unit_cost(tails, heads, caps, costs)

    # Supplies: arrival side has (acc - desired); dispatch nodes are zero;
    # sink absorbs the global excess.
    node_supplies = np.zeros(2 * n + 1, dtype=np.int64)
    node_supplies[0:2 * n:2] = acc_init - desired  # arrival nodes
    node_supplies[sink] = -int((acc_init - desired).sum())
    smcf.set_nodes_supplies(np.arange(2 * n + 1, dtype=np.int64), node_supplies)

    status = smcf.solve()
    if status != smcf.OPTIMAL:
        return None

    # Read reb-arc flows back (block B).
    flow_by_edge = {}
    for k, (i, j) in enumerate(g_edges):
        flow_by_edge[(i, j)] = smcf.flow(arc_ids[n + k])

    return [0 if i == j else flow_by_edge.get((i, j), 0) for i, j in env.edges]


def solveRebFlow(env, desiredAcc, agent_id):

    t = env.time
    edges = [(i, j) for i, j in env.G.edges]

    # Map vehicle availability and desired vehicles for each region
    #accTuple = [(n, int(env.agent_acc[agent_id][n][t+1])) for n in env.agent_acc[agent_id]]
    acc_init = {n: int(env.agent_acc[agent_id][n][t+1]) for n in env.agent_acc[agent_id]}
    #acc_init = {n: int(env.acc[n][t+1]) for n in env.acc}
    desired_vehicles = {n: int(round(desiredAcc[n])) for n in desiredAcc}

    region = [n for n in acc_init]
    # Time on each edge (used in the objective)
    time = {(i, j): env.G.edges[i, j]['time'] for i, j in edges}
    tol = 1e-6
    
    def build_model(var_cat):
        # Define the PuLP problem
        model = LpProblem("RebalancingFlowMinimization", LpMinimize)
        
        # Decision variables: rebalancing flow on each edge
        rebFlow = {(i, j): LpVariable(f"rebFlow_{i}_{j}", lowBound=0, cat=var_cat) for (i, j) in edges}

        # Objective: minimize total time (cost) of rebalancing flows
        model += lpSum(rebFlow[(i, j)] * time[(i, j)] for (i, j) in edges), "TotalRebalanceCost"
        
        # Constraints for each region (node)
        for k in region:
            # 1. Flow conservation constraint (ensure net inflow/outflow achieves desired vehicle distribution)
            model += (
                lpSum(rebFlow[(j, i)]-rebFlow[(i, j)] for (i, j) in edges if j != i and i==k)
            ) >= desired_vehicles[k] - acc_init[k], f"FlowConservation_{k}"

            # 2. Rebalancing flows from region i should not exceed the available vehicles in region i
            model += (
                lpSum(rebFlow[(i, j)] for (i, j) in edges if i != j and i==k) <= acc_init[k], 
                f"RebalanceSupply_{k}"
            )
        return model, rebFlow
    
    model, rebFlow = build_model('Continuous')
    status = model.solve(CPLEX_PY(msg=False, threads=1, **{"preprocessing.presolve": 0}))
    if LpStatus[status] != "Optimal":
        return None
    else: 
        fractional = False
        flow = defaultdict(float)
        for (i, j) in edges:
            flow[(i, j)] = rebFlow[(i, j)].varValue
            if abs(flow[(i, j)] - round(flow[(i, j)])) > tol: 
                fractional = True
                break 
        if fractional:
            model, rebFlow = build_model('Integer')
            status = model.solve(CPLEX_PY(msg=False, threads=1, **{"preprocessing.presolve": 0}))
            if LpStatus[status] != "Optimal":
                return None
            else:
                flow = defaultdict(float)
                for (i, j) in edges:
                    flow[(i, j)] = rebFlow[(i, j)].varValue
        action = [int(round(flow[i,j])) for i,j in env.edges]
        return action
 
