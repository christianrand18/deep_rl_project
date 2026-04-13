"""
Minimal Rebalancing Cost 
------------------------
This file contains the specifications for the Min Reb Cost problem.
"""
from collections import defaultdict
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, CPLEX_PY


def solveRebFlow(env, desiredAcc):
    t = env.time
    edges = [(i, j) for i, j in env.G.edges]

    # Map vehicle availability and desired vehicles for each region
    acc_init = {n: int(env.acc[n][t+1]) for n in env.acc}
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
