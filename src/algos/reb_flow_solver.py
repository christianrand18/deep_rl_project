"""
Minimal Rebalancing Cost
------------------------
Solves the minimum-cost rebalancing flow LP via scipy/HiGHS, bypassing PuLP
model-building overhead (which dominated runtime for these small graphs).
"""
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


def solveRebFlow(env, desiredAcc):
    t = env.time
    edges = [(i, j) for i, j in env.G.edges]
    n_edges = len(edges)

    acc_init = {n: int(env.acc[n][t + 1]) for n in env.acc}
    desired  = {n: int(round(desiredAcc[n])) for n in desiredAcc}
    region   = list(acc_init.keys())
    n_reg    = len(region)

    # Objective: minimise total rebalancing travel time
    c = np.array([env.G.edges[i, j]['time'] for i, j in edges], dtype=float)

    # Constraint matrix  (2*n_reg rows, n_edges cols)
    # Rows 0..n_reg-1      : flow conservation  (flipped to <=)
    # Rows n_reg..2*n_reg-1: supply bound        (outflow <= acc)
    #
    # Original conservation (>=): inflow[k] - outflow[k] >= desired[k] - acc[k]
    # Flipped      (<=):          outflow[k] - inflow[k] <= acc[k]   - desired[k]
    A = np.zeros((2 * n_reg, n_edges))
    b = np.zeros(2 * n_reg)

    reg_idx = {r: idx for idx, r in enumerate(region)}

    for m, (i, j) in enumerate(edges):
        if i == j:
            continue
        if i in reg_idx:
            r = reg_idx[i]
            A[r, m]          += 1   # outflow from i  (conservation)
            A[n_reg + r, m]   = 1   # supply constraint
        if j in reg_idx:
            r = reg_idx[j]
            A[r, m]          -= 1   # inflow to j     (conservation)

    for idx, r in enumerate(region):
        b[idx]          = acc_init[r] - desired[r]
        b[n_reg + idx]  = acc_init[r]

    # LP relaxation
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')
    if res.status != 0:
        return None

    x   = res.x
    tol = 1e-6

    if np.any(np.abs(x - np.round(x)) > tol):
        # Integer fallback
        result = milp(
            c,
            constraints=LinearConstraint(A, lb=-np.inf, ub=b),
            integrality=np.ones(n_edges),
            bounds=Bounds(lb=0),
        )
        if result.status != 0:
            return None
        x = result.x

    flow = defaultdict(float)
    for m, (i, j) in enumerate(edges):
        flow[(i, j)] = x[m]

    return [int(round(flow[i, j])) for i, j in env.edges]
