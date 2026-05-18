#!/usr/bin/env python3
"""
benchmark_match_step.py — Micro-benchmarks for match_step_simple bottlenecks.

Times each suspicious operation in isolation with timeit, then scales by
estimated call counts to show total contribution per episode.
"""
import timeit

# ---------------------------------------------------------------------------
# Call-count estimates per episode (20 steps, 12 regions, ~12 neighbours each)
# Adjust if you have better numbers from the simulation.
# ---------------------------------------------------------------------------
N_STEPS        = 20
N_REGIONS      = 12
N_NEIGHBOURS   = 12          # average out-edges per region
N_OD_PAIRS     = N_REGIONS * N_NEIGHBOURS   # 144 per step
N_AGENTS       = 2
DEMAND_PER_OD  = 5           # average passengers per (origin, dest) per step
QUEUE_PER_REG  = 3           # average passengers waiting per region per agent

calls_shuffle      = N_OD_PAIRS * N_AGENTS * N_STEPS          # random.Random().shuffle per ep
calls_random_uni   = (N_OD_PAIRS * N_AGENTS * DEMAND_PER_OD   # enter() per passenger
                    + N_REGIONS * N_AGENTS * QUEUE_PER_REG     # match() per queued pax
                    ) * N_STEPS
calls_pax_create   = N_OD_PAIRS * N_AGENTS * DEMAND_PER_OD * N_STEPS
calls_trip_append  = N_OD_PAIRS * N_STEPS                      # trip_assignments.append
calls_pax_filter   = N_REGIONS * N_AGENTS * N_STEPS            # list comprehension per region

REPS = 100_000  # timeit repetitions

def bench(label, stmt, setup="pass", reps=REPS, calls_per_ep=None):
    t = timeit.timeit(stmt, setup=setup, number=reps)
    per_call_us = t / reps * 1e6
    line = f"  {label:<40} {per_call_us:7.2f} µs/call"
    if calls_per_ep is not None:
        total_ms = per_call_us * calls_per_ep / 1e3
        line += f"   × {calls_per_ep:>6} calls/ep  →  {total_ms:6.1f} ms/ep"
    print(line)


print("=" * 80)
print("  Micro-benchmarks for match_step_simple suspects")
print("  (call counts are estimates — adjust DEMAND_PER_OD / QUEUE_PER_REG above)")
print("=" * 80)

# 1. The shuffle pattern: random.Random(seed).shuffle(list)
bench(
    "random.Random(seed).shuffle(list-3)",
    "random.Random(10).shuffle(lst)",
    setup="import random; lst=[1,2,3]",
    calls_per_ep=calls_shuffle,
)

# 2. random.uniform — called inside choice_passenger_enter/accept
bench(
    "random.uniform(0,1)",
    "random.uniform(0,1)",
    setup="import random",
    calls_per_ep=calls_random_uni,
)

# 3. Passenger object creation (one per demand unit)
bench(
    "Passenger(...) construction",
    "Passenger(1, 0, 1, 5, 12.0, max_wait=2)",
    setup="from src.envs.structures import Passenger",
    calls_per_ep=calls_pax_create,
)

# 4. trip_assignments dict append (13-field dict per OD pair)
bench(
    "trip_assignments.append({13 fields})",
    ("lst.append({'time':1,'origin':2,'destination':3,'travel_time':4.0,"
     "'price_agent0':5.0,'price_agent1':6.0,'utility_agent0':7.0,"
     "'utility_agent1':8.0,'utility_reject':0.0,'prob_agent0':0.3,"
     "'prob_agent1':0.3,'prob_reject':0.4,'demand_agent0':2,"
     "'demand_agent1':2,'demand_rejected':1,'total_demand':5})"),
    setup="lst=[]",
    calls_per_ep=calls_trip_append,
)

# 5. List comprehension for queue cleanup  [x for x in lst if i not in set]
bench(
    "queue cleanup list comprehension",
    "[q for i,q in enumerate(queue) if i not in leave_set]",
    setup="queue=list(range(5)); leave_set={1,3}",
    calls_per_ep=calls_pax_filter,
)

# 6. np.random.choice (used in uniform-wage choice model)
bench(
    "np.random.choice(labels, size=5)",
    "np.random.choice(labels, size=5, p=probs)",
    setup=("import numpy as np; labels=np.array(['agent0','agent1','reject']);"
           " probs=np.array([0.4,0.4,0.2])"),
    calls_per_ep=N_OD_PAIRS * N_STEPS,
)

# 7. Full np.random.choice choice model block
bench(
    "full choice-model block (uniform wage)",
    """
choices = np.random.choice(labels, size=d, p=probs)
d0 = np.sum(choices == 'agent0')
d1 = np.sum(choices == 'agent1')
""",
    setup=("import numpy as np; labels=np.array(['agent0','agent1','reject']);"
           f" probs=np.array([0.4,0.4,0.2]); d={DEMAND_PER_OD}"),
    calls_per_ep=N_OD_PAIRS * N_STEPS,
)

# 8. Baseline: what is the cost of one pax.match() call?
bench(
    "pax.match(t)  [random.uniform inside]",
    "pax.match(5)",
    setup="from src.envs.structures import Passenger; pax=Passenger(1,0,1,5,12.0)",
    calls_per_ep=N_REGIONS * N_AGENTS * QUEUE_PER_REG * N_STEPS,
)

# 9. pax.enter()
bench(
    "pax.enter()",
    "pax.enter()",
    setup="from src.envs.structures import Passenger; pax=Passenger(1,0,1,5,12.0)",
    calls_per_ep=N_OD_PAIRS * N_AGENTS * DEMAND_PER_OD * N_STEPS,
)

print()
print("NOTE: 'calls/ep' are rough estimates. The →ms/ep column shows estimated")
print("      contribution to the 260ms/episode budget measured in benchmark_speed.py.")
