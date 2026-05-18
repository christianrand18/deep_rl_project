#!/usr/bin/env python3
"""
benchmark_speed.py — Wall-clock timing benchmark for the multi-agent training loop.

Run BEFORE and AFTER solver changes to measure speedup:

    python benchmark_speed.py --episodes 30
    # edit reb_flow_solver_multi_agent.py (e.g. remove threads=1)
    python benchmark_speed.py --episodes 30
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from dotenv import load_dotenv

from src.algos.a2c_gnn_multi_agent import A2C
from src.algos.reb_flow_solver_multi_agent import solveRebFlow
from src.envs.amod_env_multi import Scenario, AMoD
from src.misc.utils import dictsum

_reb_executor = ThreadPoolExecutor(max_workers=2)

def _parallel_reb(env, desiredAcc):
    futs = {a: _reb_executor.submit(solveRebFlow, env, desiredAcc[a], a) for a in [0, 1]}
    return {a: futs[a].result() for a in [0, 1]}

load_dotenv()

# Calibrated simulation parameters (identical to main scripts)
DEMAND_RATIO    = {"san_francisco": 2,     "nyc_man_south": 1.0,  "washington_dc": 4.2}
JSON_HR         = {"san_francisco": 19,    "nyc_man_south": 19,   "washington_dc": 19}
BETA            = {"san_francisco": 0.2,   "nyc_man_south": 0.5,  "washington_dc": 0.5}
CHOICE_INTERCEPT= {"san_francisco": 14.15, "nyc_man_south": 9.84, "washington_dc": 11.75}
WAGE            = {"san_francisco": 17.76, "nyc_man_south": 22.77,"washington_dc": 25.26}


def parse_args():
    p = argparse.ArgumentParser(description="Speed benchmark for multi-agent training")
    p.add_argument("--episodes",    type=int, default=20,            help="episodes to time (default: 20)")
    p.add_argument("--city",        type=str, default="nyc_man_south",help="city (default: nyc_man_south)")
    p.add_argument("--mode",        type=int, default=2,             help="env mode 0/1/2 (default: 2)")
    p.add_argument("--max_steps",   type=int, default=20,            help="steps per episode (default: 20)")
    p.add_argument("--seed",        type=int, default=10,            help="random seed (default: 10)")
    p.add_argument("--hidden_size", type=int, default=256,           help="GNN hidden size (default: 256)")
    p.add_argument("--look_ahead",  type=int, default=6,             help="look-ahead steps (default: 6)")
    return p.parse_args()


def build_env_and_models(args):
    city = args.city
    scenario = Scenario(
        json_file=f"data/scenario_{city}.json",
        demand_ratio=DEMAND_RATIO[city] * 2,
        json_hr=JSON_HR[city],
        sd=args.seed,
        json_tstep=3,
        tf=args.max_steps,
        impute=0,
        supply_ratio=1.0,
        agent0_vehicle_ratio=0.5,
        total_vehicles=None,
    )
    env = AMoD(
        scenario, args.mode,
        beta=BETA[city], jitter=1, max_wait=2,
        choice_price_mult=1.0, seed=args.seed,
        fix_agent=2,
        choice_intercept=CHOICE_INTERCEPT[city],
        wage=WAGE[city],
        use_dynamic_wage_man_south=False,
        od_price_actions=False,
    )
    input_size = args.look_ahead + 6  # aggregated prices with share_info
    device = torch.device("cpu")
    model_agents = {
        a: A2C(
            env=env,
            input_size=input_size,
            hidden_size=args.hidden_size,
            device=device,
            p_lr=2e-4, q_lr=6e-4,
            T=args.look_ahead,
            scale_factor=0.01,
            json_file=f"data/scenario_{city}.json",
            mode=args.mode,
            actor_clip=1000, critic_clip=1000,
            gamma=0.97,
            agent_id=a,
            observe_od_prices=False,
            od_price_actions=False,
            no_share_info=False,
            reward_scale=2000.0,
        )
        for a in [0, 1]
    }
    for a in [0, 1]:
        model_agents[a].train()
    return env, model_agents


def run_episode(env, model_agents, mode):
    """One full training episode. Returns (wall_time, timings_dict)."""
    t = {"cplex": 0.0, "match": 0.0, "reb_step": 0.0, "select": 0.0, "train": 0.0}
    n_cplex_calls = 0

    def timed_parallel_reb(env, desiredAcc):
        t0 = time.perf_counter()
        result = _parallel_reb(env, desiredAcc)
        t["cplex"] += time.perf_counter() - t0
        return result

    obs = env.reset()
    action_rl = {a: [0.0] * env.nregion for a in [0, 1]}
    done = False
    t_ep = time.perf_counter()

    while not done:
        if mode == 0:
            t0 = time.perf_counter()
            obs, paxreward, done, info, _, _, _ = env.match_step_simple()
            t["match"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            action_rl = {a: model_agents[a].select_action(obs[a]) for a in [0, 1]}
            t["select"] += time.perf_counter() - t0

            desiredAcc = {
                a: {env.region[i]: int(action_rl[a][i] * dictsum(env.agent_acc[a], env.time + 1))
                    for i in range(env.nregion)}
                for a in [0, 1]
            }
            rebAction = timed_parallel_reb(env, desiredAcc)
            n_cplex_calls += 2

            t0 = time.perf_counter()
            _, rebreward, done, info, _, _, _ = env.reb_step(rebAction)
            t["reb_step"] += time.perf_counter() - t0

            for a in [0, 1]:
                model_agents[a].rewards.append(paxreward[a] + rebreward[a])

        elif mode == 1:
            t0 = time.perf_counter()
            obs, paxreward, done, info, _, _, _ = env.match_step_simple(action_rl)
            t["match"] += time.perf_counter() - t0

            for a in [0, 1]:
                model_agents[a].rewards.append(paxreward[a])

            t0 = time.perf_counter()
            action_rl = {a: model_agents[a].select_action(obs[a]) for a in [0, 1]}
            t["select"] += time.perf_counter() - t0

            env.matching_update()

        elif mode == 2:
            t0 = time.perf_counter()
            obs, paxreward, done, info, _, _, _ = env.match_step_simple(action_rl)
            t["match"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            action_rl = {a: model_agents[a].select_action(obs[a]) for a in [0, 1]}
            t["select"] += time.perf_counter() - t0

            desiredAcc = {
                a: {env.region[i]: int(action_rl[a][i, -1] * dictsum(env.agent_acc[a], env.time + 1))
                    for i in range(env.nregion)}
                for a in [0, 1]
            }
            rebAction = timed_parallel_reb(env, desiredAcc)
            n_cplex_calls += 2

            t0 = time.perf_counter()
            _, rebreward, done, info, _, _, _ = env.reb_step(rebAction)
            t["reb_step"] += time.perf_counter() - t0

            for a in [0, 1]:
                model_agents[a].rewards.append(paxreward[a] + rebreward[a])

        else:
            raise ValueError(f"Benchmark supports modes 0, 1, 2 (got {mode})")

    t0 = time.perf_counter()
    for a in [0, 1]:
        model_agents[a].training_step(update_actor=True)
    t["train"] = time.perf_counter() - t0

    t["n_cplex"] = n_cplex_calls
    t["wall"] = time.perf_counter() - t_ep
    return t["wall"], t


def main():
    args = parse_args()
    env, model_agents = build_env_and_models(args)

    # Warmup: one episode to prime JIT / CPLEX cache, not counted
    print("Warming up (1 episode)...", flush=True)
    run_episode(env, model_agents, args.mode)

    print(f"Running {args.episodes} timed episodes...\n", flush=True)
    all_timings = []

    for ep in range(args.episodes):
        ep_time, timings = run_episode(env, model_agents, args.mode)
        all_timings.append(timings)
        print(f"  ep {ep+1:>3}/{args.episodes}  {ep_time:.2f}s  "
              f"(match={timings['match']:.2f}s  select={timings['select']:.2f}s  "
              f"cplex={timings['cplex']:.2f}s  reb={timings['reb_step']:.2f}s  "
              f"train={timings['train']:.2f}s)", flush=True)

    def total(key):
        return sum(t[key] for t in all_timings)

    wall_total = total("wall")
    keys = ["match", "select", "cplex", "reb_step", "train"]
    labels = {
        "match":    "match_step_simple",
        "select":   "select_action (GNN)",
        "cplex":    "CPLEX solver",
        "reb_step": "reb_step",
        "train":    "training_step",
    }

    n_cplex      = total("n_cplex")
    calls_per_ep = n_cplex / args.episodes
    accounted    = sum(total(k) for k in keys)
    other        = wall_total - accounted

    print()
    print("=" * 58)
    print(f"  Results — {args.episodes} ep | mode={args.mode} | city={args.city}")
    print("=" * 58)
    print(f"  Wall time total  : {wall_total:.1f} s")
    print(f"  Per episode      : {np.mean([t['wall'] for t in all_timings]):.2f} s"
          f"  (±{np.std([t['wall'] for t in all_timings]):.2f})")
    print()
    print("  Breakdown (per episode avg):")
    for k in keys:
        tot = total(k)
        pct = 100 * tot / wall_total
        per_ep = tot / args.episodes
        extra = ""
        if k == "cplex" and n_cplex > 0:
            extra = f"  [{1000*tot/n_cplex:.1f} ms/call, {calls_per_ep:.0f} calls/ep]"
        print(f"    {labels[k]:<22}: {per_ep:.3f} s  ({pct:4.1f}%){extra}")
    if other > 0.001:
        print(f"    {'other/overhead':<22}: {other/args.episodes:.3f} s  ({100*other/wall_total:4.1f}%)")
    print()
    print(f"  Projected 100k ep: {wall_total / args.episodes * 100_000 / 3600:.1f} hours")
    print("=" * 58)


if __name__ == "__main__":
    main()
