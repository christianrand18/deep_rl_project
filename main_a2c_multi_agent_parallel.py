"""
main_a2c_multi_agent_parallel.py — Parallel A2C multi-agent training.

N worker processes each run one full episode simultaneously with the current
policy. After all N finish, their gradients are averaged and one model update
is applied. Effective speedup ≈ N× on an N-core machine.

Only mode 2 (joint pricing + rebalancing) is supported.

Usage:
    python main_a2c_multi_agent_parallel.py --n_workers 8 --city nyc_man_south
"""
import multiprocessing as mp
import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from tqdm import trange

from src.algos.a2c_gnn_multi_agent import A2C
from src.algos.reb_flow_solver_multi_agent import solveRebFlow
from src.arguments import A2CArgumentBuilder
from src.envs.amod_env_multi import AMoD, Scenario
from src.misc.utils import dictsum

load_dotenv()

# Calibrated simulation parameters (identical to main scripts)
DEMAND_RATIO     = {'san_francisco': 2,     'nyc_man_south': 1.0,  'washington_dc': 4.2}
JSON_HR          = {'san_francisco': 19,    'nyc_man_south': 19,   'washington_dc': 19}
BETA             = {'san_francisco': 0.2,   'nyc_man_south': 0.5,  'washington_dc': 0.5}
CHOICE_INTERCEPT = {'san_francisco': 14.15, 'nyc_man_south': 9.84, 'washington_dc': 11.75}
WAGE             = {'san_francisco': 17.76, 'nyc_man_south': 22.77,'washington_dc': 25.26}

# ---------------------------------------------------------------------------
# Worker-side globals — initialised once per worker process by _worker_init
# ---------------------------------------------------------------------------
_env    = None
_models = None
_cfg    = None


def _worker_init(cfg):
    """Called once in each worker process to build a private env + model pair."""
    global _env, _models, _cfg
    _cfg  = cfg
    city  = cfg['city']

    scenario = Scenario(
        json_file=f"data/scenario_{city}.json",
        demand_ratio=DEMAND_RATIO[city] * 2,
        json_hr=JSON_HR[city],
        sd=cfg['seed'],
        json_tstep=cfg['json_tstep'],
        tf=cfg['max_steps'],
        impute=cfg['impute'],
        supply_ratio=cfg['supply_ratio'],
        agent0_vehicle_ratio=cfg['agent0_vehicle_ratio'],
        total_vehicles=cfg['total_vehicles'],
    )
    _env = AMoD(
        scenario, mode=2,
        beta=BETA[city], jitter=cfg['jitter'], max_wait=cfg['maxt'],
        choice_price_mult=cfg['choice_price_mult'], seed=cfg['seed'],
        fix_agent=cfg['fix_agent'],
        choice_intercept=CHOICE_INTERCEPT[city],
        wage=WAGE[city],
        use_dynamic_wage_man_south=cfg['use_dynamic_wage_man_south'],
        od_price_actions=cfg['od_price_actions'],
    )
    _models = {
        a: A2C(
            env=_env,
            input_size=cfg['input_size'],
            hidden_size=cfg['hidden_size'],
            device=torch.device('cpu'),
            p_lr=cfg['p_lr'],
            q_lr=cfg['q_lr'],
            T=cfg['look_ahead'],
            scale_factor=cfg['scale_factor'],
            json_file=f"data/scenario_{city}.json",
            mode=2,
            actor_clip=cfg['actor_clip'],
            critic_clip=cfg['critic_clip'],
            gamma=cfg['gamma'],
            agent_id=a,
            observe_od_prices=cfg['od_price_observe'],
            od_price_actions=cfg['od_price_actions'],
            no_share_info=cfg['no_share_info'],
            reward_scale=cfg['reward_scalar'],
        )
        for a in [0, 1]
    }


def _compute_grads(model, update_actor):
    """
    Compute gradients from the episode accumulated in model.rewards /
    model.saved_actions. Does NOT call optimizer.step — that happens in main.
    Returns a dict of cloned CPU gradient tensors plus scalar loss metrics.
    """
    R = 0
    returns = []
    for r in model.rewards[::-1]:
        R = r + model.gamma * R
        returns.insert(0, R)

    returns_t = torch.tensor(returns, dtype=torch.float32) / model.reward_scale

    policy_losses, value_losses, advantages = [], [], []
    for (log_prob, value), R_t in zip(model.saved_actions, returns_t):
        adv = R_t - value.item()
        advantages.append(adv)
        policy_losses.append(-log_prob * adv)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R_t])))

    a_loss = torch.stack(policy_losses).mean()
    v_loss = torch.stack(value_losses).mean()

    actor_grads = None
    if update_actor:
        for p in model.actor.parameters():
            if p.grad is not None:
                p.grad.zero_()
        a_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), model.actor_clip)
        actor_grads = {
            n: p.grad.clone()
            for n, p in model.actor.named_parameters()
            if p.grad is not None
        }

    for p in model.critic.parameters():
        if p.grad is not None:
            p.grad.zero_()
    v_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.critic.parameters(), model.critic_clip)
    critic_grads = {
        n: p.grad.clone()
        for n, p in model.critic.named_parameters()
        if p.grad is not None
    }

    del model.rewards[:]
    del model.saved_actions[:]

    return {
        'actor':    actor_grads,
        'critic':   critic_grads,
        'a_loss':   a_loss.item(),
        'v_loss':   v_loss.item(),
        'adv_mean': float(np.mean(advantages)),
        'adv_std':  float(np.std(advantages)),
    }


def _worker_run(task):
    """
    Run one mode-2 episode with the given model weights.
    Returns (grads_per_agent, episode_metrics).
    """
    weights, episode_seed, update_actor = task

    # Sync worker models with the latest main-process weights
    for a in [0, 1]:
        _models[a].load_state_dict(weights[a])
        _models[a].train()

    _env.seed = episode_seed
    obs = _env.reset()
    action_rl = {a: [0.0] * _env.nregion for a in [0, 1]}
    done = False

    ep_reward  = {0: 0.0, 1: 0.0}
    ep_served  = {0: 0,   1: 0}
    ep_rebcost = {0: 0.0, 1: 0.0}
    ep_rejected      = 0
    ep_total_demand  = 0

    while not done:
        obs, paxreward, done, info, system_info, _, _ = _env.match_step_simple(action_rl)

        action_rl = {a: _models[a].select_action(obs[a]) for a in [0, 1]}

        desiredAcc = {
            a: {
                _env.region[i]: int(
                    action_rl[a][i, -1] * dictsum(_env.agent_acc[a], _env.time + 1)
                )
                for i in range(_env.nregion)
            }
            for a in [0, 1]
        }
        rebAction = {a: solveRebFlow(_env, desiredAcc[a], a) for a in [0, 1]}
        _, rebreward, done, info, system_info, _, _ = _env.reb_step(rebAction)

        for a in [0, 1]:
            r = paxreward[a] + rebreward[a]
            _models[a].rewards.append(r)
            ep_reward[a]  += r
            ep_served[a]  += info[a]['served_demand']
            ep_rebcost[a] += info[a]['rebalancing_cost']

        ep_rejected     += system_info['rejected_demand']
        ep_total_demand += system_info['total_demand']

    grads   = {a: _compute_grads(_models[a], update_actor) for a in [0, 1]}
    metrics = {
        'reward':       ep_reward,
        'served':       ep_served,
        'reb_cost':     ep_rebcost,
        'rejected':     ep_rejected,
        'total_demand': ep_total_demand,
    }
    return grads, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for d in ['saved_files', 'ckpt', 'logs']:
        os.makedirs(d, exist_ok=True)

    parser = A2CArgumentBuilder.build_multi_agent_parser()
    parser.add_argument(
        '--n_workers', type=int, default=4,
        help='number of parallel episode workers (default: 4)',
    )
    args = parser.parse_args()

    if args.mode != 2:
        raise ValueError("main_a2c_multi_agent_parallel.py only supports mode 2.")

    city   = args.city
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Build main-process env (for model construction only — not used for rollouts)
    scenario = Scenario(
        json_file=f"data/scenario_{city}.json",
        demand_ratio=DEMAND_RATIO[city] * 2,
        json_hr=JSON_HR[city],
        sd=args.seed,
        json_tstep=args.json_tstep,
        tf=args.max_steps,
        impute=args.impute,
        supply_ratio=args.supply_ratio,
        agent0_vehicle_ratio=args.agent0_vehicle_ratio,
        total_vehicles=args.total_vehicles,
    )
    env = AMoD(
        scenario, mode=2,
        beta=BETA[city], jitter=args.jitter, max_wait=args.maxt,
        choice_price_mult=args.choice_price_mult, seed=args.seed,
        fix_agent=args.fix_agent,
        choice_intercept=CHOICE_INTERCEPT[city],
        wage=WAGE[city],
        use_dynamic_wage_man_south=args.use_dynamic_wage_man_south,
        od_price_actions=args.od_price_actions,
    )

    if args.od_price_observe:
        input_size = args.look_ahead + 3 + (3 * env.nregion if not args.no_share_info else env.nregion)
    else:
        input_size = args.look_ahead + (6 if not args.no_share_info else 4)

    model_agents = {
        a: A2C(
            env=env,
            input_size=input_size,
            hidden_size=args.hidden_size,
            device=device,
            p_lr=args.p_lr,
            q_lr=args.q_lr,
            T=args.look_ahead,
            scale_factor=args.scale_factor,
            json_file=f"data/scenario_{city}.json",
            mode=2,
            actor_clip=args.actor_clip,
            critic_clip=args.critic_clip,
            gamma=args.gamma,
            agent_id=a,
            observe_od_prices=args.od_price_observe,
            od_price_actions=args.od_price_actions,
            no_share_info=args.no_share_info,
            reward_scale=args.reward_scalar,
        )
        for a in [0, 1]
    }

    if args.load:
        for a in [0, 1]:
            path = f"ckpt/{args.checkpoint_path}_agent{a+1}_running.pth"
            model_agents[a].load_checkpoint(path=path)
            print(f"Loaded agent {a} from {path}")

    worker_cfg = dict(
        city=city, seed=args.seed, json_tstep=args.json_tstep,
        max_steps=args.max_steps, impute=args.impute,
        supply_ratio=args.supply_ratio,
        agent0_vehicle_ratio=args.agent0_vehicle_ratio,
        total_vehicles=args.total_vehicles,
        jitter=args.jitter, maxt=args.maxt,
        choice_price_mult=args.choice_price_mult,
        fix_agent=args.fix_agent,
        use_dynamic_wage_man_south=args.use_dynamic_wage_man_south,
        od_price_actions=args.od_price_actions,
        input_size=input_size,
        hidden_size=args.hidden_size,
        p_lr=args.p_lr, q_lr=args.q_lr,
        look_ahead=args.look_ahead, scale_factor=args.scale_factor,
        actor_clip=args.actor_clip, critic_clip=args.critic_clip,
        gamma=args.gamma,
        od_price_observe=args.od_price_observe,
        no_share_info=args.no_share_info,
        reward_scalar=args.reward_scalar,
    )

    wandb.init(
        entity="bertram-hage-danmarks-tekniske-universitet-dtu",
        project="deep_RL_project",
        name=f"{args.checkpoint_path}_p{args.n_workers}",
        config={**vars(args), 'n_workers': args.n_workers},
    )

    # Persistent worker pool — env + models initialised once per process
    pool = mp.Pool(
        processes=args.n_workers,
        initializer=_worker_init,
        initargs=(worker_cfg,),
    )

    n_updates  = args.max_episodes // args.n_workers
    best_reward = -np.inf
    epochs = trange(n_updates)

    for i_update in epochs:
        total_episodes = (i_update + 1) * args.n_workers
        update_actor   = (total_episodes >= args.critic_warmup_episodes)

        # Send current weights to workers (cpu copies — picklable)
        weights = {
            a: {k: v.cpu() for k, v in model_agents[a].state_dict().items()}
            for a in [0, 1]
        }

        # Each worker gets a distinct seed so episodes differ
        tasks = [
            (weights, args.seed + i_update * args.n_workers + w, update_actor)
            for w in range(args.n_workers)
        ]
        results     = pool.map(_worker_run, tasks)
        all_grads   = [r[0] for r in results]
        all_metrics = [r[1] for r in results]

        # Average and apply gradients
        for a in [0, 1]:
            if update_actor and all_grads[0][a]['actor'] is not None:
                model_agents[a].optimizers['a_optimizer'].zero_grad()
                for name, param in model_agents[a].actor.named_parameters():
                    if name in all_grads[0][a]['actor']:
                        param.grad = torch.stack(
                            [g[a]['actor'][name] for g in all_grads]
                        ).mean(0).to(device)
                model_agents[a].optimizers['a_optimizer'].step()

            model_agents[a].optimizers['c_optimizer'].zero_grad()
            for name, param in model_agents[a].critic.named_parameters():
                if name in all_grads[0][a]['critic']:
                    param.grad = torch.stack(
                        [g[a]['critic'][name] for g in all_grads]
                    ).mean(0).to(device)
            model_agents[a].optimizers['c_optimizer'].step()

        # Aggregate episode metrics (mean across workers)
        ep_reward  = {a: np.mean([m['reward'][a]   for m in all_metrics]) for a in [0, 1]}
        ep_served  = {a: np.mean([m['served'][a]   for m in all_metrics]) for a in [0, 1]}
        ep_rebcost = {a: np.mean([m['reb_cost'][a] for m in all_metrics]) for a in [0, 1]}
        ep_rejected     = np.mean([m['rejected']     for m in all_metrics])
        ep_demand       = np.mean([m['total_demand'] for m in all_metrics])
        a_loss_mean     = np.mean([g[0]['a_loss']    for g in all_grads])
        v_loss_mean     = np.mean([g[0]['v_loss']    for g in all_grads])
        adv_mean        = np.mean([g[0]['adv_mean']  for g in all_grads])

        wandb.log({
            'episode':                  total_episodes,
            'rewards/agent0':           ep_reward[0],
            'rewards/agent1':           ep_reward[1],
            'rewards/total':            ep_reward[0] + ep_reward[1],
            'demand/served_agent0':     ep_served[0],
            'demand/served_agent1':     ep_served[1],
            'demand/rejected':          ep_rejected,
            'demand/total':             ep_demand,
            'revenue_costs/reb_agent0': ep_rebcost[0],
            'revenue_costs/reb_agent1': ep_rebcost[1],
            'training/actor_loss':      a_loss_mean,
            'training/critic_loss':     v_loss_mean,
            'training/advantage_mean':  adv_mean,
            'training/warmup_active':   0 if update_actor else 1,
        })

        combined = ep_reward[0] + ep_reward[1]
        if combined >= best_reward:
            best_reward = combined
            for a in [0, 1]:
                model_agents[a].save_checkpoint(
                    path=f"ckpt/{args.checkpoint_path}_agent{a+1}_sample.pth")
        for a in [0, 1]:
            model_agents[a].save_checkpoint(
                path=f"ckpt/{args.checkpoint_path}_agent{a+1}_running.pth")

        epochs.set_description(
            f"Update {i_update+1} (ep {total_episodes}) | "
            f"R0={ep_reward[0]:.1f} R1={ep_reward[1]:.1f} | "
            f"Lactor={a_loss_mean:.3f} Lcritic={v_loss_mean:.3f}"
        )

    pool.close()
    pool.join()
    wandb.finish()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
