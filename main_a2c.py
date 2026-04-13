import argparse
import copy
import os
import pickle

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from tqdm import trange

from src.algos.a2c_gnn import A2C
from src.algos.reb_flow_solver import solveRebFlow
from src.envs.amod_env import Scenario, AMoD
from src.misc.utils import dictsum, nestdictsum

# Load environment variables from .env file
load_dotenv()

# Calibrated simulation parameters
demand_ratio = {'san_francisco': 2,'nyc_man_south': 1.0, 'washington_dc': 4.2}

json_hr = {'san_francisco':19,'nyc_man_south': 19, 'washington_dc': 19}

beta = {'san_francisco': 0.2,'nyc_man_south': 0.5, 'washington_dc': 0.5}

choice_intercept = {'san_francisco': 14.79, 'nyc_man_south': 10.55, 'washington_dc': 12.45}

wage = {'san_francisco': 17.76, 'nyc_man_south': 22.77, 'washington_dc': 25.26}

test_tstep = {'san_francisco': 3, 'nyc_man_south': 3, 'washington_dc':3}

parser = argparse.ArgumentParser(description="A2C-GNN")

# Simulator parameters
parser.add_argument(
    "--seed", type=int, default=10, metavar="S", help="random seed (default: 10)"
)
parser.add_argument(
    "--demand_ratio",
    type=float,
    default=1,
    metavar="S",
    help="demand_ratio (default: 1)",
)
parser.add_argument(
    "--json_hr", type=int, default=19, metavar="S", help="json_hr (default: 19)"
)
parser.add_argument(
    "--json_tstep",
    type=int,
    default=3,
    metavar="S",
    help="minutes per timestep (default: 3min)",
)
parser.add_argument(
    '--mode', 
    type=int, 
    default=2,
    help='rebalancing mode. (0:rebalancing only, 1:pricing only, 2:both, 3:baseline no reb/fixed price, 4:baseline uniform reb/fixed price. default 2)',
)

parser.add_argument(
    "--beta",
    type=float,
    default=0.5,
    metavar="S",
    help="cost of rebalancing (default: 0.5)",
)

# Model parameters
parser.add_argument(
    "--test", 
    action="store_true",
    default=False, 
    help="activates test mode for agent evaluation"
)
parser.add_argument(
    "--cplexpath",
    type=str,
    default="/apps/cplex/cplex1210/opl/bin/x86-64_linux/", # Changed to HPC PATH
    help="defines directory of the CPLEX installation",
)

parser.add_argument(
    "--directory",
    type=str,
    default="saved_files",
    help="defines directory where to save files",
)
parser.add_argument(
    "--max_episodes",
    type=int,
    default=100000,
    metavar="N",
    help="number of episodes to train agent (default: 100k)",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=20,
    metavar="N",
    help="number of steps per episode (default: 20)",
)
parser.add_argument(
    "--cuda", 
    action="store_true",
    default=False,
    help="Enables CUDA training",
)

parser.add_argument(
    "--jitter",
    type=int,
    default=1,
    help="jitter for demand 0 (default: 1)",
)
parser.add_argument(
    "--maxt",
    type=int,
    default=2,
    help="maximum passenger waiting time in time steps (default: 2)",
)

parser.add_argument(
    "--hidden_size",
    type=int,
    default=256,
    help="hidden size of neural networks (default: 256)",
)

parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="A2C",
    help="name of checkpoint file to save/load (default: A2C)",
)
parser.add_argument(
    "--load",
    action="store_true",
    default=False,
    help="either to start training from checkpoint (default: False)",
)

parser.add_argument(
    "--model_type",
    type=str,
    default="running",
    choices=["running", "test", "sample"],
    help="which checkpoint variant to load: running, test, or sample (default: running)",
)

parser.add_argument(
    "--actor_clip",
    type=float,
    default=1000,
    help="clip value for actor gradient clipping (default: 1000)",
)

parser.add_argument(
    "--critic_clip",
    type=float,
    default=1000,
    help="clip value for critic gradient clipping (default: 1000)",
)

parser.add_argument(
    "--critic_warmup_episodes",
    type=int,
    default=1000,
    help="number of episodes to train only critic before training actor (default: 1000)",
)

parser.add_argument(
    "--p_lr",
    type=float,
    default=2e-4,
    help="learning rate for policy network (default: 2e-4)",
)

parser.add_argument(
    "--q_lr",
    type=float,
    default=6e-4,
    help="learning rate for Q networks (default: 6e-4)",
)

parser.add_argument(
    "--city",
    type=str,
    default="nyc_man_south",
    help="city to train on",
)

parser.add_argument(
    "--impute",
    type=int,
    default=0,
    help="Whether impute the zero price (default: False)",
)

parser.add_argument(
    "--supply_ratio",
    type=float,
    default=1.0,
    help="supply scaling factor (default: 1)",
)

parser.add_argument(
    "--look_ahead",
    type=int,
    default=6,
    help="Time steps to look ahead (default: 6)",
)

parser.add_argument(
    "--scale_factor",
    type=float,
    default=0.01,
    help="Scale factor (default: 0.01)",
)

parser.add_argument(
    "--gamma",
    type=float,
    default=0.97,
    help="Discount factor (default: 0.97)",
)

parser.add_argument(
    "--choice_price_mult",
    type=float,
    default=1.0,
    help="Choice price multiplier (default: 1.0)",
)

parser.add_argument(
    "--od_price_observe",
    action="store_true",
    default=False,
    help="Use OD price matrices instead of aggregated prices per region (default: False)",
)

parser.add_argument(
    "--od_price_actions",
    action="store_true",
    default=False,
    help="Use OD-based price scalars (N×N outputs) instead of origin-based (N outputs) (default: False)",
)

parser.add_argument(
    "--reward_scalar",
    type=float,
    default=2000.0,
    help="Reward scaling factor (default: 2000.0)",
)

parser.add_argument(
    "--fix_baseline",
    action="store_true",
    default=False,
    help="Fix baseline behavior: use base price and initial vehicle distribution (default: False)",
)

# Parser arguments
args = parser.parse_args()

# Automatically enable od_price_observe when od_price_actions is True
if args.od_price_actions and not args.od_price_observe:
    print("=" * 80)
    print("INFO: Automatically enabling --od_price_observe since --od_price_actions is set")
    print("      (OD-based actions require OD-based observations)")
    print("=" * 80)
    args.od_price_observe = True

# Set device
args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# Set city
city = args.city

if not args.test:
    # Set up weights and biases
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    wandb.init(
        project="thesis",
        name=args.checkpoint_path,
        config=args,
    )

    # Create the scenario
    scenario = Scenario(
                json_file=f"data/scenario_{city}.json",
                demand_ratio=demand_ratio[city]*2,
                json_hr=json_hr[city],
                sd=args.seed,
                json_tstep=args.json_tstep,
                tf=args.max_steps,
                impute=args.impute,
                supply_ratio=args.supply_ratio)

    # Create the environment
    env = AMoD(scenario, args.mode, beta=beta[city], jitter=args.jitter, max_wait=args.maxt, choice_price_mult=args.choice_price_mult, seed = args.seed, fix_baseline=args.fix_baseline, choice_intercept=choice_intercept[city], wage=wage[city], od_price_actions=args.od_price_actions)
    
    # Print baseline information
    if args.fix_baseline:
        print("\n" + "="*50)
        print("FIXED BASELINE MODE ACTIVATED")
        print("="*50)
        print("Behavior:")
        print("  - Prices: Always using base price (price scalar = 0.5)")
        print("  - Rebalancing: Always rebalancing to initial distribution")
        initial_vehicles = env.get_initial_vehicles()
        print(f"  - Initial vehicles: {initial_vehicles}")
        print(f"  - Target distribution: {dict(env.initial_acc)}")
        print("="*50 + "\n")
    elif args.mode == 3:
        print("\n" + "="*80)
        print("BASELINE MODE (Mode 3): No learning, fixed policy")
        print("="*80)
        print("- Both agents use BASE PRICE (scalar=0.5)")
        print("- NO rebalancing (vehicles stay where trips end)")
        print("- Provides baseline for comparison")
        print("="*80 + "\n")
    elif args.mode == 4:
        print("\n" + "="*80)
        print("BASELINE MODE (Mode 4): No learning, fixed policy with uniform rebalancing")
        print("="*80)
        print("- Both agents use BASE PRICE (scalar=0.5)")
        print("- UNIFORM rebalancing (distribute vehicles equally across regions)")
        print("- Provides baseline for comparison")
        print("="*80 + "\n")
    else:
        print("\n" + "="*50)
        print("NORMAL TRAINING MODE")
        print("="*50)
        print("Agent will learn both pricing and rebalancing strategies")
        print("="*50 + "\n")

    # Only create model if not in baseline mode (mode 3 or 4)
    if args.mode not in [3, 4]:
        # Calculate input size based on price type
        if args.od_price_observe:
            # OD price matrices: T (future) + 3 (current_avb, queue, demand) + 2*nregion (own and competitor OD prices)
            input_size = args.look_ahead + 3 + env.nregion
        else:
            # Aggregated prices: T (future) + 3 (current_avb, queue, demand) + 2 (own and competitor aggregated prices)
            input_size = args.look_ahead + 4

        # Load the model 
        model = A2C(
                env=env,
                input_size=input_size,
                hidden_size=args.hidden_size,
                device=device,
                p_lr=args.p_lr,
                q_lr=args.q_lr,
                mode = args.mode,
                T = args.look_ahead,
                scale_factor = args.scale_factor,
                json_file=f"data/scenario_{city}.json",
                actor_clip=args.actor_clip,
                critic_clip=args.critic_clip,
                gamma=args.gamma,
                observe_od_prices=args.od_price_observe,
                od_price_actions=args.od_price_actions,
                reward_scale=args.reward_scalar,
                job_id=args.checkpoint_path
            )

        if args.load:
            print("load checkpoint")
            checkpoint_path = f"ckpt/{args.checkpoint_path}_{args.model_type}.pth"
            model.load_checkpoint(path=checkpoint_path)
            print("loaded checkpoint from", checkpoint_path)
        
        model.train()  # set model in train mode
    else:
        model = None  # No model needed for baseline modes

    #Initialize lists for logging
    log = {'train_reward': [], 
            'train_served_demand': [], 
            'train_reb_cost': []}
    train_episodes = args.max_episodes  # set max number of training episodes
    epochs = trange(train_episodes)  # epoch iterator
    best_reward = -np.inf  # set best reward
    best_reward_test = -np.inf  # set best reward

    # Check metrics
    epoch_demand_list = []
    epoch_reward_list = []
    epoch_waiting_list = []
    epoch_servedrate_list = []
    epoch_rebalancing_cost = []
    price_history = []

    # Get initial vehicles
    initial_vehicles = env.get_initial_vehicles()

    for i_episode in epochs:
        obs = env.reset()  # initialize environment

        # Save original demand for reference
        demand_ori = nestdictsum(env.demand)
        if i_episode == train_episodes - 1:
            export = {"demand_ori":copy.deepcopy(env.demand)}
        action_rl = None
        episode_reward = 0
        episode_served_demand = 0
        episode_unserved_demand = 0
        episode_rebalancing_cost = 0
        episode_total_revenue = 0
        episode_operating_cost = 0
        episode_waiting = 0
        # System-level demand tracking (not agent-specific)
        episode_rejected_demand = 0  # Total demand rejected by choice model
        episode_total_demand = 0  # Total demand generated in system
        episode_rejection_rates = []  # Track rejection rate per step
        episode_true_profit = 0
        episode_adjusted_profit = 0
        episode_unprofitable_trips = 0
        actions = []
        actions_price = []  # Track price scalars during episode
        
        # Track concentration parameters and log probabilities per step
        episode_concentrations = []  # List of concentration arrays per step
        episode_logprobs = []  # List of log probabilities per step

        current_eps = []
        done = False
        step = 0

        while not done:
            if env.mode == 0:
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple()
                episode_reward += paxreward
        
                # Select action for rebalancing
                if args.fix_baseline:
                    # Fixed baseline: distribute vehicles uniformly across all regions
                    current_total = dictsum(env.acc, env.time + 1)
                    base_per_region = current_total // env.nregion
                    remainder = current_total % env.nregion
                    # Distribute uniformly with remainder going to first regions
                    desiredAcc = {
                        env.region[i]: base_per_region + (1 if i < remainder else 0)
                        for i in range(env.nregion)
                    }
                else:
                    action_rl, concentration, logprob = model.select_action(obs, return_concentration=True)
                    episode_concentrations.append(concentration)
                    episode_logprobs.append(logprob)
                    # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                    desiredAcc = {env.region[i]: int(action_rl[i] *dictsum(env.acc,env.time+1))for i in range(len(env.region))}

                # solve minimum rebalancing distance problem (Step 3 in paper)
                rebAction = solveRebFlow(env, desiredAcc)

                # Take rebalancing action in environment
                new_obs, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
                episode_reward += rebreward
                
                # Only add to model rewards if not in fixed baseline mode
                if not args.fix_baseline:
                    model.rewards.append(paxreward + rebreward)

            elif env.mode == 1:
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                episode_reward += paxreward
                
                # Only add to model rewards if not in fixed baseline mode
                if not args.fix_baseline:
                    model.rewards.append(paxreward)
                
                # Select action for pricing
                if args.fix_baseline:
                    # Fixed baseline: use price scalar of 0.5 (keeps base price)
                    if args.od_price_actions:
                        action_rl = np.full((env.nregion, env.nregion), 0.5)
                    else:
                        action_rl = np.array([0.5] * env.nregion)
                else:
                    action_rl, concentration, logprob = model.select_action(obs, return_concentration=True)
                    episode_concentrations.append(concentration)
                    episode_logprobs.append(logprob)
                    
                env.matching_update()

            elif env.mode == 2:
                # Perform matching with the pricing action (from previous iteration or None for first)
                # Pass the full action - env handles extracting pricing based on od_price_actions
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)

                episode_reward += paxreward
                
                # Select action BEFORE reb_step (while acc[n][time+1] still exists)
                # This matches the multi-agent flow
                if args.fix_baseline:
                    # Fixed baseline: create action with price scalar 0.5 and uniform rebalancing
                    total_vehicles = sum(env.initial_acc.values())
                    reb_action_prop = np.array([
                        env.initial_acc[env.region[i]] / total_vehicles 
                        for i in range(env.nregion)
                    ])
                    if args.od_price_actions:
                        # OD Mode 2 action shape: [nregion, nregion+1] where [:, :nregion] = OD prices, [:, -1] = reb
                        action_rl = np.column_stack([
                            np.full((env.nregion, env.nregion), 0.5),  # OD prices
                            reb_action_prop.reshape(-1, 1)  # Rebalancing
                        ])
                    else:
                        # Origin Mode 2 action shape: [nregion, 2] where [:, 0] = price scalar, [:, 1] = reb action
                        action_rl = np.column_stack([
                            np.array([0.5] * env.nregion),  # Price scalar = 0.5 (base price)
                            reb_action_prop  # Rebalancing proportions from initial distribution
                        ])
                else:
                    action_rl, concentration, logprob = model.select_action(obs, return_concentration=True)
                    episode_concentrations.append(concentration)
                    episode_logprobs.append(logprob)

                # Compute desired accumulation for rebalancing
                if args.fix_baseline:
                    # Distribute vehicles uniformly across all regions
                    current_total = dictsum(env.acc, env.time + 1)
                    base_per_region = current_total // env.nregion
                    remainder = current_total % env.nregion
                    desiredAcc = {
                        env.region[i]: base_per_region + (1 if i < remainder else 0)
                        for i in range(env.nregion)
                    }
                else:
                    # Use the newly selected action for rebalancing
                    desiredAcc = {
                        env.region[i]: int(
                            action_rl[i][-1] * dictsum(env.acc, env.time + 1))
                        for i in range(len(env.region))
                    }

                # solve minimum rebalancing distance problem (Step 3 in paper)
                rebAction = solveRebFlow(env, desiredAcc)
                # Take rebalancing action in environment
                new_obs, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
                episode_reward += rebreward
                
                # Only add to model rewards if not in fixed baseline mode
                if not args.fix_baseline:
                    model.rewards.append(paxreward + rebreward)
                
                # Update obs for next iteration
                obs = new_obs
            
            elif env.mode == 3:
                # === BASELINE MODE: No rebalancing, fixed prices ===
                # Use fixed price (scalar = 0.5)
                action_rl = np.array([0.5] * env.nregion)
                
                # Matching step with fixed prices
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                
                # Track rewards (no rebalancing cost in baseline)
                episode_reward += paxreward
                
                # NO rebalancing step - just update vehicle arrivals from completed passenger trips
                env.matching_update()
            
            elif env.mode == 4:
                # === BASELINE MODE: Uniform rebalancing, fixed prices ===
                # Use fixed price (scalar = 0.5)
                action_rl = np.array([0.5] * env.nregion)
                
                # Matching step with fixed prices
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                
                # Track rewards
                episode_reward += paxreward
                
                # Uniform rebalancing: distribute vehicles equally across all regions
                current_total = dictsum(env.acc, env.time + 1)
                base_per_region = current_total // env.nregion
                remainder = current_total % env.nregion
                desiredAcc = {
                    env.region[i]: base_per_region + (1 if i < remainder else 0)
                    for i in range(env.nregion)
                }
                
                # Solve rebalancing flows
                rebAction = solveRebFlow(env, desiredAcc)
                
                # Execute rebalancing step
                _, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
                episode_reward += rebreward
            
            else:
                raise ValueError("Only mode 0, 1, 2, 3, and 4 are allowed")
            
            # track performance over episode
            episode_served_demand += info["served_demand"]
            episode_unserved_demand += info["unserved_demand"]
            episode_rebalancing_cost += info["rebalancing_cost"]
            episode_total_revenue += info["revenue"]
            episode_operating_cost += info["operating_cost"]
            episode_waiting += info['served_waiting']
            episode_true_profit += info['true_profit']
            episode_adjusted_profit += info['adjusted_profit']
            episode_unprofitable_trips += info['unprofitable_trips']
            
            # Track system-level metrics (not agent-specific)
            episode_rejected_demand += system_info["rejected_demand"]
            episode_total_demand += system_info["total_demand"]
            episode_rejection_rates.append(system_info["rejection_rate"])
            
            # Track actions and prices
            if not args.fix_baseline:
                actions.append(action_rl)
            
            # Track mean price scalar (modes 1-4)
            if args.mode in [1, 3, 4]:
                actions_price.append(np.mean(2 * np.array(action_rl)))
            elif args.mode == 2:
                if args.od_price_actions:
                    actions_price.append(np.mean(2 * np.array(action_rl)[:, :env.nregion]))
                else:
                    actions_price.append(np.mean(2 * np.array(action_rl)[:, 0]))

            step += 1

        # Only perform training step if not in baseline mode
        if args.mode not in [3, 4] and not args.fix_baseline:
            # Determine if we should update actor based on warmup period
            update_actor = (i_episode >= args.critic_warmup_episodes)
            
            grad_norms = model.training_step(update_actor=update_actor)  # update model after episode and get metrics
        else:
            # In baseline mode, provide dummy grad norms for logging
            grad_norms = {
                "actor_grad_norm": 0.0,
                "critic_grad_norm": 0.0,
                "actor_loss": 0.0,
                "critic_loss": 0.0
            }

        # Get total vehicles for verification
        total_vehicles = env.get_total_vehicles()

        # Calculate vehicle discrepancy
        vehicle_discrepancy = abs(total_vehicles - initial_vehicles)
        
        # Calculate mean price scalar for the episode (modes 1 and 2 only)
        mean_price_scalar = 0
        if args.mode != 0 and len(actions_price) > 0:
            mean_price_scalar = np.mean(actions_price)
        
        # Add training metrics to wandb (organized into sections)
        log_dict = {
            "episode": i_episode + 1,
            
            # Rewards section
            "rewards/episode_reward": episode_reward,
            "rewards/true_profit": episode_true_profit,
            "rewards/adjusted_profit": episode_adjusted_profit,
            
            # Demand section
            "demand/served": episode_served_demand,
            "demand/unserved": episode_unserved_demand,
            "demand/total_demand": episode_total_demand,
            "demand/rejected": episode_rejected_demand,
            "demand/rejection_rate": np.mean(episode_rejection_rates) if len(episode_rejection_rates) > 0 else 0,
            "demand/unprofitable_trips": episode_unprofitable_trips,
            
            # Revenue and costs section
            "revenue_costs/total_revenue": episode_total_revenue,
            "revenue_costs/operating_cost": episode_operating_cost,
            "revenue_costs/rebalancing_cost": episode_rebalancing_cost,
            
            # Service quality section
            "service/waiting_time": episode_waiting/episode_served_demand if episode_served_demand > 0 else 0,
            
            # Training metrics section
            "training/actor_loss": grad_norms['actor_loss'],
            "training/advantage_mean": grad_norms.get('advantage_mean', 0.0),
            "training/advantage_std": grad_norms.get('advantage_std', 0.0),
            "training/critic_loss": grad_norms['critic_loss'],
            "training/actor_grad_norm": grad_norms['actor_grad_norm'],
            "training/reward_scale": model.reward_scale if args.mode not in [3, 4] and not args.fix_baseline else 1000.0,
            "training/critic_grad_norm": grad_norms['critic_grad_norm'],
            
            # Vehicles section
            "vehicles/total": total_vehicles,
            "vehicles/initial": initial_vehicles,
            "vehicles/discrepancy": vehicle_discrepancy,
        }
        
        # Add warmup tracking
        if args.critic_warmup_episodes > 0:
            log_dict["training/critic_warmup_active"] = 1 if i_episode < args.critic_warmup_episodes else 0
            log_dict["training/warmup_progress"] = min(1.0, i_episode / args.critic_warmup_episodes)
        
        # Add price scalar only for modes 1 and 2
        if args.mode != 0:
            log_dict["pricing/mean_price_scalar"] = mean_price_scalar
        
        # Add concentration parameters and log probabilities tracking (skip baseline modes)
        if args.mode not in [3, 4] and not args.fix_baseline and len(episode_concentrations) > 0:
            # Mean log probability across the episode
            log_dict["training/mean_log_prob"] = np.mean(episode_logprobs)
            
            if args.mode == 0:
                # Mode 0: Dirichlet - shape [batch, nregion, 1]
                concentrations_array = np.array(episode_concentrations)
                # Track mean concentration (alpha) per region
                mean_concentrations = concentrations_array.mean(axis=0).squeeze()  # [nregion, 1] -> [nregion]
                for region_idx in range(env.nregion):
                    log_dict[f"concentration/region_{region_idx}_alpha"] = mean_concentrations[region_idx]
                # Track overall mean
                log_dict["concentration/mean_alpha"] = mean_concentrations.mean()
                
            elif args.mode == 1:
                if args.od_price_actions:
                    # Mode 1 OD: Beta - shape [steps, 1, nregion, nregion, 2]
                    concentrations_array = np.array(episode_concentrations)
                    mean_concentrations = concentrations_array.mean(axis=0).squeeze()  # [nregion, nregion, 2]
                    log_dict["concentration/mean_alpha"] = mean_concentrations[:, :, 0].mean()
                    log_dict["concentration/mean_beta"] = mean_concentrations[:, :, 1].mean()
                else:
                    # Mode 1: Beta - shape [batch, nregion, 2]
                    concentrations_array = np.array(episode_concentrations)
                    # Track mean alpha and beta per region
                    mean_concentrations = concentrations_array.mean(axis=0).squeeze()  # [nregion, 2]
                    for region_idx in range(env.nregion):
                        log_dict[f"concentration/region_{region_idx}_alpha"] = mean_concentrations[region_idx, 0]
                        log_dict[f"concentration/region_{region_idx}_beta"] = mean_concentrations[region_idx, 1]
                    # Track overall means
                    log_dict["concentration/mean_alpha"] = mean_concentrations[:, 0].mean()
                    log_dict["concentration/mean_beta"] = mean_concentrations[:, 1].mean()
                
            elif args.mode == 2:
                if args.od_price_actions:
                    # Mode 2 OD: concentrations are dicts {'beta': [1, N, N, 2], 'dirichlet': [1, N, 1]}
                    beta_arrays = np.array([c['beta'] for c in episode_concentrations])
                    dirichlet_arrays = np.array([c['dirichlet'] for c in episode_concentrations])
                    mean_beta = beta_arrays.mean(axis=0).squeeze()  # [N, N, 2]
                    mean_dirichlet = dirichlet_arrays.mean(axis=0).squeeze()  # [N]
                    log_dict["concentration/mean_price_alpha"] = mean_beta[:, :, 0].mean()
                    log_dict["concentration/mean_price_beta"] = mean_beta[:, :, 1].mean()
                    log_dict["concentration/mean_reb_alpha"] = mean_dirichlet.mean()
                else:
                    # Mode 2: Beta + Dirichlet - shape [batch, nregion, 3]
                    concentrations_array = np.array(episode_concentrations)
                    # Track pricing (alpha, beta) and rebalancing (alpha_reb) per region
                    mean_concentrations = concentrations_array.mean(axis=0).squeeze()  # [nregion, 3]
                    for region_idx in range(env.nregion):
                        log_dict[f"concentration/region_{region_idx}_price_alpha"] = mean_concentrations[region_idx, 0]
                        log_dict[f"concentration/region_{region_idx}_price_beta"] = mean_concentrations[region_idx, 1]
                        log_dict[f"concentration/region_{region_idx}_reb_alpha"] = mean_concentrations[region_idx, 2]
                    # Track overall means
                    log_dict["concentration/mean_price_alpha"] = mean_concentrations[:, 0].mean()
                    log_dict["concentration/mean_price_beta"] = mean_concentrations[:, 1].mean()
                    log_dict["concentration/mean_reb_alpha"] = mean_concentrations[:, 2].mean()
        
        wandb.log(log_dict)

        # Keep metrics
        epoch_reward_list.append(episode_reward)
        epoch_demand_list.append(env.arrivals)
        epoch_waiting_list.append(episode_waiting/episode_served_demand)
        epoch_servedrate_list.append(episode_served_demand/env.arrivals)
        epoch_rebalancing_cost.append(episode_rebalancing_cost)

        # Keep price (only needed for pricing training)
        price_history.append(actions)

        epochs.set_description(
                    f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | Grad Norms: Actor={grad_norms['actor_grad_norm']:.2f}, Critic={grad_norms['critic_grad_norm']:.2f}| Loss: Actor ={grad_norms['actor_loss']:.2f}, Critic={grad_norms['critic_loss']:.2f}"
                )
        # Checkpoint best performing model (skip for baseline modes 3 and 4)
        if args.mode not in [3, 4]:
            if episode_reward >= best_reward:
                model.save_checkpoint(
                    path=f"ckpt/{args.checkpoint_path}_sample.pth")
                best_reward = episode_reward
            model.save_checkpoint(path=f"ckpt/{args.checkpoint_path}_running.pth")
            if i_episode % 100 == 0:
                    model.eval()
                    test_reward, test_served_demand, test_rebalancing_cost = model.test_agent(
                        10, env, args.cplexpath, args.directory)
                    model.train()

                    if test_reward >= best_reward_test:
                        best_reward_test = test_reward
                        model.save_checkpoint(
                            path=f"ckpt/{args.checkpoint_path}_test.pth")
    
    wandb.finish()
    metricPath = f"{args.directory}/train_logs/"
    if not os.path.exists(metricPath):
        os.makedirs(metricPath)
    np.save(f"{args.directory}/train_logs/{city}_rewards_waiting_mode{args.mode}_{train_episodes}.npy", np.array([epoch_reward_list,epoch_waiting_list,epoch_servedrate_list,epoch_demand_list,epoch_rebalancing_cost]))
    np.save(f"{args.directory}/train_logs/{city}_price_mode{args.mode}_{train_episodes}.npy", np.array(price_history))

    export["avail_distri"] = env.acc
    export["demand_scaled"] = env.demand
    with open(f"{args.directory}/train_logs/{city}_export_mode{args.mode}_{train_episodes}.pickle", 'wb') as f:
        pickle.dump(export, f)

else:
    # Create the scenario
    scenario = Scenario(
                json_file=f"data/scenario_{city}.json",
                demand_ratio=demand_ratio[city]*2,
                json_hr=json_hr[city],
                sd=args.seed,
                json_tstep=args.json_tstep,
                tf=args.max_steps,
                impute=args.impute,
                supply_ratio=args.supply_ratio)
    
    # Create the environment
    env = AMoD(scenario, args.mode, beta=beta[city], jitter=args.jitter, max_wait=args.maxt, choice_price_mult=args.choice_price_mult, seed = args.seed, fix_baseline=args.fix_baseline, choice_intercept=choice_intercept[city], wage=wage[city], od_price_actions=args.od_price_actions)

    # Only create model if not in baseline mode (mode 3 or 4)
    if args.mode not in [3, 4]:
        # Calculate input size based on price type
        if args.od_price_observe:
            # OD price matrices: T (future) + 3 (current_avb, queue, demand) + 2*nregion (own and competitor OD prices)
            input_size = args.look_ahead + 3 + env.nregion
        else:
            # Aggregated prices: T (future) + 3 (current_avb, queue, demand) + 2 (own and competitor aggregated prices)
            input_size = args.look_ahead + 4

        # Load the model 
        model = A2C(
                env=env,
                input_size=input_size,
                hidden_size=args.hidden_size,
                device=device,
                p_lr=args.p_lr,
                q_lr=args.q_lr,
                mode = args.mode,
                T = args.look_ahead,
                scale_factor = args.scale_factor,
                json_file=f"data/scenario_{city}.json",
                actor_clip=args.actor_clip,
                critic_clip=args.critic_clip,
                gamma=args.gamma,
                observe_od_prices=args.od_price_observe,
                od_price_actions=args.od_price_actions,
                reward_scale=args.reward_scalar,
                job_id=args.checkpoint_path
            )

        if args.load:
            print("load checkpoint")
            checkpoint_path = f"ckpt/{args.checkpoint_path}_{args.model_type}.pth"
            model.load_checkpoint(path=checkpoint_path)
            print("loaded checkpoint from", checkpoint_path)
        
        model.eval()  # set model in evaluation mode for testing
    elif args.mode == 3:
        print("\n" + "="*80)
        print("TEST MODE - BASELINE (Mode 3): No learning, fixed policy")
        print("="*80)
        print("- Using BASE PRICE (scalar=0.5)")
        print("- NO rebalancing (vehicles stay where trips end)")
        print("- Provides baseline for comparison")
        print("="*80 + "\n")
        model = None
    elif args.mode == 4:
        print("\n" + "="*80)
        print("TEST MODE - BASELINE (Mode 4): No learning, fixed policy with uniform rebalancing")
        print("="*80)
        print("- Using BASE PRICE (scalar=0.5)")
        print("- UNIFORM rebalancing (distribute vehicles equally across regions)")
        print("- Provides baseline for comparison")
        print("="*80 + "\n")
        model = None

    test_episodes = args.max_episodes  # set max number of training episodes
    epochs = trange(test_episodes)  # epoch iterator
    best_reward = -np.inf  # set best reward
    best_reward_test = -np.inf  # set best reward

     # Initialize lists for logging
    log = {"test_reward": [], "test_served_demand": [], "test_reb_cost": []}

    rewards = []
    demands = []
    costs = []
    arrivals = []

    demand_original_steps = []
    demand_scaled_steps = []
    reb_steps = []
    reb_ori_steps = []
    reb_num = []
    pax_steps = []
    pax_wait = []
    actions_step = []
    price_mean = []
    available_steps = []
    rebalancing_cost_steps = []
    price_original_steps = []
    queue_steps = []
    waiting_steps = []

    for episode in range(10):
        actions = []
        actions_price = []
        rebalancing_cost = []
        rebalancing_num = []
        queue = []

        episode_reward = 0
        episode_served_demand = 0
        episode_price = []
        episode_rebalancing_cost = 0
        episode_waiting = 0
        obs = env.reset()
        # Original demand and price
        demand_original_steps.append(env.demand)
        price_original_steps.append(env.price)

        action_rl = None       
        done = False

        while not done:
            if env.mode == 0:
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple()
                episode_reward += paxreward
        
                action_rl = model.select_action(obs, deterministic=True)  # Choose an action
                actions.append(action_rl)

                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                desiredAcc = {env.region[i]: int(action_rl[i] *dictsum(env.acc,env.time+1))for i in range(len(env.region))}

                # solve minimum rebalancing distance problem (Step 3 in paper)
                rebAction = solveRebFlow(env, desiredAcc)

                # Take rebalancing action in environment
                _, rebreward, done, _, system_info, _, _ = env.reb_step(rebAction)
                episode_reward += rebreward
                
            elif env.mode == 1:
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                episode_reward += paxreward
                action_rl = model.select_action(obs, deterministic=True)  # Choose an action
                env.matching_update()

            elif env.mode == 2:
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                episode_reward += paxreward

                action_rl = model.select_action(obs, deterministic=True)  # Choose an action)

                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                desiredAcc = {
                    env.region[i]: int(
                        action_rl[i][-1] * dictsum(env.acc, env.time + 1))
                    for i in range(len(env.region))
                }

                # solve minimum rebalancing distance problem (Step 3 in paper)
                rebAction = solveRebFlow(env, desiredAcc)

                # Take rebalancing action in environment
                _, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
                episode_reward += rebreward 
            
            elif env.mode == 3:
                # === BASELINE MODE: No rebalancing, fixed prices ===
                # Use fixed price (scalar = 0.5)
                action_rl = np.array([0.5] * env.nregion)
                
                # Matching step with fixed prices
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                
                # Track rewards (no rebalancing cost in baseline)
                episode_reward += paxreward
                
                # NO rebalancing step - just update vehicle arrivals from completed passenger trips
                env.matching_update()
            
            elif env.mode == 4:
                # === BASELINE MODE: Uniform rebalancing, fixed prices ===
                # Use fixed price (scalar = 0.5)
                action_rl = np.array([0.5] * env.nregion)
                
                # Matching step with fixed prices
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                
                # Track rewards
                episode_reward += paxreward
                
                # Uniform rebalancing: distribute vehicles equally across all regions
                current_total = dictsum(env.acc, env.time + 1)
                base_per_region = current_total // env.nregion
                remainder = current_total % env.nregion
                desiredAcc = {
                    env.region[i]: base_per_region + (1 if i < remainder else 0)
                    for i in range(env.nregion)
                }
                
                # Solve rebalancing flows
                rebAction = solveRebFlow(env, desiredAcc)
                
                # Execute rebalancing step
                _, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
                episode_reward += rebreward
            
            else:
                raise ValueError("Only mode 0, 1, 2, 3, and 4 are allowed")
            
            episode_served_demand += info["served_demand"]
            episode_rebalancing_cost += info["rebalancing_cost"]
            episode_waiting += info['served_waiting']
            actions.append(action_rl)
            if args.mode == 1:
                actions_price.append(np.mean(2*np.array(action_rl)))
            elif args.mode == 2:
                if args.od_price_actions:
                    actions_price.append(np.mean(2*np.array(action_rl)[:, :env.nregion]))
                else:
                    actions_price.append(np.mean(2*np.array(action_rl)[:,0]))
            elif args.mode in [3, 4]:
                # Mode 3 and 4: Fixed price scalar of 0.5 (base price = 1.0)
                actions_price.append(np.mean(2*np.array(action_rl)))
            rebalancing_cost.append(info["rebalancing_cost"])
            # queue.append([len(env.queue[i]) for i in sorted(env.queue.keys())])
            queue.append(np.mean([len(env.queue[i]) for i in env.queue.keys()]))
        
        # Send current statistics to screen
        epochs.set_description(
            f"Episode {episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost}"
        )
        # Log KPIs
        demand_scaled_steps.append(env.demand)
        available_steps.append(env.acc)
        reb_steps.append(env.rebFlow)
        reb_ori_steps.append(env.rebFlow_ori)
        pax_steps.append(env.paxFlow)
        pax_wait.append(env.paxWait)
        reb_od = 0
        for (o,d),flow in env.rebFlow.items():
            reb_od += sum(flow.values())
        reb_num.append(reb_od)
        actions_step.append(actions)
        if args.mode != 0:
            price_mean.append(np.mean(actions_price))
        
        rebalancing_cost_steps.append(rebalancing_cost)
        queue_steps.append(np.mean(queue))
        waiting_steps.append(episode_waiting/episode_served_demand * args.json_tstep)

        rewards.append(episode_reward)
        demands.append(episode_served_demand)
        costs.append(episode_rebalancing_cost)
        arrivals.append(env.arrivals)

    # Save metrics file
    np.save(f"{args.directory}/{city}_actions_mode{args.mode}.npy", np.array(actions_step))
    np.save(f"{args.directory}/{city}_queue_mode{args.mode}.npy", np.array(queue_steps))

    if env.mode != 1: 
        with open(f"{args.directory}/{city}_reb_mode{args.mode}.pickle", 'wb') as f:
            pickle.dump(reb_steps, f)
        with open(f"{args.directory}/{city}_reb_ori_mode{args.mode}.pickle", 'wb') as f:
            pickle.dump(reb_ori_steps, f)

    with open(f"{args.directory}/{city}_pax_mode{args.mode}.pickle", 'wb') as f:
        pickle.dump(pax_steps, f)
    with open(f"{args.directory}/{city}_pax_wait_mode{args.mode}.pickle", 'wb') as f:
        pickle.dump(pax_wait, f)                     
    with open(f"{args.directory}/{city}_demand_scaled_mode{args.mode}.pickle", 'wb') as f:
        pickle.dump(demand_scaled_steps, f)    

    print("Rewards (mean, std):", f"{np.mean(rewards):.2f}", f"{np.std(rewards):.2f}")
    print("Served demand (mean, std):", f"{np.mean(demands):.2f}", f"{np.std(demands):.2f}")
    print("Rebalancing cost (mean, std):", f"{np.mean(costs):.2f}", f"{np.std(costs):.2f}")
    print("Waiting time (mean, std):", f"{np.mean(waiting_steps):.2f}", f"{np.std(waiting_steps):.2f}")
    print("Queue length (mean, std):", f"{np.mean(queue_steps):.2f}", f"{np.std(queue_steps):.2f}")
    print("Arrivals (mean, std):", f"{np.mean(arrivals):.2f}", f"{np.std(arrivals):.2f}")
    print("Rebalancing trips (mean, std):", f"{np.mean(reb_num):.2f}", f"{np.std(reb_num):.2f}")
    if args.mode != 0:
        print("Price scalar (mean, std):", f"{np.mean(price_mean):.2f}", f"{np.std(price_mean):.2f}")