import argparse
import copy
import os
import pickle

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from tqdm import trange

from src.algos.a2c_gnn_multi_agent import A2C
from src.algos.reb_flow_solver_multi_agent import solveRebFlow
from src.envs.amod_env_multi import Scenario, AMoD
from src.misc.utils import dictsum, nestdictsum

# Load environment variables from .env file
load_dotenv()

# Calibrated simulation parameters
demand_ratio = {'san_francisco': 2,'nyc_man_south': 1.0, 'washington_dc': 4.2}

json_hr = {'san_francisco':19,'nyc_man_south': 19, 'washington_dc': 19}

beta = {'san_francisco': 0.2,'nyc_man_south': 0.5, 'washington_dc': 0.5}

choice_intercept = {'san_francisco': 14.15, 'nyc_man_south': 9.84, 'washington_dc': 11.75}

wage = {'san_francisco': 17.76, 'nyc_man_south': 22.77, 'washington_dc': 25.26}

test_tstep = {'san_francisco': 3, 'nyc_man_south': 3, 'washington_dc':3}

parser = argparse.ArgumentParser(description="A2C-GNN")

# Simulator parameters
parser.add_argument(
    "--seed", type=int, default=10, metavar="S", help="random seed (default: 10)"
)

parser.add_argument(
    "--model_type",
    type=str,
    default="running", 
    help="Defines the type of model (default: running)",
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
    help='rebalancing mode. (0:manual, 1:pricing, 2:both, 3:baseline (no reb, fixed price), 4:baseline (uniform reb, fixed price). default 2)',
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
    "--load_checkpoint_path",
    type=str,
    default=None,
    help="name of checkpoint file to load from (if different from checkpoint_path). If not specified, uses checkpoint_path (default: None)",
)
parser.add_argument(
    "--load",
    action="store_true",
    default=False,
    help="either to start training from checkpoint (default: False)",
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
    default=50,
    help="number of episodes to train only critic before training actor (default: 50)",
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
    "--agent0_vehicle_ratio",
    type=float,
    default=0.5,
    help="Proportion of vehicles for agent 0 (default: 0.5, range: 0.0-1.0)",
)

parser.add_argument(
    "--total_vehicles",
    type=int,
    default=None,
    help="Total number of vehicles in the system. If None, reads from dataset (default: None)",
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
    "--fix_agent",
    type=int,
    default=2,
    choices=[0, 1, 2],
    help="Fix agent behavior for testing: 0=fix agent 0, 1=fix agent 1, 2=no fixing (default: 2)",
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
    "--no_share_info",
    action="store_true",
    default=False,
    help="Don't share competitor pricing info between agents (default: False)",
)

parser.add_argument(
    "--reward_scalar",
    type=float,
    default=2000.0,
    help="Reward scaling factor (default: 1000.0)",
)

parser.add_argument(
    "--use_dynamic_wage_man_south",
    action="store_true",
    default=False,
    help="Enable region-specific wage distributions for NYC Manhattan South (default: False)",
)

# Parser arguments
args = parser.parse_args()

# Validate agent0_vehicle_ratio
if not (0.0 <= args.agent0_vehicle_ratio <= 1.0):
    raise ValueError(f"agent0_vehicle_ratio must be between 0.0 and 1.0, got {args.agent0_vehicle_ratio}")

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


def test_agents(model_agents, test_episodes, env, cplexpath, directory, max_episodes, mode, fix_agent=2, job_id=None):
        epochs = range(test_episodes)  # epoch iterator
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        for _ in epochs:
            eps_reward = {0: 0, 1: 0}
            eps_served_demand = {0: 0, 1: 0}
            eps_rebalancing_cost = {0: 0, 1: 0}
            obs = env.reset()
            action_rl = [0]*env.nregion
            done = False
            while not done:
                if env.mode == 0:
                    # Make Match Step
                    obs, paxreward, done, info, system_info, _, _ = env.match_step_simple()

                    # Update episode reward
                    eps_reward = {a: eps_reward[a] + paxreward[a] for a in [0, 1]}

                    # Get actions
                    action_rl = {}
                    for a in [0, 1]:
                        if a == fix_agent:
                            # Fixed agent: use actual initial distribution for rebalancing
                            total_vehicles = sum(env.agent_initial_acc[a].values())
                            action_rl[a] = np.array([
                                env.agent_initial_acc[a][env.region[i]] / total_vehicles 
                                for i in range(env.nregion)
                            ])
                        else:
                            action_rl[a] = model_agents[a].select_action(obs[a], deterministic=True)

                    # Compute desired accumulation for all agents
                    desiredAcc = {}
                    for a in [0, 1]:
                        if a == fix_agent:
                            # For fixed agent, set desiredAcc to initial distribution
                            desiredAcc[a] = {env.region[i]: env.agent_initial_acc[a][env.region[i]] for i in range(env.nregion)}
                        else:
                            # For active agent, use action to determine desired distribution
                            desiredAcc[a] = {
                                env.region[i]: int(action_rl[a][i] * dictsum(env.agent_acc[a], env.time + 1))
                                for i in range(env.nregion)
                            }

                    # Compute rebalancing flows for both agents sequentially
                    rebAction = {a: solveRebFlow(env, desiredAcc[a], a) for a in [0, 1]}
                    
                    _, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
                    eps_reward = {a: eps_reward[a] + rebreward[a] for a in [0, 1]}
                    
                    
                elif env.mode == 1:
                    obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)

                    eps_reward = {a: eps_reward[a] + paxreward[a] for a in [0, 1]}

                    # Get actions
                    action_rl = {}
                    for a in [0, 1]:
                        if a == fix_agent:
                            # Fixed agent: environment handles price override to 0.5
                            if env.od_price_actions:
                                action_rl[a] = np.full((env.nregion, env.nregion), 0.5)
                            else:
                                action_rl[a] = np.array([0.5] * env.nregion)
                        else:
                            action_rl[a] = model_agents[a].select_action(obs[a], deterministic=True)

                    # Matching update (global step)
                    env.matching_update()
                
                elif env.mode == 2:
                    # --- Matching step ---
                    obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                    
                    eps_reward = {a: eps_reward[a] + paxreward[a] for a in [0, 1]}

                    # Get actions
                    action_rl = {}
                    for a in [0, 1]:
                        if a == fix_agent:
                            # Fixed agent: environment handles price override to 0.5
                            total_vehicles = sum(env.agent_initial_acc[a].values())
                            reb_action = np.array([
                                env.agent_initial_acc[a][env.region[i]] / total_vehicles 
                                for i in range(env.nregion)
                            ])
                            if env.od_price_actions:
                                # Mode 2 OD action shape: [nregion, nregion+1]
                                action_rl[a] = np.column_stack([
                                    np.full((env.nregion, env.nregion), 0.5),  # OD prices
                                    reb_action.reshape(-1, 1)  # Rebalancing
                                ])
                            else:
                                # Mode 2 action shape: [nregion, 2] where [:, 0] = price, [:, 1] = reb
                                action_rl[a] = np.column_stack([
                                    np.array([0.5] * env.nregion),  # Price (will be overridden by env)
                                    reb_action  # Rebalancing: actual initial distribution
                                ])
                        else:
                            action_rl[a] = model_agents[a].select_action(obs[a], deterministic=True)

                    # --- Desired Acc computation ---
                    # Compute desired accumulation for all agents
                    desiredAcc = {}
                    for a in [0, 1]:
                        if a == fix_agent:
                            # For fixed agent, set desiredAcc to initial distribution
                            desiredAcc[a] = {env.region[i]: env.agent_initial_acc[a][env.region[i]] for i in range(env.nregion)}
                        else:
                            # For active agent, use action to determine desired distribution
                            desiredAcc[a] = {
                                env.region[i]: int(action_rl[a][i, -1] * dictsum(env.agent_acc[a], env.time + 1))
                                for i in range(env.nregion)
                            }
                    
                    # --- Rebalancing step ---
                    # Compute rebalancing flows for both agents sequentially
                    rebAction = {a: solveRebFlow(env, desiredAcc[a], a) for a in [0, 1]}
                
                    _, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
                
                    eps_reward = {a: eps_reward[a] + rebreward[a] for a in [0, 1]}
                
                elif env.mode == 3:
                    # === BASELINE MODE: No rebalancing, fixed prices ===
                    # Use fixed price (scalar = 0.5 for both agents)
                    action_rl = {
                        0: np.array([0.5] * env.nregion),
                        1: np.array([0.5] * env.nregion)
                    }
                    
                    # Matching step with fixed prices
                    obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                    
                    # Track rewards (no rebalancing cost in baseline)
                    eps_reward = {a: eps_reward[a] + paxreward[a] for a in [0, 1]}
                    
                    # NO rebalancing step - just update vehicle arrivals from completed passenger trips
                    env.matching_update()
                
                elif env.mode == 4:
                    # === BASELINE MODE: Uniform rebalancing, fixed prices ===
                    # Use fixed price (scalar = 0.5 for both agents)
                    action_rl = {
                        0: np.array([0.5] * env.nregion),
                        1: np.array([0.5] * env.nregion)
                    }
                    
                    # Matching step with fixed prices
                    obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                    
                    # Track rewards
                    eps_reward = {a: eps_reward[a] + paxreward[a] for a in [0, 1]}
                    
                    # Uniform rebalancing: distribute vehicles equally across all regions
                    desiredAcc = {}
                    for a in [0, 1]:
                        current_total = dictsum(env.agent_acc[a], env.time + 1)
                        base_per_region = current_total // env.nregion
                        remainder = current_total % env.nregion
                        desiredAcc[a] = {
                            env.region[i]: base_per_region + (1 if i < remainder else 0)
                            for i in range(env.nregion)
                        }
                    
                    # Solve rebalancing flows sequentially
                    rebAction = {a: solveRebFlow(env, desiredAcc[a], a) for a in [0, 1]}
                    
                    # Execute rebalancing step
                    _, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
                    eps_reward = {a: eps_reward[a] + rebreward[a] for a in [0, 1]}
                
                else:
                    raise ValueError("Only mode 0, 1, 2, 3, and 4 are allowed")  
                   
                for a in [0, 1]:
                    eps_served_demand[a] += info[a]["served_demand"]
                    eps_rebalancing_cost[a] += info[a]["rebalancing_cost"]
                
            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)

        # Calculate means for each agent
        mean_reward = {
            0: np.mean([ep[0] for ep in episode_reward]),
            1: np.mean([ep[1] for ep in episode_reward])
        }
        mean_served_demand = {
            0: np.mean([ep[0] for ep in episode_served_demand]),
            1: np.mean([ep[1] for ep in episode_served_demand])
        }
        mean_rebalancing_cost = {
            0: np.mean([ep[0] for ep in episode_rebalancing_cost]),
            1: np.mean([ep[1] for ep in episode_rebalancing_cost])
        }
        
        return (
            mean_reward,
            mean_served_demand,
            mean_rebalancing_cost,
        )


# Set up weights and biases (only for training mode)
if not args.test:
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    run = wandb.init(
        project="thesis",
        name=args.checkpoint_path,
        config=args,
    )

# Set city
city = args.city
if not args.test:
    # Create the scenario
    scenario = Scenario(
                json_file=f"data/scenario_{city}.json",
                demand_ratio=demand_ratio[city]*2,
                json_hr=json_hr[city],
                sd=args.seed,
                json_tstep=args.json_tstep,
                tf=args.max_steps,
                impute=args.impute,
                supply_ratio=args.supply_ratio,
                agent0_vehicle_ratio=args.agent0_vehicle_ratio,
                total_vehicles=args.total_vehicles)

    # Create the environment
    env = AMoD(scenario, args.mode, beta=beta[city], jitter=args.jitter, max_wait=args.maxt, choice_price_mult=args.choice_price_mult, seed = args.seed, fix_agent=args.fix_agent, choice_intercept=choice_intercept[city], wage=wage[city], use_dynamic_wage_man_south=args.use_dynamic_wage_man_south, od_price_actions=args.od_price_actions)
    
    # Print fixed agent information
    if args.fix_agent == 0:
        print("=" * 80)
        print("FIXED AGENT MODE: Agent 0 is FIXED")
        print("- Agent 0 uses BASE PRICES (scalar=0.5, no learning)")
        print("- Agent 0 is included in choice model and can receive demand")
        print("- Agent 0 vehicles reset to initial distribution each step")
        print("- Agent 1 is LEARNING (adjusts prices dynamically)")
        print("=" * 80)
    elif args.fix_agent == 1:
        print("=" * 80)
        print("FIXED AGENT MODE: Agent 1 is FIXED")
        print("- Agent 1 uses BASE PRICES (scalar=0.5, no learning)")
        print("- Agent 1 is included in choice model and can receive demand")
        print("- Agent 1 vehicles reset to initial distribution each step")
        print("- Agent 0 is LEARNING (adjusts prices dynamically)")
        print("=" * 80)
    else:
        print("=" * 80)
        print("NORMAL MODE: Both agents are active and learning")
        print("- Demand is split via choice model")
        print("- Both agents learn simultaneously")
        print("=" * 80)

    # Only create models if not in baseline mode (mode 3 or 4)
    if args.mode not in [3, 4]:
        # Calculate input size based on price type
        if args.od_price_observe:
            if args.no_share_info:
                # Only own prices: T (future) + 3 (current_avb, queue, demand) + nregion (own OD prices)
                input_size = args.look_ahead + 3 + env.nregion
            else:
                # OD price matrices: T (future) + 3 (current_avb, queue, demand) + 3*nregion (own, competitor, and difference OD prices)
                input_size = args.look_ahead + 3 + 3 * env.nregion
        else:
            if args.no_share_info:
                # Only own price: T (future) + 3 (current_avb, queue, demand) + 1 (own aggregated price)
                input_size = args.look_ahead + 4
            else:
                # Aggregated prices: T (future) + 3 (current_avb, queue, demand) + 3 (own, competitor, and difference aggregated prices)
                input_size = args.look_ahead + 6
        
        model_agents = {
                a: A2C(
                    env=env,
                    input_size=input_size,
                    hidden_size=args.hidden_size,
                    device=device,
                    p_lr=args.p_lr,
                    q_lr=args.q_lr,
                    T = args.look_ahead,
                    scale_factor = args.scale_factor,
                    json_file=f"data/scenario_{city}.json",
                    mode=args.mode,
                    actor_clip=args.actor_clip,
                    critic_clip=args.critic_clip,
                    gamma=args.gamma,
                    agent_id = a,
                    observe_od_prices = args.od_price_observe,
                    od_price_actions = args.od_price_actions,
                    no_share_info = args.no_share_info,
                    reward_scale=args.reward_scalar,
                )
                for a in [0, 1]
            }
    elif args.mode == 3:
        print("=" * 80)
        print("BASELINE MODE (Mode 3): No learning, fixed policy")
        print("- Both agents use BASE PRICE (scalar=0.5)")
        print("- NO rebalancing performed")
        print("- Provides baseline for comparison")
        print("=" * 80)
        model_agents = None  # No models needed
    else:  # mode == 4
        print("=" * 80)
        print("BASELINE MODE (Mode 4): No learning, fixed policy with uniform rebalancing")
        print("- Both agents use BASE PRICE (scalar=0.5)")
        print("- UNIFORM rebalancing (distribute vehicles equally across regions)")
        print("- Provides baseline for comparison")
        print("=" * 80)
        model_agents = None  # No models needed

    if args.load and args.mode not in [3, 4]:
        # Determine which checkpoint path to load from
        load_from = args.load_checkpoint_path if args.load_checkpoint_path is not None else args.checkpoint_path
        print(f"Loading checkpoint from: {load_from}")
        if args.load_checkpoint_path is not None:
            print(f"Will save new checkpoints to: {args.checkpoint_path}")
        for agent_id in [0, 1]:
            checkpoint_path = f"ckpt/{load_from}_agent{agent_id+1}_running.pth"
            model_agents[agent_id].load_checkpoint(path=checkpoint_path)
            print(f"Loaded checkpoint for agent {agent_id} from {checkpoint_path}")
        print("Loaded models from checkpoint successfully")

    train_episodes = args.max_episodes  # set max number of training episodes
    epochs = trange(train_episodes)  # epoch iterator

    best_reward = -np.inf  # set best training reward
    best_reward_test = -np.inf  # set best test reward

    # Only train agents if not in baseline mode
    if args.mode not in [3, 4]:
        for agent_id in [0, 1]:
            model_agents[agent_id].train()
        
        # Print critic warmup information
        if args.critic_warmup_episodes > 0:
            print("=" * 80)
            print(f"CRITIC WARMUP ENABLED: {args.critic_warmup_episodes} episodes")
            print(f"- Episodes 0-{args.critic_warmup_episodes-1}: Critic only (actor frozen)")
            print(f"- Episodes {args.critic_warmup_episodes}+: Both actor and critic training")
            print("=" * 80 + "\n")
    else:
        # Mode 3 or 4: Baseline modes - skip model training, run simulation with fixed policy
        print("\nRunning baseline mode with fixed policy...")
        print("No training will be performed. Running test episodes only.\n")

    # Check metrics
    epoch_demand_list = []
    epoch_reward_list = []
    epoch_waiting_list = []
    epoch_servedrate_list = []
    epoch_rebalancing_cost = []

    # Get initial vehicles
    initial_vehicles = env.get_initial_vehicles()

    for i_episode in epochs:
        obs = env.reset()  # initialize environment

        action_rl = {
                a: [0.0] * env.nregion for a in [0, 1]
            }
        
        # Save original demand for reference
        demand_ori = nestdictsum(env.demand)

        if i_episode == train_episodes - 1:
            export = {"demand_ori":copy.deepcopy(env.demand)}

        episode_reward = {0: 0, 1: 0}
        episode_served_demand = {0: 0, 1: 0}
        episode_unserved_demand = {0: 0, 1: 0}
        episode_rebalancing_cost = {0: 0, 1: 0}
        episode_total_revenue = {0: 0, 1: 0}
        episode_total_operating_cost = {0: 0, 1: 0}
        episode_waiting = {0: 0, 1: 0}
        # System-level demand tracking (not agent-specific)
        episode_rejected_demand = 0  # Total demand rejected by choice model
        episode_total_demand = 0  # Total demand generated in system
        episode_rejection_rates = []  # Track rejection rate per step
        # Profitability tracking
        episode_true_profit = {0: 0, 1: 0}
        episode_adjusted_profit = {0: 0, 1: 0}
        episode_unprofitable_trips = {0: 0, 1: 0}
        actions_price = {0: [], 1: []}  # Track price scalars during episode
        
        # Track concentration parameters during episode (different structures per mode)
        if env.mode == 0:
            # Mode 0: Only Dirichlet for rebalancing
            actions_concentration_dirichlet = {0: [], 1: []}
            # Track episode-level min/max across all regions and timesteps
            episode_min_concentration_dirichlet = {0: float('inf'), 1: float('inf')}
            episode_max_concentration_dirichlet = {0: float('-inf'), 1: float('-inf')}
        elif env.mode == 1:
            # Mode 1: Only Beta (alpha, beta) for pricing
            actions_concentration_alpha = {0: [], 1: []}
            actions_concentration_beta = {0: [], 1: []}
            episode_min_concentration_alpha = {0: float('inf'), 1: float('inf')}
            episode_max_concentration_alpha = {0: float('-inf'), 1: float('-inf')}
            episode_min_concentration_beta = {0: float('inf'), 1: float('inf')}
            episode_max_concentration_beta = {0: float('-inf'), 1: float('-inf')}
        else:  # mode 2
            # Mode 2: Beta (alpha, beta) for pricing + Dirichlet for rebalancing
            actions_concentration_alpha = {0: [], 1: []}
            actions_concentration_beta = {0: [], 1: []}
            actions_concentration_dirichlet = {0: [], 1: []}
            episode_min_concentration_alpha = {0: float('inf'), 1: float('inf')}
            episode_max_concentration_alpha = {0: float('-inf'), 1: float('-inf')}
            episode_min_concentration_beta = {0: float('inf'), 1: float('inf')}
            episode_max_concentration_beta = {0: float('-inf'), 1: float('-inf')}
            episode_min_concentration_dirichlet = {0: float('inf'), 1: float('inf')}
            episode_max_concentration_dirichlet = {0: float('-inf'), 1: float('-inf')}
        
        # Track log probabilities during episode
        episode_logprobs = {0: [], 1: []}

        done = False
        step = 0

        while not done:
            if env.mode == 0:
                # Make Match Step
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple()

                # Update episode reward
                episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}

                # Get actions and concentrations
                action_rl = {}
                concentrations = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent: use actual initial distribution for rebalancing
                        # Convert initial vehicle counts to proportions
                        total_vehicles = sum(env.agent_initial_acc[a].values())
                        action_rl[a] = np.array([
                            env.agent_initial_acc[a][env.region[i]] / total_vehicles 
                            for i in range(env.nregion)
                        ])
                        concentrations[a] = np.zeros((env.nregion, 1))  # Dummy for tracking
                    else:
                        action_rl[a], concentrations[a], logprob = model_agents[a].select_action(obs[a], return_concentration=True)
                        episode_logprobs[a].append(logprob)
                
                # Track concentration (mode 0: Dirichlet concentration for rebalancing)
                for a in [0, 1]:
                    if a != args.fix_agent:
                        actions_concentration_dirichlet[a].append(np.mean(concentrations[a]))
                        # Update episode-level min/max
                        episode_min_concentration_dirichlet[a] = min(episode_min_concentration_dirichlet[a], np.min(concentrations[a]))
                        episode_max_concentration_dirichlet[a] = max(episode_max_concentration_dirichlet[a], np.max(concentrations[a]))

                # Determine which agents are active (not fixed)
                # Compute desired accumulation for all agents
                desiredAcc = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # For fixed agent, distribute vehicles uniformly across all regions
                        current_total = dictsum(env.agent_acc[a], env.time + 1)
                        base_per_region = current_total // env.nregion
                        remainder = current_total % env.nregion
                        # Distribute uniformly with remainder going to first regions
                        desiredAcc[a] = {
                            env.region[i]: base_per_region + (1 if i < remainder else 0)
                            for i in range(env.nregion)
                        }
                    else:
                        # For active agent, use action to determine desired distribution
                        desiredAcc[a] = {
                            env.region[i]: int(action_rl[a][i] * dictsum(env.agent_acc[a], env.time + 1))
                            for i in range(env.nregion)
                        }

                # Compute rebalancing flows for both agents sequentially
                rebAction = {a: solveRebFlow(env, desiredAcc[a], a) for a in [0, 1]}
                
                new_obs, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
                episode_reward = {a: episode_reward[a] + rebreward[a] for a in [0, 1]}
                
                for agent_id in [0, 1]:
                    model_agents[agent_id].rewards.append((paxreward[agent_id] + rebreward[agent_id]))

            elif env.mode == 1:
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)

                episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}

                for agent_id in [0, 1]:
                    model_agents[agent_id].rewards.append(paxreward[agent_id])

                # Get actions and concentrations
                action_rl = {}
                concentrations = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent: environment handles price override to 0.5
                        # Just provide any valid pricing action (will be ignored)
                        if args.od_price_actions:
                            action_rl[a] = np.full((env.nregion, env.nregion), 0.5)
                        else:
                            action_rl[a] = np.array([0.5] * env.nregion)
                        concentrations[a] = np.zeros((env.nregion, 2))  # Dummy for tracking
                    else:
                        action_rl[a], concentrations[a], logprob = model_agents[a].select_action(obs[a], return_concentration=True)
                        episode_logprobs[a].append(logprob)
                
                # Track prices during episode (mode 1: action_rl is price scalar)
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent always uses 0.5 scalar
                        actions_price[a].append(1.0)  # 2 * 0.5 = 1.0 (base price)
                    else:
                        actions_price[a].append(np.mean(2 * np.array(action_rl[a])))
                
                # Track concentration (mode 1: Beta distribution - alpha and beta)
                for a in [0, 1]:
                    if a != args.fix_agent:
                        if args.od_price_actions:
                            # OD: concentrations shape [1, nregion, nregion, 2]
                            actions_concentration_alpha[a].append(np.mean(concentrations[a][0, :, :, 0]))
                            actions_concentration_beta[a].append(np.mean(concentrations[a][0, :, :, 1]))
                            episode_min_concentration_alpha[a] = min(episode_min_concentration_alpha[a], np.min(concentrations[a][0, :, :, 0]))
                            episode_max_concentration_alpha[a] = max(episode_max_concentration_alpha[a], np.max(concentrations[a][0, :, :, 0]))
                            episode_min_concentration_beta[a] = min(episode_min_concentration_beta[a], np.min(concentrations[a][0, :, :, 1]))
                            episode_max_concentration_beta[a] = max(episode_max_concentration_beta[a], np.max(concentrations[a][0, :, :, 1]))
                        else:
                            # Origin: concentrations shape [1, nregion, 2]
                            actions_concentration_alpha[a].append(np.mean(concentrations[a][0, :, 0]))
                            actions_concentration_beta[a].append(np.mean(concentrations[a][0, :, 1]))
                            episode_min_concentration_alpha[a] = min(episode_min_concentration_alpha[a], np.min(concentrations[a][0, :, 0]))
                            episode_max_concentration_alpha[a] = max(episode_max_concentration_alpha[a], np.max(concentrations[a][0, :, 0]))
                            episode_min_concentration_beta[a] = min(episode_min_concentration_beta[a], np.min(concentrations[a][0, :, 1]))
                            episode_max_concentration_beta[a] = max(episode_max_concentration_beta[a], np.max(concentrations[a][0, :, 1]))

                # Matching update (global step)
                env.matching_update()
            
            elif env.mode == 2:
                # --- Matching step ---
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                
                episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}

                # Get actions and concentrations
                action_rl = {}
                concentrations = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent: environment handles price override to 0.5
                        total_vehicles = sum(env.agent_initial_acc[a].values())
                        reb_action = np.array([
                            env.agent_initial_acc[a][env.region[i]] / total_vehicles 
                            for i in range(env.nregion)
                        ])
                        if args.od_price_actions:
                            # Mode 2 OD action shape: [nregion, nregion+1] where [:, :nregion] = OD prices, [:, -1] = reb
                            action_rl[a] = np.column_stack([
                                np.full((env.nregion, env.nregion), 0.5),  # OD prices (will be overridden)
                                reb_action.reshape(-1, 1)  # Rebalancing
                            ])
                        else:
                            # Mode 2 action shape: [nregion, 2] where [:, 0] = price scalar, [:, 1] = reb action
                            action_rl[a] = np.column_stack([
                                np.array([0.5] * env.nregion),  # Price (will be overridden to 0.5 by env)
                                reb_action  # Rebalancing: actual initial distribution
                            ])
                        concentrations[a] = np.zeros((env.nregion, 3))  # Dummy for tracking
                    else:
                        action_rl[a], concentrations[a], logprob = model_agents[a].select_action(obs[a], return_concentration=True)
                        episode_logprobs[a].append(logprob)
                
                # Track prices during episode (mode 2: price part of action)
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent always uses 0.5 scalar
                        actions_price[a].append(1.0)  # 2 * 0.5 = 1.0 (base price)
                    else:
                        if args.od_price_actions:
                            # OD: action shape [nregion, nregion+1], prices in [:, :nregion]
                            actions_price[a].append(np.mean(2 * np.array(action_rl[a])[:, :env.nregion]))
                        else:
                            actions_price[a].append(np.mean(2 * np.array(action_rl[a])[:, 0]))
                
                # Track concentration (mode 2: Beta + Dirichlet)
                for a in [0, 1]:
                    if a != args.fix_agent:
                        if args.od_price_actions:
                            # OD: concentrations is dict {'beta': [1, nregion, nregion, 2], 'dirichlet': [1, nregion, 1]}
                            actions_concentration_alpha[a].append(np.mean(concentrations[a]['beta'][0, :, :, 0]))
                            actions_concentration_beta[a].append(np.mean(concentrations[a]['beta'][0, :, :, 1]))
                            actions_concentration_dirichlet[a].append(np.mean(concentrations[a]['dirichlet'][0, :, 0]))
                            episode_min_concentration_alpha[a] = min(episode_min_concentration_alpha[a], np.min(concentrations[a]['beta'][0, :, :, 0]))
                            episode_max_concentration_alpha[a] = max(episode_max_concentration_alpha[a], np.max(concentrations[a]['beta'][0, :, :, 0]))
                            episode_min_concentration_beta[a] = min(episode_min_concentration_beta[a], np.min(concentrations[a]['beta'][0, :, :, 1]))
                            episode_max_concentration_beta[a] = max(episode_max_concentration_beta[a], np.max(concentrations[a]['beta'][0, :, :, 1]))
                            episode_min_concentration_dirichlet[a] = min(episode_min_concentration_dirichlet[a], np.min(concentrations[a]['dirichlet'][0, :, 0]))
                            episode_max_concentration_dirichlet[a] = max(episode_max_concentration_dirichlet[a], np.max(concentrations[a]['dirichlet'][0, :, 0]))
                        else:
                            # Origin: concentrations[a] has shape (1, nregion, 3)
                            actions_concentration_alpha[a].append(np.mean(concentrations[a][0, :, 0]))
                            actions_concentration_beta[a].append(np.mean(concentrations[a][0, :, 1]))
                            actions_concentration_dirichlet[a].append(np.mean(concentrations[a][0, :, 2]))
                            episode_min_concentration_alpha[a] = min(episode_min_concentration_alpha[a], np.min(concentrations[a][0, :, 0]))
                            episode_max_concentration_alpha[a] = max(episode_max_concentration_alpha[a], np.max(concentrations[a][0, :, 0]))
                            episode_min_concentration_beta[a] = min(episode_min_concentration_beta[a], np.min(concentrations[a][0, :, 1]))
                            episode_max_concentration_beta[a] = max(episode_max_concentration_beta[a], np.max(concentrations[a][0, :, 1]))
                            episode_min_concentration_dirichlet[a] = min(episode_min_concentration_dirichlet[a], np.min(concentrations[a][0, :, 2]))
                            episode_max_concentration_dirichlet[a] = max(episode_max_concentration_dirichlet[a], np.max(concentrations[a][0, :, 2]))
                    
                # --- Desired Acc computation ---
                # Compute desired accumulation for all agents
                desiredAcc = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # For fixed agent, distribute vehicles uniformly across all regions
                        current_total = dictsum(env.agent_acc[a], env.time + 1)
                        base_per_region = current_total // env.nregion
                        remainder = current_total % env.nregion
                        # Distribute uniformly with remainder going to first regions
                        desiredAcc[a] = {
                            env.region[i]: base_per_region + (1 if i < remainder else 0)
                            for i in range(env.nregion)
                        }
                    else:
                        # For active agent, use action to determine desired distribution
                        desiredAcc[a] = {
                            env.region[i]: int(action_rl[a][i, -1] * dictsum(env.agent_acc[a], env.time + 1))
                            for i in range(env.nregion)
                        }
                
                # --- Rebalancing step ---
                # Compute rebalancing flows for both agents sequentially
                rebAction = {a: solveRebFlow(env, desiredAcc[a], a) for a in [0, 1]}
            
                new_obs, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
            
                episode_reward = {a: episode_reward[a] + rebreward[a] for a in [0, 1]}
                for agent_id in [0, 1]:
                    model_agents[agent_id].rewards.append((paxreward[agent_id] + rebreward[agent_id]))
            
            elif env.mode == 3:
                # === BASELINE MODE: No rebalancing, fixed prices ===
                # Use fixed price (scalar = 0.5 for both agents)
                action_rl = {
                    0: np.array([0.5] * env.nregion),
                    1: np.array([0.5] * env.nregion)
                }
                
                # Matching step with fixed prices
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                
                # Track rewards (no rebalancing cost in baseline)
                episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}
                
                # NO rebalancing step - just update vehicle arrivals from completed passenger trips
                env.matching_update()
            
            elif env.mode == 4:
                # === BASELINE MODE 4: Uniform rebalancing, fixed prices ===
                # Use fixed price (scalar = 0.5 for both agents)
                action_rl = {
                    0: np.array([0.5] * env.nregion),
                    1: np.array([0.5] * env.nregion)
                }
                
                # Matching step with fixed prices
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                
                # Track rewards
                episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}
                
                # UNIFORM rebalancing: distribute vehicles equally across all regions
                desiredAcc = {}
                for a in [0, 1]:
                    # Calculate total available vehicles for this agent
                    current_total = dictsum(env.agent_acc[a], env.time + 1)
                    base_per_region = current_total // env.nregion
                    remainder = current_total % env.nregion
                    # Distribute uniformly with remainder going to first regions
                    desiredAcc[a] = {
                        env.region[i]: base_per_region + (1 if i < remainder else 0)
                        for i in range(env.nregion)
                    }
                
                # Compute rebalancing flows for both agents sequentially
                rebAction = {a: solveRebFlow(env, desiredAcc[a], a) for a in [0, 1]}
                
                new_obs, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
                episode_reward = {a: episode_reward[a] + rebreward[a] for a in [0, 1]}
            
            else:
                raise ValueError("Only mode 0, 1, 2, 3, and 4 are allowed")

            # Track agent-specific metrics
            for a in [0, 1]:
                    episode_served_demand[a] += info[a]["served_demand"]
                    episode_unserved_demand[a] += info[a]["unserved_demand"]
                    episode_rebalancing_cost[a] += info[a]["rebalancing_cost"]
                    episode_total_revenue[a] += info[a]["revenue"]
                    episode_total_operating_cost[a] += info[a]["operating_cost"]
                    episode_waiting[a] += info[a]["served_waiting"]
                    # Track profitability metrics
                    episode_true_profit[a] += info[a].get("true_profit", 0)
                    episode_adjusted_profit[a] += info[a].get("adjusted_profit", 0)
                    episode_unprofitable_trips[a] += info[a].get("unprofitable_trips", 0)
            
            # Track system-level metrics (not agent-specific)
            episode_rejected_demand += system_info["rejected_demand"]
            episode_total_demand += system_info["total_demand"]
            episode_rejection_rates.append(system_info["rejection_rate"])
        
            step += 1
        
        # Update both agent models after episode and collect training metrics
        grad_norms = {}
        if args.mode not in [3, 4]:
            # Determine if we should update actor based on warmup period
            update_actor = (i_episode >= args.critic_warmup_episodes)
            
            # Normal modes: update models
            for a in [0, 1]:
                if a == args.fix_agent:
                    # Fixed agent: skip learning, return dummy metrics (same as single agent)
                    grad_norms[a] = {
                        "actor_grad_norm": 0.0,
                        "critic_grad_norm": 0.0,
                        "actor_loss": 0.0,
                        "critic_loss": 0.0,
                        "advantage_mean": 0.0,
                        "advantage_std": 0.0,
                    }
                    # Clear the fixed agent's buffers without updating
                    model_agents[a].rewards = []
                    model_agents[a].saved_actions = []
                else:
                    grad_norms[a] = model_agents[a].training_step(update_actor=update_actor)  # update model after episode and get metrics
        else:
            # Baseline modes (mode 3 or 4): no training, return dummy metrics for both agents
            for a in [0, 1]:
                grad_norms[a] = {
                    "actor_grad_norm": 0.0,
                    "critic_grad_norm": 0.0,
                    "actor_loss": 0.0,
                    "critic_loss": 0.0,
                    "advantage_mean": 0.0,
                    "advantage_std": 0.0,
                }

        # Get total vehicles for verification (returns dict with {agent_id: total_vehicles})
        total_vehicles = env.get_total_vehicles()

        # Calculate vehicle discrepancy: initial vehicles minus sum of both agents' vehicles
        total_vehicles_both_agents = total_vehicles[0] + total_vehicles[1]
        vehicle_discrepancy = abs(initial_vehicles - total_vehicles_both_agents)

        # Calculate mean price scalar per agent (for modes 1 and 2)
        mean_price_scalar = {0: 0, 1: 0}
        if env.mode != 0:
            for a in [0, 1]:
                mean_price_scalar[a] = np.mean(actions_price[a]) if len(actions_price[a]) > 0 else 0

        # Calculate concentration statistics per agent (mode-specific)
        if env.mode == 0:
            mean_concentration_dirichlet = {0: 0, 1: 0}
            min_concentration_dirichlet = {0: 0, 1: 0}
            max_concentration_dirichlet = {0: 0, 1: 0}
            for a in [0, 1]:
                if len(actions_concentration_dirichlet[a]) > 0:
                    mean_concentration_dirichlet[a] = np.mean(actions_concentration_dirichlet[a])
                    # Use episode-level tracked min/max
                    min_concentration_dirichlet[a] = episode_min_concentration_dirichlet[a]
                    max_concentration_dirichlet[a] = episode_max_concentration_dirichlet[a]
        elif env.mode == 1:
            mean_concentration_alpha = {0: 0, 1: 0}
            mean_concentration_beta = {0: 0, 1: 0}
            min_concentration_alpha = {0: 0, 1: 0}
            min_concentration_beta = {0: 0, 1: 0}
            max_concentration_alpha = {0: 0, 1: 0}
            max_concentration_beta = {0: 0, 1: 0}
            for a in [0, 1]:
                if len(actions_concentration_alpha[a]) > 0:
                    mean_concentration_alpha[a] = np.mean(actions_concentration_alpha[a])
                    mean_concentration_beta[a] = np.mean(actions_concentration_beta[a])
                    # Use episode-level tracked min/max
                    min_concentration_alpha[a] = episode_min_concentration_alpha[a]
                    max_concentration_alpha[a] = episode_max_concentration_alpha[a]
                    min_concentration_beta[a] = episode_min_concentration_beta[a]
                    max_concentration_beta[a] = episode_max_concentration_beta[a]
        else:  # mode 2
            mean_concentration_alpha = {0: 0, 1: 0}
            mean_concentration_beta = {0: 0, 1: 0}
            mean_concentration_dirichlet = {0: 0, 1: 0}
            min_concentration_alpha = {0: 0, 1: 0}
            min_concentration_beta = {0: 0, 1: 0}
            min_concentration_dirichlet = {0: 0, 1: 0}
            max_concentration_alpha = {0: 0, 1: 0}
            max_concentration_beta = {0: 0, 1: 0}
            max_concentration_dirichlet = {0: 0, 1: 0}
            for a in [0, 1]:
                if len(actions_concentration_alpha[a]) > 0:
                    mean_concentration_alpha[a] = np.mean(actions_concentration_alpha[a])
                    mean_concentration_beta[a] = np.mean(actions_concentration_beta[a])
                    mean_concentration_dirichlet[a] = np.mean(actions_concentration_dirichlet[a])
                    # Use episode-level tracked min/max
                    min_concentration_alpha[a] = episode_min_concentration_alpha[a]
                    max_concentration_alpha[a] = episode_max_concentration_alpha[a]
                    min_concentration_beta[a] = episode_min_concentration_beta[a]
                    max_concentration_beta[a] = episode_max_concentration_beta[a]
                    min_concentration_dirichlet[a] = episode_min_concentration_dirichlet[a]
                    max_concentration_dirichlet[a] = episode_max_concentration_dirichlet[a]

        # Add training metrics to wandb
        log_dict = {
        "episode": i_episode + 1,
        # Agent 0 metrics
        "agent0/episode_reward": episode_reward[0],
        "agent0/episode_served_demand": episode_served_demand[0],
        "agent0/episode_unserved_demand": episode_unserved_demand[0],
        "agent0/episode_rebalancing_cost": episode_rebalancing_cost[0],
        "agent0/episode_waiting_time": episode_waiting[0]/episode_served_demand[0] if episode_served_demand[0] > 0 else 0,
        "agent0/total_revenue": episode_total_revenue[0],
        "agent0/total_operating_cost": episode_total_operating_cost[0],
        "agent0/actor_grad_norm": grad_norms[0]["actor_grad_norm"],
        "agent0/critic_grad_norm": grad_norms[0]["critic_grad_norm"],
        "agent0/actor_loss": grad_norms[0]["actor_loss"],
        "agent0/critic_loss": grad_norms[0]["critic_loss"],
        "agent0/advantage_mean": grad_norms[0]["advantage_mean"],
        "agent0/advantage_std": grad_norms[0]["advantage_std"],
        # Agent 0 profitability metrics
        "agent0/true_profit": episode_true_profit[0],
        "agent0/adjusted_profit": episode_adjusted_profit[0],
        "agent0/unprofitable_trips": episode_unprofitable_trips[0],
        # Agent 1 metrics
        "agent1/episode_reward": episode_reward[1],
        "agent1/episode_served_demand": episode_served_demand[1],
        "agent1/episode_unserved_demand": episode_unserved_demand[1],
        "agent1/episode_rebalancing_cost": episode_rebalancing_cost[1],
        "agent1/episode_waiting_time": episode_waiting[1]/episode_served_demand[1] if episode_served_demand[1] > 0 else 0,
        "agent1/total_revenue": episode_total_revenue[1],
        "agent1/total_operating_cost": episode_total_operating_cost[1],
        "agent1/actor_grad_norm": grad_norms[1]["actor_grad_norm"],
        "agent1/critic_grad_norm": grad_norms[1]["critic_grad_norm"],
        "agent1/actor_loss": grad_norms[1]["actor_loss"],
        "agent1/critic_loss": grad_norms[1]["critic_loss"],
        "agent1/advantage_mean": grad_norms[1]["advantage_mean"],
        "agent1/advantage_std": grad_norms[1]["advantage_std"],
        # Agent 1 profitability metrics
        "agent1/true_profit": episode_true_profit[1],
        "agent1/adjusted_profit": episode_adjusted_profit[1],
        "agent1/unprofitable_trips": episode_unprofitable_trips[1],
        # Combined metrics
        "combined/total_reward": episode_reward[0] + episode_reward[1],
        "combined/total_served_demand": episode_served_demand[0] + episode_served_demand[1],
        "combined/total_unserved_demand": episode_unserved_demand[0] + episode_unserved_demand[1],
        "combined/total_rebalancing_cost": episode_rebalancing_cost[0] + episode_rebalancing_cost[1],
        # Combined demand tracking (system-level)
        "combined/total_demand": episode_total_demand,
        "combined/rejected_demand": episode_rejected_demand,
        "combined/rejection_rate": np.mean(episode_rejection_rates) if len(episode_rejection_rates) > 0 else 0,
        # Combined profitability metrics
        "combined/total_true_profit": episode_true_profit[0] + episode_true_profit[1],
        "combined/total_adjusted_profit": episode_adjusted_profit[0] + episode_adjusted_profit[1],
        "combined/total_unprofitable_trips": episode_unprofitable_trips[0] + episode_unprofitable_trips[1],
        # Vehicle tracking
        "vehicles/agent0_total": total_vehicles[0],
        "vehicles/agent1_total": total_vehicles[1],
        "vehicles/combined_total": total_vehicles_both_agents,
        "vehicles/initial": initial_vehicles,
        "vehicles/discrepancy": vehicle_discrepancy
        }
        
        # Add price scalar metrics only for modes 1 and 2
        if env.mode != 0:
            log_dict["agent0/mean_price_scalar"] = mean_price_scalar[0]
            log_dict["agent1/mean_price_scalar"] = mean_price_scalar[1]
        
        # Add concentration metrics (mode-specific)
        if env.mode == 0:
            log_dict["agent0/mean_concentration_dirichlet"] = mean_concentration_dirichlet[0]
            log_dict["agent0/min_concentration_dirichlet"] = min_concentration_dirichlet[0]
            log_dict["agent0/max_concentration_dirichlet"] = max_concentration_dirichlet[0]
            log_dict["agent1/mean_concentration_dirichlet"] = mean_concentration_dirichlet[1]
            log_dict["agent1/min_concentration_dirichlet"] = min_concentration_dirichlet[1]
            log_dict["agent1/max_concentration_dirichlet"] = max_concentration_dirichlet[1]
        elif env.mode == 1:
            log_dict["agent0/mean_concentration_alpha"] = mean_concentration_alpha[0]
            log_dict["agent0/mean_concentration_beta"] = mean_concentration_beta[0]
            log_dict["agent0/min_concentration_alpha"] = min_concentration_alpha[0]
            log_dict["agent0/min_concentration_beta"] = min_concentration_beta[0]
            log_dict["agent0/max_concentration_alpha"] = max_concentration_alpha[0]
            log_dict["agent0/max_concentration_beta"] = max_concentration_beta[0]
            log_dict["agent1/mean_concentration_alpha"] = mean_concentration_alpha[1]
            log_dict["agent1/mean_concentration_beta"] = mean_concentration_beta[1]
            log_dict["agent1/min_concentration_alpha"] = min_concentration_alpha[1]
            log_dict["agent1/min_concentration_beta"] = min_concentration_beta[1]
            log_dict["agent1/max_concentration_alpha"] = max_concentration_alpha[1]
            log_dict["agent1/max_concentration_beta"] = max_concentration_beta[1]
        else:  # mode 2
            log_dict["agent0/mean_concentration_alpha"] = mean_concentration_alpha[0]
            log_dict["agent0/mean_concentration_beta"] = mean_concentration_beta[0]
            log_dict["agent0/mean_concentration_dirichlet"] = mean_concentration_dirichlet[0]
            log_dict["agent0/min_concentration_alpha"] = min_concentration_alpha[0]
            log_dict["agent0/min_concentration_beta"] = min_concentration_beta[0]
            log_dict["agent0/min_concentration_dirichlet"] = min_concentration_dirichlet[0]
            log_dict["agent0/max_concentration_alpha"] = max_concentration_alpha[0]
            log_dict["agent0/max_concentration_beta"] = max_concentration_beta[0]
            log_dict["agent0/max_concentration_dirichlet"] = max_concentration_dirichlet[0]
            log_dict["agent1/mean_concentration_alpha"] = mean_concentration_alpha[1]
            log_dict["agent1/mean_concentration_beta"] = mean_concentration_beta[1]
            log_dict["agent1/mean_concentration_dirichlet"] = mean_concentration_dirichlet[1]
            log_dict["agent1/min_concentration_alpha"] = min_concentration_alpha[1]
            log_dict["agent1/min_concentration_beta"] = min_concentration_beta[1]
            log_dict["agent1/min_concentration_dirichlet"] = min_concentration_dirichlet[1]
            log_dict["agent1/max_concentration_alpha"] = max_concentration_alpha[1]
            log_dict["agent1/max_concentration_beta"] = max_concentration_beta[1]
            log_dict["agent1/max_concentration_dirichlet"] = max_concentration_dirichlet[1]
        
        # Add log probability tracking for both agents
        for agent_id in [0, 1]:
            if len(episode_logprobs[agent_id]) > 0 and agent_id != args.fix_agent:
                log_dict[f"agent{agent_id}/mean_log_prob"] = np.mean(episode_logprobs[agent_id])
        
        # Add warmup tracking
        if args.critic_warmup_episodes > 0:
            log_dict["training/critic_warmup_active"] = 1 if i_episode < args.critic_warmup_episodes else 0
            log_dict["training/warmup_progress"] = min(1.0, i_episode / args.critic_warmup_episodes)
        
        # Add reward scaling factors for both agents (in non-baseline modes)
        if args.mode not in [3, 4]:
            for a in [0, 1]:
                if a != args.fix_agent:
                    log_dict[f"agent{a}/reward_scale"] = model_agents[a].reward_scale
        
        wandb.log(log_dict)

        # Keep metrics for both agents
        epoch_reward_list.append(episode_reward)  # This is already a dict {0: reward0, 1: reward1}
        
        # Track total arrivals per agent
        total_arrivals = {0: env.agent_arrivals[0], 1: env.agent_arrivals[1]}
        epoch_demand_list.append(total_arrivals)

        # Calculate waiting time per agent (handle division by zero)
        epoch_waiting_list.append({
            0: episode_waiting[0]/episode_served_demand[0] if episode_served_demand[0] > 0 else 0,
            1: episode_waiting[1]/episode_served_demand[1] if episode_served_demand[1] > 0 else 0
        })
        # Calculate served rate per agent (handle division by zero)
        epoch_servedrate_list.append({
            0: episode_served_demand[0]/env.agent_arrivals[0] if env.agent_arrivals[0] > 0 else 0,
            1: episode_served_demand[1]/env.agent_arrivals[1] if env.agent_arrivals[1] > 0 else 0
        })
        epoch_rebalancing_cost.append(episode_rebalancing_cost)  # This is already a dict {0: cost0, 1: cost1}

        # Update progress bar with metrics from both agents
        total_reward = episode_reward[0] + episode_reward[1]
        epochs.set_description(
            f"Episode {i_episode+1} | "
            f"Total Reward: {total_reward:.2f} | "
            f"Agent0: R={episode_reward[0]:.2f}, AGrad={grad_norms[0]['actor_grad_norm']:.2f}, CGrad={grad_norms[0]['critic_grad_norm']:.2f}, ALoss={grad_norms[0]['actor_loss']:.2f}, CLoss={grad_norms[0]['critic_loss']:.2f} | "
            f"Agent1: R={episode_reward[1]:.2f}, AGrad={grad_norms[1]['actor_grad_norm']:.2f}, CGrad={grad_norms[1]['critic_grad_norm']:.2f}, ALoss={grad_norms[1]['actor_loss']:.2f}, CLoss={grad_norms[1]['critic_loss']:.2f}"
        )
        
        # Checkpoint best performing models (based on combined reward)
        if args.mode not in [3, 4]:
            if total_reward >= best_reward:
                for agent_id in [0, 1]:
                    model_agents[agent_id].save_checkpoint(
                        path=f"ckpt/{args.checkpoint_path}_agent{agent_id+1}_sample.pth")
                best_reward = total_reward
            
            # Save running checkpoints for both agents
            for agent_id in [0, 1]:
                model_agents[agent_id].save_checkpoint(
                    path=f"ckpt/{args.checkpoint_path}_agent{agent_id+1}_running.pth")
        
        if i_episode % 100 == 0 and args.mode not in [3, 4]:
            for agent_id in [0, 1]:
                model_agents[agent_id].eval()
            test_reward, test_served_demand, test_rebalancing_cost = test_agents(
                    model_agents=model_agents, test_episodes=10, env=env, cplexpath=args.cplexpath, directory=args.directory, max_episodes=args.max_episodes, mode=args.mode, job_id=args.checkpoint_path)
            for agent_id in [0, 1]:
                model_agents[agent_id].train()

            test_reward = test_reward[0] + test_reward[1]
            if test_reward >= best_reward_test:
                        best_reward_test = test_reward
                        model_agents[0].save_checkpoint(
                            path=f"ckpt/{args.checkpoint_path}_agent1_test.pth")
                        model_agents[1].save_checkpoint(
                            path=f"ckpt/{args.checkpoint_path}_agent2_test.pth")

    # Training loop finished - save all metrics
    wandb.finish()

    metricPath = f"{args.directory}/train_logs/"
    if not os.path.exists(metricPath):
        os.makedirs(metricPath)

    # Save metrics for multi-agent setting
    # epoch_reward_list, epoch_waiting_list, epoch_servedrate_list, epoch_rebalancing_cost, epoch_actor_loss, and epoch_critic_loss are lists of dicts
    np.save(
        f"{args.directory}/train_logs/{city}_rewards_waiting_mode{args.mode}_{train_episodes}.npy", 
        np.array([epoch_reward_list, epoch_waiting_list, epoch_servedrate_list, epoch_demand_list, epoch_rebalancing_cost], dtype=object)
    )

    # Save export data with multi-agent structure
    export["avail_distri"] = env.agent_acc  # Multi-agent vehicle distribution
    export["demand_scaled"] = env.demand
    with open(f"{args.directory}/train_logs/{city}_export_mode{args.mode}_{train_episodes}.pickle", 'wb') as f:
        pickle.dump(export, f)

    print(f"\nTraining completed! Metrics saved to {metricPath}")

else:
    scenario = Scenario(
                json_file=f"data/scenario_{city}.json",
                demand_ratio=demand_ratio[city]*2,
                json_hr=json_hr[city],
                sd=args.seed,
                json_tstep=args.json_tstep,
                tf=args.max_steps,
                impute=args.impute,
                supply_ratio=args.supply_ratio,
                agent0_vehicle_ratio=args.agent0_vehicle_ratio,
                total_vehicles=args.total_vehicles)

    env = AMoD(scenario, args.mode, beta=beta[city], jitter=args.jitter, max_wait=args.maxt, choice_price_mult=args.choice_price_mult, seed = args.seed, fix_agent=args.fix_agent, choice_intercept=choice_intercept[city], wage=wage[city], use_dynamic_wage_man_south=args.use_dynamic_wage_man_south, od_price_actions=args.od_price_actions)
    
    # Print fixed agent information
    if args.fix_agent == 0:
        print("=" * 80)
        print("TEST MODE - FIXED AGENT: Agent 0 is FIXED")
        print("- Agent 0 uses BASE PRICES (scalar=0.5, no learning)")
        print("- Agent 0 is included in choice model and can receive demand")
        print("- Agent 0 vehicles reset to initial distribution each step")
        print("- Agent 1 uses learned policy")
        print("=" * 80)
    elif args.fix_agent == 1:
        print("=" * 80)
        print("TEST MODE - FIXED AGENT: Agent 1 is FIXED")
        print("- Agent 1 uses BASE PRICES (scalar=0.5, no learning)")
        print("- Agent 1 is included in choice model and can receive demand")
        print("- Agent 1 vehicles reset to initial distribution each step")
        print("- Agent 0 uses learned policy")
        print("=" * 80)
    else:
        print("=" * 80)
        print("TEST MODE - NORMAL: Both agents are active")
        print("- Demand is split via choice model")
        print("=" * 80)

    # Calculate input size based on price type
    if args.od_price_observe:
        if args.no_share_info:
            # Only own prices: T (future) + 3 (current_avb, queue, demand) + nregion (own OD prices)
            input_size = args.look_ahead + 3 + env.nregion
        else:
            # OD price matrices: T (future) + 3 (current_avb, queue, demand) + 3*nregion (own, competitor, and difference OD prices)
            input_size = args.look_ahead + 3 + 3 * env.nregion
    else:
        if args.no_share_info:
            # Only own price: T (future) + 3 (current_avb, queue, demand) + 1 (own aggregated price)
            input_size = args.look_ahead + 4
        else:
            # Aggregated prices: T (future) + 3 (current_avb, queue, demand) + 3 (own, competitor, and difference aggregated prices)
            input_size = args.look_ahead + 6
    
    # Only create models if not in baseline mode (mode 3 or 4)
    if args.mode not in [3, 4]:
        model_agents = {
                a: A2C(
                    env=env,
                    input_size=input_size,
                    hidden_size=args.hidden_size,
                    device=device,
                    p_lr=args.p_lr,
                    q_lr=args.q_lr,
                    T = args.look_ahead,
                    scale_factor = args.scale_factor,
                    json_file=f"data/scenario_{city}.json",
                    mode=args.mode,
                    actor_clip=args.actor_clip,
                    critic_clip=args.critic_clip,
                    gamma=args.gamma,
                    agent_id = a,
                    observe_od_prices = args.od_price_observe,
                    od_price_actions = args.od_price_actions,
                    no_share_info = args.no_share_info,
                    reward_scale=args.reward_scalar,
                )
                for a in [0, 1]
            }

        print("load models")
        for agent_id in [0, 1]:
            checkpoint_path = f"ckpt/{args.checkpoint_path}_agent{agent_id+1}_{args.model_type}.pth"
            model_agents[agent_id].load_checkpoint(path=checkpoint_path)
            print(f"Loaded checkpoint for agent {agent_id} from {checkpoint_path}")
        print("Loaded models from checkpoint successfully")
        
        for agent_id in [0, 1]:
            model_agents[agent_id].eval()
    elif args.mode == 3:
        print("=" * 80)
        print("TEST MODE - BASELINE (Mode 3): No learning, fixed policy")
        print("- Both agents use BASE PRICE (scalar=0.5)")
        print("- NO rebalancing performed")
        print("- Provides baseline for comparison")
        print("=" * 80)
        model_agents = None  # No models needed
    else:  # mode == 4
        print("=" * 80)
        print("TEST MODE - BASELINE (Mode 4): No learning, fixed policy with uniform rebalancing")
        print("- Both agents use BASE PRICE (scalar=0.5)")
        print("- UNIFORM rebalancing (distribute vehicles equally across regions)")
        print("- Provides baseline for comparison")
        print("=" * 80)
        model_agents = None  # No models needed
    
    # Initialize lists for logging
    log = {"test_reward": [], "test_served_demand": [], "test_reb_cost": []}
    
    epoch_reward_list = []
    epoch_demand_list = []
    epoch_rebalancing_cost = []
    epoch_waiting_list = []
    epoch_queue_length_list = []
    epoch_arrivals_list = []
    epoch_rebalancing_list = []
    epoch_price_mean_list = []
    
    # Wage tracking (only for dynamic wage scenarios)
    epoch_avg_wage_list = []
    
    # Initialize concentration tracking lists based on mode
    if env.mode == 0:
        epoch_concentration_dirichlet_list = []
    elif env.mode == 1:
        epoch_concentration_alpha_list = []
        epoch_concentration_beta_list = []
    elif env.mode == 2:
        epoch_concentration_alpha_list = []
        epoch_concentration_beta_list = []
        epoch_concentration_dirichlet_list = []
    # Mode 3 and 4: No concentration tracking needed (baseline modes)

    # Storage for trip data from last episode
    trip_data_last_episode = []
    
    # Storage for temporal visualization data from last episode (agent actions)
    visualization_data = {
        'agent_price_scalars': {0: [], 1: []},      # {agent_id: [timestep actions]} - List of price scalar arrays per timestep
        'agent_reb_actions': {0: [], 1: []},        # {agent_id: [timestep actions]} - List of rebalancing action arrays per timestep (Dirichlet)
        'agent_reb_flows': {0: [], 1: []},          # {agent_id: [timestep flows]} - Actual rebAction flows per edge per timestep
        'agent_acc_temporal': {0: [], 1: []},       # {agent_id: [timestep states]} - Vehicle availability at each timestep
        'agent_demand': {0: [], 1: []},             # {agent_id: [timestep demand]} - Demand assigned to each agent per region per timestep
        'edges': list(env.edges),                   # List of (origin, destination) tuples
        'metadata': {
            'num_regions': env.nregion,
            'num_timesteps': args.max_steps,
            'city': city,
            'mode': args.mode,
            'fix_agent': args.fix_agent,
            'max_episodes': args.max_episodes,
            'checkpoint_path': args.checkpoint_path,
            'regions': list(env.region)
        }
    }
    
    for episode in range(10):
        eps_reward = {0: 0, 1: 0}
        eps_demand = {0: 0, 1: 0}
        eps_rebalancing_cost = {0: 0, 1: 0}
        eps_waiting = {0: 0, 1: 0}
        eps_queue_length = {0: 0, 1: 0}
        eps_arrivals = {0: 0, 1: 0}
        eps_rebalancing = {0: 0, 1: 0}
        actions_price = {0: [], 1: []}
        
        # Wage tracking for episode
        eps_wage_samples = []
        
        # Initialize concentration tracking for episode based on mode
        if env.mode == 0:
            actions_concentration_dirichlet = {0: [], 1: []}
        elif env.mode == 1:
            actions_concentration_alpha = {0: [], 1: []}
            actions_concentration_beta = {0: [], 1: []}
        elif env.mode == 2:
            actions_concentration_alpha = {0: [], 1: []}
            actions_concentration_beta = {0: [], 1: []}
            actions_concentration_dirichlet = {0: [], 1: []}
        # Mode 3 and 4: No concentration tracking needed (baseline modes)
        
        obs = env.reset()

        action_rl = [0]*env.nregion        
        done = False

        while not done:
            if env.mode == 0:
                # Make Match Step
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple()
                
                # Collect wage samples if available
                if system_info.get('avg_wage') is not None:
                    eps_wage_samples.append(system_info['avg_wage'])

                # Update episode reward
                eps_reward = {a: eps_reward[a] + paxreward[a] for a in [0, 1]}

                # Get actions and concentrations
                action_rl = {}
                concentrations = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent: use actual initial distribution for rebalancing
                        total_vehicles = sum(env.agent_initial_acc[a].values())
                        action_rl[a] = np.array([
                            env.agent_initial_acc[a][env.region[i]] / total_vehicles 
                            for i in range(env.nregion)
                        ])
                        concentrations[a] = np.zeros((env.nregion, 1))
                    else:
                        action_conc = model_agents[a].select_action(obs[a], deterministic=True, return_concentration=True)
                        action_rl[a] = action_conc[0]
                        concentrations[a] = action_conc[1]
                
                # Track concentration (mode 0: Dirichlet for rebalancing)
                for a in [0, 1]:
                    if a != args.fix_agent:
                        actions_concentration_dirichlet[a].append(np.mean(concentrations[a]))

                # Track rebalancing actions for visualization (last episode only)
                if episode == 9:
                    for a in [0, 1]:
                        visualization_data['agent_reb_actions'][a].append(action_rl[a].copy())
                        # Track current vehicle availability
                        acc_current = np.array([env.agent_acc[a].get(env.region[i], {}).get(env.time, 0) for i in range(env.nregion)])
                        visualization_data['agent_acc_temporal'][a].append(acc_current)
                        # Track demand per region for this agent
                        demand_current = np.array([sum(env.agent_demand[a].get((env.region[i], j), {}).get(env.time, 0) for j in env.region) for i in range(env.nregion)])
                        visualization_data['agent_demand'][a].append(demand_current)

                # Compute desired accumulation for all agents
                desiredAcc = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # For fixed agent, set desiredAcc to initial distribution
                        desiredAcc[a] = {env.region[i]: env.agent_initial_acc[a][env.region[i]] for i in range(env.nregion)}
                    else:
                        # For active agent, use action to determine desired distribution
                        desiredAcc[a] = {
                            env.region[i]: int(action_rl[a][i] * dictsum(env.agent_acc[a], env.time + 1))
                            for i in range(env.nregion)
                        }

                # Compute rebalancing flows for both agents sequentially
                rebAction = {a: solveRebFlow(env, desiredAcc[a], a) for a in [0, 1]}
                
                # Capture actual rebalancing flows for visualization (last episode only)
                if episode == 9:
                    for a in [0, 1]:
                        visualization_data['agent_reb_flows'][a].append(np.array(rebAction[a]))
                
                _, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
                eps_reward = {a: eps_reward[a] + rebreward[a] for a in [0, 1]}
                
                
            elif env.mode == 1:
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                
                # Collect wage samples if available
                if system_info.get('avg_wage') is not None:
                    eps_wage_samples.append(system_info['avg_wage'])

                eps_reward = {a: eps_reward[a] + paxreward[a] for a in [0, 1]}

                # Get actions and concentrations
                action_rl = {}
                concentrations = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent: environment handles price override to 0.5
                        action_rl[a] = np.array([0.5] * env.nregion)
                        concentrations[a] = np.zeros((env.nregion, 2))
                    else:
                        action_conc = model_agents[a].select_action(obs[a], deterministic=True, return_concentration=True)
                        action_rl[a] = action_conc[0]
                        concentrations[a] = action_conc[1]
                
                # Track prices during episode (mode 1: action_rl is price scalar)
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent always uses 0.5 scalar
                        actions_price[a].append(1.0)  # 2 * 0.5 = 1.0 (base price)
                    else:
                        actions_price[a].append(np.mean(2 * np.array(action_rl[a])))
                
                # Track concentration (mode 1: Beta distribution - alpha and beta)
                for a in [0, 1]:
                    if a != args.fix_agent:
                        actions_concentration_alpha[a].append(np.mean(concentrations[a][0, :, 0]))
                        actions_concentration_beta[a].append(np.mean(concentrations[a][0, :, 1]))

                # Track price scalars for visualization (last episode only)
                if episode == 9:
                    for a in [0, 1]:
                        visualization_data['agent_price_scalars'][a].append(action_rl[a].copy())
                        # Track current vehicle availability
                        acc_current = np.array([env.agent_acc[a].get(env.region[i], {}).get(env.time, 0) for i in range(env.nregion)])
                        visualization_data['agent_acc_temporal'][a].append(acc_current)
                        # Track demand per region for this agent
                        demand_current = np.array([sum(env.agent_demand[a].get((env.region[i], j), {}).get(env.time, 0) for j in env.region) for i in range(env.nregion)])
                        visualization_data['agent_demand'][a].append(demand_current)

                # Matching update (global step)
                env.matching_update()
            
            elif env.mode == 2:
                # --- Matching step ---
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                
                # Collect wage samples if available
                if system_info.get('avg_wage') is not None:
                    eps_wage_samples.append(system_info['avg_wage'])
                
                eps_reward = {a: eps_reward[a] + paxreward[a] for a in [0, 1]}

                # Get actions and concentrations
                action_rl = {}
                concentrations = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent: environment handles price override to 0.5
                        # Mode 2 action shape: [nregion, 2] where [:, 0] = price scalar, [:, 1] = reb action
                        total_vehicles = sum(env.agent_initial_acc[a].values())
                        reb_action = np.array([
                            env.agent_initial_acc[a][env.region[i]] / total_vehicles 
                            for i in range(env.nregion)
                        ])
                        action_rl[a] = np.column_stack([
                            np.array([0.5] * env.nregion),  # Price (will be overridden by env)
                            reb_action  # Rebalancing: actual initial distribution
                        ])
                        concentrations[a] = np.zeros((env.nregion, 3))
                    else:
                        action_conc = model_agents[a].select_action(obs[a], deterministic=True, return_concentration=True)
                        action_rl[a] = action_conc[0]
                        concentrations[a] = action_conc[1]
                
                # Track prices during episode (mode 2: action_rl[:,0] is price scalar)
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent always uses 0.5 scalar
                        actions_price[a].append(1.0)  # 2 * 0.5 = 1.0 (base price)
                    else:
                        if args.od_price_actions:
                            # OD: action shape [nregion, nregion+1], prices in [:, :nregion]
                            actions_price[a].append(np.mean(2 * np.array(action_rl[a])[:, :env.nregion]))
                        else:
                            actions_price[a].append(np.mean(2 * np.array(action_rl[a])[:, 0]))
                
                # Track concentration (mode 2: Beta + Dirichlet)
                for a in [0, 1]:
                    if a != args.fix_agent:
                        if args.od_price_actions:
                            # OD: concentrations is dict {'beta': [1, nregion, nregion, 2], 'dirichlet': [1, nregion, 1]}
                            actions_concentration_alpha[a].append(np.mean(concentrations[a]['beta'][0, :, :, 0]))
                            actions_concentration_beta[a].append(np.mean(concentrations[a]['beta'][0, :, :, 1]))
                            actions_concentration_dirichlet[a].append(np.mean(concentrations[a]['dirichlet'][0, :, 0]))
                        else:
                            actions_concentration_alpha[a].append(np.mean(concentrations[a][:, 0]))
                            actions_concentration_beta[a].append(np.mean(concentrations[a][:, 1]))
                            actions_concentration_dirichlet[a].append(np.mean(concentrations[a][:, 2]))

                # Track price scalars and rebalancing actions for visualization (last episode only)
                if episode == 9:
                    for a in [0, 1]:
                        visualization_data['agent_price_scalars'][a].append(action_rl[a][:, 0].copy())  # Price scalars
                        visualization_data['agent_reb_actions'][a].append(action_rl[a][:, 1].copy())   # Reb actions
                        # Track current vehicle availability
                        acc_current = np.array([env.agent_acc[a].get(env.region[i], {}).get(env.time, 0) for i in range(env.nregion)])
                        visualization_data['agent_acc_temporal'][a].append(acc_current)
                        # Track demand per region for this agent
                        demand_current = np.array([sum(env.agent_demand[a].get((env.region[i], j), {}).get(env.time, 0) for j in env.region) for i in range(env.nregion)])
                        visualization_data['agent_demand'][a].append(demand_current)

                # --- Desired Acc computation ---
                # Compute desired accumulation for all agents
                desiredAcc = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # For fixed agent, set desiredAcc to initial distribution
                        desiredAcc[a] = {env.region[i]: env.agent_initial_acc[a][env.region[i]] for i in range(env.nregion)}
                    else:
                        # For active agent, use action to determine desired distribution
                        desiredAcc[a] = {
                            env.region[i]: int(action_rl[a][i, -1] * dictsum(env.agent_acc[a], env.time + 1))
                            for i in range(env.nregion)
                        }
                
                # --- Rebalancing step ---
                # Compute rebalancing flows for both agents sequentially
                rebAction = {a: solveRebFlow(env, desiredAcc[a], a) for a in [0, 1]}
                
                # Capture actual rebalancing flows for visualization (last episode only)
                if episode == 9:
                    for a in [0, 1]:
                        visualization_data['agent_reb_flows'][a].append(np.array(rebAction[a]))
            
                _, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
            
                eps_reward = {a: eps_reward[a] + rebreward[a] for a in [0, 1]}
            
            elif env.mode == 3:
                # === BASELINE MODE: No rebalancing, fixed prices ===
                # Use fixed price (scalar = 0.5 for both agents)
                action_rl = {
                    0: np.array([0.5] * env.nregion),
                    1: np.array([0.5] * env.nregion)
                }
                
                # Matching step with fixed prices
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                
                # Collect wage samples if available
                if system_info.get('avg_wage') is not None:
                    eps_wage_samples.append(system_info['avg_wage'])
                
                # Track rewards (no rebalancing cost in baseline)
                eps_reward = {a: eps_reward[a] + paxreward[a] for a in [0, 1]}
                
                # Track prices for logging (always 1.0 = base price)
                for a in [0, 1]:
                    actions_price[a].append(1.0)
                
                # Track price scalars for visualization (last episode only)
                if episode == 9:
                    for a in [0, 1]:
                        visualization_data['agent_price_scalars'][a].append(action_rl[a].copy())
                        # Track current vehicle availability
                        acc_current = np.array([env.agent_acc[a].get(env.region[i], {}).get(env.time, 0) for i in range(env.nregion)])
                        visualization_data['agent_acc_temporal'][a].append(acc_current)
                        # Track demand per region for this agent
                        demand_current = np.array([sum(env.agent_demand[a].get((env.region[i], j), {}).get(env.time, 0) for j in env.region) for i in range(env.nregion)])
                        visualization_data['agent_demand'][a].append(demand_current)
                
                # NO rebalancing step - just update vehicle arrivals from completed passenger trips
                env.matching_update()
            
            elif env.mode == 4:
                # === BASELINE MODE 4: Uniform rebalancing, fixed prices ===
                # Use fixed price (scalar = 0.5 for both agents)
                action_rl = {
                    0: np.array([0.5] * env.nregion),
                    1: np.array([0.5] * env.nregion)
                }
                
                # Matching step with fixed prices
                obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                
                # Collect wage samples if available
                if system_info.get('avg_wage') is not None:
                    eps_wage_samples.append(system_info['avg_wage'])
                
                # Track rewards
                eps_reward = {a: eps_reward[a] + paxreward[a] for a in [0, 1]}
                
                # Track prices for logging (always 1.0 = base price)
                for a in [0, 1]:
                    actions_price[a].append(1.0)
                
                # UNIFORM rebalancing: distribute vehicles equally across all regions
                desiredAcc = {}
                for a in [0, 1]:
                    # Calculate total available vehicles for this agent
                    current_total = dictsum(env.agent_acc[a], env.time + 1)
                    base_per_region = current_total // env.nregion
                    remainder = current_total % env.nregion
                    # Distribute uniformly with remainder going to first regions
                    desiredAcc[a] = {
                        env.region[i]: base_per_region + (1 if i < remainder else 0)
                        for i in range(env.nregion)
                    }
                
                # Track uniform rebalancing actions for visualization (last episode only)
                if episode == 9:
                    for a in [0, 1]:
                        uniform_reb_action = np.array([base_per_region + (1 if i < remainder else 0) for i in range(env.nregion)])
                        uniform_reb_action = uniform_reb_action / current_total if current_total > 0 else uniform_reb_action
                        visualization_data['agent_reb_actions'][a].append(uniform_reb_action)
                        visualization_data['agent_price_scalars'][a].append(action_rl[a].copy())
                        # Track current vehicle availability
                        acc_current = np.array([env.agent_acc[a].get(env.region[i], {}).get(env.time, 0) for i in range(env.nregion)])
                        visualization_data['agent_acc_temporal'][a].append(acc_current)
                        # Track demand per region for this agent
                        demand_current = np.array([sum(env.agent_demand[a].get((env.region[i], j), {}).get(env.time, 0) for j in env.region) for i in range(env.nregion)])
                        visualization_data['agent_demand'][a].append(demand_current)
                
                # Compute rebalancing flows for both agents sequentially
                rebAction = {a: solveRebFlow(env, desiredAcc[a], a) for a in [0, 1]}
                
                _, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
                eps_reward = {a: eps_reward[a] + rebreward[a] for a in [0, 1]}
            
            else:
                raise ValueError("Only mode 0, 1, 2, 3, and 4 are allowed")
        
            for a in [0, 1]:
                eps_demand[a] += info[a]["served_demand"]
                eps_rebalancing_cost[a] += info[a]["rebalancing_cost"]
                eps_waiting[a] += info[a]["served_waiting"]
        
        # After episode ends, capture episode-level metrics
        # Queue length: mean queue length across all regions for each agent (computed at episode end)
        for a in [0, 1]:
            eps_queue_length[a] = np.mean([len(env.agent_queue[a][i]) for i in env.agent_queue[a].keys()]) if len(env.agent_queue[a].keys()) > 0 else 0
            reb_od = 0
            for (o, d), flow in env.agent_rebFlow[a].items():
                reb_od += sum(flow.values())
            eps_rebalancing[a] = reb_od
        
        # Arrivals: total arrivals for the episode (tracked by environment)
        eps_arrivals = {0: env.agent_arrivals[0], 1: env.agent_arrivals[1]}
        
        # Price mean: average price across all steps for each agent (mode 1 and 2 only)
        eps_price_mean = {0: 0, 1: 0}
        if env.mode != 0:
            for a in [0, 1]:
                eps_price_mean[a] = np.mean(actions_price[a]) if len(actions_price[a]) > 0 else 0
        
        # Concentration mean: average concentration across all steps for each agent (mode-specific)
        if env.mode == 0:
            eps_concentration_dirichlet = {0: 0, 1: 0}
            for a in [0, 1]:
                eps_concentration_dirichlet[a] = np.mean(actions_concentration_dirichlet[a]) if len(actions_concentration_dirichlet[a]) > 0 else 0
        elif env.mode == 1:
            eps_concentration_alpha = {0: 0, 1: 0}
            eps_concentration_beta = {0: 0, 1: 0}
            for a in [0, 1]:
                eps_concentration_alpha[a] = np.mean(actions_concentration_alpha[a]) if len(actions_concentration_alpha[a]) > 0 else 0
                eps_concentration_beta[a] = np.mean(actions_concentration_beta[a]) if len(actions_concentration_beta[a]) > 0 else 0
        elif env.mode == 2:
            eps_concentration_alpha = {0: 0, 1: 0}
            eps_concentration_beta = {0: 0, 1: 0}
            eps_concentration_dirichlet = {0: 0, 1: 0}
            for a in [0, 1]:
                eps_concentration_alpha[a] = np.mean(actions_concentration_alpha[a]) if len(actions_concentration_alpha[a]) > 0 else 0
                eps_concentration_beta[a] = np.mean(actions_concentration_beta[a]) if len(actions_concentration_beta[a]) > 0 else 0
                eps_concentration_dirichlet[a] = np.mean(actions_concentration_dirichlet[a]) if len(actions_concentration_dirichlet[a]) > 0 else 0
        # Mode 3 and 4: No concentration tracking
        
        # Wage mean: average wage across all steps (only when use_dynamic_wage_man_south=True)
        eps_avg_wage = np.mean(eps_wage_samples) if len(eps_wage_samples) > 0 else None
        
        # Convert waiting time to minutes (from time steps)
        for a in [0, 1]:
            eps_waiting[a] = (eps_waiting[a] / eps_demand[a] * args.json_tstep) if eps_demand[a] > 0 else 0
        
        # Append episode results to epoch lists
        epoch_reward_list.append(eps_reward) # Done
        epoch_demand_list.append(eps_demand) # Done
        epoch_rebalancing_cost.append(eps_rebalancing_cost) # Done
        epoch_waiting_list.append(eps_waiting) # Done
        epoch_queue_length_list.append(eps_queue_length) # Done
        epoch_arrivals_list.append(eps_arrivals) # Done
        epoch_rebalancing_list.append(eps_rebalancing) # Done
        epoch_price_mean_list.append(eps_price_mean) # Done
        epoch_avg_wage_list.append(eps_avg_wage)
        
        # Append concentration results (mode-specific)
        if env.mode == 0:
            epoch_concentration_dirichlet_list.append(eps_concentration_dirichlet)
        elif env.mode == 1:
            epoch_concentration_alpha_list.append(eps_concentration_alpha)
            epoch_concentration_beta_list.append(eps_concentration_beta)
        elif env.mode == 2:
            epoch_concentration_alpha_list.append(eps_concentration_alpha)
            epoch_concentration_beta_list.append(eps_concentration_beta)
            epoch_concentration_dirichlet_list.append(eps_concentration_dirichlet)
        # Mode 3 and 4: No concentration tracking
        
        # Capture trip data from the last episode
        if episode == 9:  # Last episode (0-indexed, so episode 9 is the 10th)
            trip_data_last_episode = env.get_trip_assignments()

    # After all episodes, compute statistics across episodes for multi-agent case
    
    # Save visualization data from last episode to pickle file
    visualization_filename = f"{args.directory}/visualization_data/{args.checkpoint_path}_viz_data.pkl"
    os.makedirs(f"{args.directory}/visualization_data", exist_ok=True)
    
    # Convert lists to numpy arrays for easier handling
    for agent_id in [0, 1]:
        if len(visualization_data['agent_price_scalars'][agent_id]) > 0:
            visualization_data['agent_price_scalars'][agent_id] = np.array(visualization_data['agent_price_scalars'][agent_id])
        if len(visualization_data['agent_reb_actions'][agent_id]) > 0:
            visualization_data['agent_reb_actions'][agent_id] = np.array(visualization_data['agent_reb_actions'][agent_id])
        if len(visualization_data['agent_reb_flows'][agent_id]) > 0:
            visualization_data['agent_reb_flows'][agent_id] = np.array(visualization_data['agent_reb_flows'][agent_id])
        if len(visualization_data['agent_acc_temporal'][agent_id]) > 0:
            visualization_data['agent_acc_temporal'][agent_id] = np.array(visualization_data['agent_acc_temporal'][agent_id])
        if len(visualization_data['agent_demand'][agent_id]) > 0:
            visualization_data['agent_demand'][agent_id] = np.array(visualization_data['agent_demand'][agent_id])
    
    with open(visualization_filename, 'wb') as f:
        pickle.dump(visualization_data, f)
    
    print(f"\n{'='*80}")
    print(f"Visualization data saved to {visualization_filename}")
    print(f"Data structure:")
    print(f"  - agent_price_scalars: {[visualization_data['agent_price_scalars'][a].shape if len(visualization_data['agent_price_scalars'][a]) > 0 else 'empty' for a in [0, 1]]}")
    print(f"  - agent_reb_actions: {[visualization_data['agent_reb_actions'][a].shape if len(visualization_data['agent_reb_actions'][a]) > 0 else 'empty' for a in [0, 1]]}")
    print(f"  - agent_reb_flows: {[visualization_data['agent_reb_flows'][a].shape if len(visualization_data['agent_reb_flows'][a]) > 0 else 'empty' for a in [0, 1]]}")
    print(f"  - agent_acc_temporal: {[visualization_data['agent_acc_temporal'][a].shape if len(visualization_data['agent_acc_temporal'][a]) > 0 else 'empty' for a in [0, 1]]}")
    print(f"  - agent_demand: {[visualization_data['agent_demand'][a].shape if len(visualization_data['agent_demand'][a]) > 0 else 'empty' for a in [0, 1]]}")
    print(f"  - edges: {len(visualization_data['edges'])} edges")
    print(f"{'='*80}\n")
    
    # Save trip data from last episode to CSV
    if trip_data_last_episode:
        import pandas as pd
        df_trips = pd.DataFrame(trip_data_last_episode)
        trip_filename = f"{args.directory}/trip_data/trip_assignments_{city}_mode{args.mode}_fixagent{args.fix_agent}_episodes{args.max_episodes}.csv"
        os.makedirs(f"{args.directory}/trip_data", exist_ok=True)
        df_trips.to_csv(trip_filename, index=False)
        print(f"\nTrip assignment data saved to {trip_filename}")
        print(f"Total trips logged: {len(trip_data_last_episode)}")
    # Extract values for each agent across all episodes
    rewards_agent0 = [ep[0] for ep in epoch_reward_list]
    rewards_agent1 = [ep[1] for ep in epoch_reward_list]
    demands_agent0 = [ep[0] for ep in epoch_demand_list]
    demands_agent1 = [ep[1] for ep in epoch_demand_list]
    costs_agent0 = [ep[0] for ep in epoch_rebalancing_cost]
    costs_agent1 = [ep[1] for ep in epoch_rebalancing_cost]
    waiting_agent0 = [ep[0] for ep in epoch_waiting_list]
    waiting_agent1 = [ep[1] for ep in epoch_waiting_list]
    queue_agent0 = [ep[0] for ep in epoch_queue_length_list]
    queue_agent1 = [ep[1] for ep in epoch_queue_length_list]
    arrivals_agent0 = [ep[0] for ep in epoch_arrivals_list]
    arrivals_agent1 = [ep[1] for ep in epoch_arrivals_list]
    
    # Combined totals
    rewards_total = [ep[0] + ep[1] for ep in epoch_reward_list]
    demands_total = [ep[0] + ep[1] for ep in epoch_demand_list]
    costs_total = [ep[0] + ep[1] for ep in epoch_rebalancing_cost]
    arrivals_total = [ep[0] + ep[1] for ep in epoch_arrivals_list]
    
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    print("\nAgent 0 Metrics:")
    print(f"  Rewards (mean, std): {np.mean(rewards_agent0):.2f}, {np.std(rewards_agent0):.2f}")
    print(f"  Served demand (mean, std): {np.mean(demands_agent0):.2f}, {np.std(demands_agent0):.2f}")
    print(f"  Rebalancing cost (mean, std): {np.mean(costs_agent0):.2f}, {np.std(costs_agent0):.2f}")
    print(f"  Waiting time (mean, std): {np.mean(waiting_agent0):.2f}, {np.std(waiting_agent0):.2f}")
    print(f"  Queue length (mean, std): {np.mean(queue_agent0):.2f}, {np.std(queue_agent0):.2f}")
    print(f"  Arrivals (mean, std): {np.mean(arrivals_agent0):.2f}, {np.std(arrivals_agent0):.2f}")
    
    print("\nAgent 1 Metrics:")
    print(f"  Rewards (mean, std): {np.mean(rewards_agent1):.2f}, {np.std(rewards_agent1):.2f}")
    print(f"  Served demand (mean, std): {np.mean(demands_agent1):.2f}, {np.std(demands_agent1):.2f}")
    print(f"  Rebalancing cost (mean, std): {np.mean(costs_agent1):.2f}, {np.std(costs_agent1):.2f}")
    print(f"  Waiting time (mean, std): {np.mean(waiting_agent1):.2f}, {np.std(waiting_agent1):.2f}")
    print(f"  Queue length (mean, std): {np.mean(queue_agent1):.2f}, {np.std(queue_agent1):.2f}")
    print(f"  Arrivals (mean, std): {np.mean(arrivals_agent1):.2f}, {np.std(arrivals_agent1):.2f}")
    
    print("\nCombined Metrics:")
    print(f"  Total rewards (mean, std): {np.mean(rewards_total):.2f}, {np.std(rewards_total):.2f}")
    print(f"  Total served demand (mean, std): {np.mean(demands_total):.2f}, {np.std(demands_total):.2f}")
    print(f"  Total rebalancing cost (mean, std): {np.mean(costs_total):.2f}, {np.std(costs_total):.2f}")
    print(f"  Total arrivals (mean, std): {np.mean(arrivals_total):.2f}, {np.std(arrivals_total):.2f}")
    
    # Show average wage (always available now, regardless of dynamic wage flag)
    avg_wages = [w for w in epoch_avg_wage_list if w is not None]
    if avg_wages:
        print(f"  Average wage (mean, std): {np.mean(avg_wages):.2f}, {np.std(avg_wages):.2f}")
    
    # Only show rebalancing trips for modes with rebalancing (0, 2, 4; not mode 1 or 3)
    if args.mode not in [1, 3]:
        reb_agent0 = [ep[0] for ep in epoch_rebalancing_list]
        reb_agent1 = [ep[1] for ep in epoch_rebalancing_list]
        reb_total = [ep[0] + ep[1] for ep in epoch_rebalancing_list]
        print(f"  Agent 0 rebalancing trips (mean, std): {np.mean(reb_agent0):.2f}, {np.std(reb_agent0):.2f}")
        print(f"  Agent 1 rebalancing trips (mean, std): {np.mean(reb_agent1):.2f}, {np.std(reb_agent1):.2f}")
        print(f"  Total rebalancing trips (mean, std): {np.mean(reb_total):.2f}, {np.std(reb_total):.2f}")
    
    # Only show price scalar for modes 1, 2, and 3 (not mode 0)
    if args.mode != 0:
        price_agent0 = [ep[0] for ep in epoch_price_mean_list]
        price_agent1 = [ep[1] for ep in epoch_price_mean_list]
        print(f"  Agent 0 price scalar (mean, std): {np.mean(price_agent0):.2f}, {np.std(price_agent0):.2f}")
        print(f"  Agent 1 price scalar (mean, std): {np.mean(price_agent1):.2f}, {np.std(price_agent1):.2f}")
    
    # Show concentration parameters (mode-specific, not for modes 3 and 4)
    if args.mode not in [3, 4]:
        print("\nConcentration Parameters:")
        if args.mode == 0:
            # Mode 0: Only Dirichlet for rebalancing
            conc_dirichlet_agent0 = [ep[0] for ep in epoch_concentration_dirichlet_list]
            conc_dirichlet_agent1 = [ep[1] for ep in epoch_concentration_dirichlet_list]
            print(f"  Agent 0 Dirichlet concentration (mean, std): {np.mean(conc_dirichlet_agent0):.2f}, {np.std(conc_dirichlet_agent0):.2f}")
            print(f"  Agent 1 Dirichlet concentration (mean, std): {np.mean(conc_dirichlet_agent1):.2f}, {np.std(conc_dirichlet_agent1):.2f}")
        elif args.mode == 1:
            # Mode 1: Beta (alpha, beta) for pricing
            conc_alpha_agent0 = [ep[0] for ep in epoch_concentration_alpha_list]
            conc_alpha_agent1 = [ep[1] for ep in epoch_concentration_alpha_list]
            conc_beta_agent0 = [ep[0] for ep in epoch_concentration_beta_list]
            conc_beta_agent1 = [ep[1] for ep in epoch_concentration_beta_list]
            print(f"  Agent 0 Beta Alpha concentration (mean, std): {np.mean(conc_alpha_agent0):.2f}, {np.std(conc_alpha_agent0):.2f}")
            print(f"  Agent 0 Beta Beta concentration (mean, std): {np.mean(conc_beta_agent0):.2f}, {np.std(conc_beta_agent0):.2f}")
            print(f"  Agent 1 Beta Alpha concentration (mean, std): {np.mean(conc_alpha_agent1):.2f}, {np.std(conc_alpha_agent1):.2f}")
            print(f"  Agent 1 Beta Beta concentration (mean, std): {np.mean(conc_beta_agent1):.2f}, {np.std(conc_beta_agent1):.2f}")
        elif args.mode == 2:
            # Mode 2: Beta (alpha, beta) for pricing + Dirichlet for rebalancing
            conc_alpha_agent0 = [ep[0] for ep in epoch_concentration_alpha_list]
            conc_alpha_agent1 = [ep[1] for ep in epoch_concentration_alpha_list]
            conc_beta_agent0 = [ep[0] for ep in epoch_concentration_beta_list]
            conc_beta_agent1 = [ep[1] for ep in epoch_concentration_beta_list]
            conc_dirichlet_agent0 = [ep[0] for ep in epoch_concentration_dirichlet_list]
            conc_dirichlet_agent1 = [ep[1] for ep in epoch_concentration_dirichlet_list]
            print(f"  Agent 0 Beta Alpha concentration (mean, std): {np.mean(conc_alpha_agent0):.2f}, {np.std(conc_alpha_agent0):.2f}")
            print(f"  Agent 0 Beta Beta concentration (mean, std): {np.mean(conc_beta_agent0):.2f}, {np.std(conc_beta_agent0):.2f}")
            print(f"  Agent 0 Dirichlet concentration (mean, std): {np.mean(conc_dirichlet_agent0):.2f}, {np.std(conc_dirichlet_agent0):.2f}")
            print(f"  Agent 1 Beta Alpha concentration (mean, std): {np.mean(conc_alpha_agent1):.2f}, {np.std(conc_alpha_agent1):.2f}")
            print(f"  Agent 1 Beta Beta concentration (mean, std): {np.mean(conc_beta_agent1):.2f}, {np.std(conc_beta_agent1):.2f}")
            print(f"  Agent 1 Dirichlet concentration (mean, std): {np.mean(conc_dirichlet_agent1):.2f}, {np.std(conc_dirichlet_agent1):.2f}")