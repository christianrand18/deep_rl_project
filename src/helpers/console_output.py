"""Console output helpers for training and evaluation scripts."""

import numpy as np


def print_od_price_observe_notice():
    print("=" * 80)
    print("INFO: Automatically enabling --od_price_observe since --od_price_actions is set")
    print("      (OD-based actions require OD-based observations)")
    print("=" * 80)


def print_single_train_mode_banner(mode, fix_baseline, initial_vehicles=None, initial_distribution=None):
    if fix_baseline:
        print("\n" + "=" * 50)
        print("FIXED BASELINE MODE ACTIVATED")
        print("=" * 50)
        print("Behavior:")
        print("  - Prices: Always using base price (price scalar = 0.5)")
        print("  - Rebalancing: Always rebalancing to initial distribution")
        print(f"  - Initial vehicles: {initial_vehicles}")
        print(f"  - Target distribution: {dict(initial_distribution)}")
        print("=" * 50 + "\n")
    elif mode == 3:
        print("\n" + "=" * 80)
        print("BASELINE MODE (Mode 3): No learning, fixed policy")
        print("=" * 80)
        print("- Both agents use BASE PRICE (scalar=0.5)")
        print("- NO rebalancing (vehicles stay where trips end)")
        print("- Provides baseline for comparison")
        print("=" * 80 + "\n")
    elif mode == 4:
        print("\n" + "=" * 80)
        print("BASELINE MODE (Mode 4): No learning, fixed policy with uniform rebalancing")
        print("=" * 80)
        print("- Both agents use BASE PRICE (scalar=0.5)")
        print("- UNIFORM rebalancing (distribute vehicles equally across regions)")
        print("- Provides baseline for comparison")
        print("=" * 80 + "\n")
    else:
        print("\n" + "=" * 50)
        print("NORMAL TRAINING MODE")
        print("=" * 50)
        print("Agent will learn both pricing and rebalancing strategies")
        print("=" * 50 + "\n")


def print_single_test_mode_banner(mode):
    if mode == 3:
        print("\n" + "=" * 80)
        print("TEST MODE - BASELINE (Mode 3): No learning, fixed policy")
        print("=" * 80)
        print("- Using BASE PRICE (scalar=0.5)")
        print("- NO rebalancing (vehicles stay where trips end)")
        print("- Provides baseline for comparison")
        print("=" * 80 + "\n")
    elif mode == 4:
        print("\n" + "=" * 80)
        print("TEST MODE - BASELINE (Mode 4): No learning, fixed policy with uniform rebalancing")
        print("=" * 80)
        print("- Using BASE PRICE (scalar=0.5)")
        print("- UNIFORM rebalancing (distribute vehicles equally across regions)")
        print("- Provides baseline for comparison")
        print("=" * 80 + "\n")


def print_single_test_summary(rewards, demands, costs, waiting_steps, queue_steps, arrivals, reb_num, mode, price_mean):
    print("Rewards (mean, std):", f"{np.mean(rewards):.2f}", f"{np.std(rewards):.2f}")
    print("Served demand (mean, std):", f"{np.mean(demands):.2f}", f"{np.std(demands):.2f}")
    print("Rebalancing cost (mean, std):", f"{np.mean(costs):.2f}", f"{np.std(costs):.2f}")
    print("Waiting time (mean, std):", f"{np.mean(waiting_steps):.2f}", f"{np.std(waiting_steps):.2f}")
    print("Queue length (mean, std):", f"{np.mean(queue_steps):.2f}", f"{np.std(queue_steps):.2f}")
    print("Arrivals (mean, std):", f"{np.mean(arrivals):.2f}", f"{np.std(arrivals):.2f}")
    print("Rebalancing trips (mean, std):", f"{np.mean(reb_num):.2f}", f"{np.std(reb_num):.2f}")
    if mode != 0:
        print("Price scalar (mean, std):", f"{np.mean(price_mean):.2f}", f"{np.std(price_mean):.2f}")


def print_multi_train_mode_banner(fix_agent):
    if fix_agent == 0:
        print("=" * 80)
        print("FIXED AGENT MODE: Agent 0 is FIXED")
        print("- Agent 0 uses BASE PRICES (scalar=0.5, no learning)")
        print("- Agent 0 is included in choice model and can receive demand")
        print("- Agent 0 vehicles reset to initial distribution each step")
        print("- Agent 1 is LEARNING (adjusts prices dynamically)")
        print("=" * 80)
    elif fix_agent == 1:
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


def print_multi_baseline_mode_banner(mode, is_test=False):
    if mode == 3:
        prefix = "TEST MODE - " if is_test else ""
        print("=" * 80)
        print(f"{prefix}BASELINE (Mode 3): No learning, fixed policy")
        print("- Both agents use BASE PRICE (scalar=0.5)")
        print("- NO rebalancing performed")
        print("- Provides baseline for comparison")
        print("=" * 80)
    elif mode == 4:
        prefix = "TEST MODE - " if is_test else ""
        print("=" * 80)
        print(f"{prefix}BASELINE (Mode 4): No learning, fixed policy with uniform rebalancing")
        print("- Both agents use BASE PRICE (scalar=0.5)")
        print("- UNIFORM rebalancing (distribute vehicles equally across regions)")
        print("- Provides baseline for comparison")
        print("=" * 80)


def print_multi_test_mode_banner(fix_agent):
    if fix_agent == 0:
        print("=" * 80)
        print("TEST MODE - FIXED AGENT: Agent 0 is FIXED")
        print("- Agent 0 uses BASE PRICES (scalar=0.5, no learning)")
        print("- Agent 0 is included in choice model and can receive demand")
        print("- Agent 0 vehicles reset to initial distribution each step")
        print("- Agent 1 uses learned policy")
        print("=" * 80)
    elif fix_agent == 1:
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


def print_critic_warmup_banner(critic_warmup_episodes):
    print("=" * 80)
    print(f"CRITIC WARMUP ENABLED: {critic_warmup_episodes} episodes")
    print(f"- Episodes 0-{critic_warmup_episodes - 1}: Critic only (actor frozen)")
    print(f"- Episodes {critic_warmup_episodes}+: Both actor and critic training")
    print("=" * 80 + "\n")


def print_multi_baseline_training_notice():
    print("\nRunning baseline mode with fixed policy...")
    print("No training will be performed. Running test episodes only.\n")


def print_multi_training_completed(metric_path):
    print(f"\nTraining completed! Metrics saved to {metric_path}")


def print_multi_visualization_summary(visualization_filename, visualization_data):
    print(f"\n{'='*80}")
    print(f"Visualization data saved to {visualization_filename}")
    print("Data structure:")
    print(
        f"  - agent_price_scalars: "
        f"{[visualization_data['agent_price_scalars'][a].shape if len(visualization_data['agent_price_scalars'][a]) > 0 else 'empty' for a in [0, 1]]}"
    )
    print(
        f"  - agent_reb_actions: "
        f"{[visualization_data['agent_reb_actions'][a].shape if len(visualization_data['agent_reb_actions'][a]) > 0 else 'empty' for a in [0, 1]]}"
    )
    print(
        f"  - agent_reb_flows: "
        f"{[visualization_data['agent_reb_flows'][a].shape if len(visualization_data['agent_reb_flows'][a]) > 0 else 'empty' for a in [0, 1]]}"
    )
    print(
        f"  - agent_acc_temporal: "
        f"{[visualization_data['agent_acc_temporal'][a].shape if len(visualization_data['agent_acc_temporal'][a]) > 0 else 'empty' for a in [0, 1]]}"
    )
    print(
        f"  - agent_demand: "
        f"{[visualization_data['agent_demand'][a].shape if len(visualization_data['agent_demand'][a]) > 0 else 'empty' for a in [0, 1]]}"
    )
    print(f"  - edges: {len(visualization_data['edges'])} edges")
    print(f"{'='*80}\n")


def print_multi_trip_data_saved(trip_filename, trip_count):
    print(f"\nTrip assignment data saved to {trip_filename}")
    print(f"Total trips logged: {trip_count}")


def print_multi_test_results_summary(
    mode,
    epoch_reward_list,
    epoch_demand_list,
    epoch_rebalancing_cost,
    epoch_waiting_list,
    epoch_queue_length_list,
    epoch_arrivals_list,
    epoch_rebalancing_list,
    epoch_price_mean_list,
    epoch_avg_wage_list,
    epoch_concentration_dirichlet_list=None,
    epoch_concentration_alpha_list=None,
    epoch_concentration_beta_list=None,
):
    epoch_concentration_dirichlet_list = epoch_concentration_dirichlet_list or []
    epoch_concentration_alpha_list = epoch_concentration_alpha_list or []
    epoch_concentration_beta_list = epoch_concentration_beta_list or []

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

    rewards_total = [ep[0] + ep[1] for ep in epoch_reward_list]
    demands_total = [ep[0] + ep[1] for ep in epoch_demand_list]
    costs_total = [ep[0] + ep[1] for ep in epoch_rebalancing_cost]
    arrivals_total = [ep[0] + ep[1] for ep in epoch_arrivals_list]

    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)

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

    avg_wages = [w for w in epoch_avg_wage_list if w is not None]
    if avg_wages:
        print(f"  Average wage (mean, std): {np.mean(avg_wages):.2f}, {np.std(avg_wages):.2f}")

    if mode not in [1, 3]:
        reb_agent0 = [ep[0] for ep in epoch_rebalancing_list]
        reb_agent1 = [ep[1] for ep in epoch_rebalancing_list]
        reb_total = [ep[0] + ep[1] for ep in epoch_rebalancing_list]
        print(f"  Agent 0 rebalancing trips (mean, std): {np.mean(reb_agent0):.2f}, {np.std(reb_agent0):.2f}")
        print(f"  Agent 1 rebalancing trips (mean, std): {np.mean(reb_agent1):.2f}, {np.std(reb_agent1):.2f}")
        print(f"  Total rebalancing trips (mean, std): {np.mean(reb_total):.2f}, {np.std(reb_total):.2f}")

    if mode != 0:
        price_agent0 = [ep[0] for ep in epoch_price_mean_list]
        price_agent1 = [ep[1] for ep in epoch_price_mean_list]
        print(f"  Agent 0 price scalar (mean, std): {np.mean(price_agent0):.2f}, {np.std(price_agent0):.2f}")
        print(f"  Agent 1 price scalar (mean, std): {np.mean(price_agent1):.2f}, {np.std(price_agent1):.2f}")

    if mode not in [3, 4]:
        print("\nConcentration Parameters:")
        if mode == 0:
            conc_dirichlet_agent0 = [ep[0] for ep in epoch_concentration_dirichlet_list]
            conc_dirichlet_agent1 = [ep[1] for ep in epoch_concentration_dirichlet_list]
            print(
                f"  Agent 0 Dirichlet concentration (mean, std): "
                f"{np.mean(conc_dirichlet_agent0):.2f}, {np.std(conc_dirichlet_agent0):.2f}"
            )
            print(
                f"  Agent 1 Dirichlet concentration (mean, std): "
                f"{np.mean(conc_dirichlet_agent1):.2f}, {np.std(conc_dirichlet_agent1):.2f}"
            )
        elif mode == 1:
            conc_alpha_agent0 = [ep[0] for ep in epoch_concentration_alpha_list]
            conc_alpha_agent1 = [ep[1] for ep in epoch_concentration_alpha_list]
            conc_beta_agent0 = [ep[0] for ep in epoch_concentration_beta_list]
            conc_beta_agent1 = [ep[1] for ep in epoch_concentration_beta_list]
            print(f"  Agent 0 Beta Alpha concentration (mean, std): {np.mean(conc_alpha_agent0):.2f}, {np.std(conc_alpha_agent0):.2f}")
            print(f"  Agent 0 Beta Beta concentration (mean, std): {np.mean(conc_beta_agent0):.2f}, {np.std(conc_beta_agent0):.2f}")
            print(f"  Agent 1 Beta Alpha concentration (mean, std): {np.mean(conc_alpha_agent1):.2f}, {np.std(conc_alpha_agent1):.2f}")
            print(f"  Agent 1 Beta Beta concentration (mean, std): {np.mean(conc_beta_agent1):.2f}, {np.std(conc_beta_agent1):.2f}")
        elif mode == 2:
            conc_alpha_agent0 = [ep[0] for ep in epoch_concentration_alpha_list]
            conc_alpha_agent1 = [ep[1] for ep in epoch_concentration_alpha_list]
            conc_beta_agent0 = [ep[0] for ep in epoch_concentration_beta_list]
            conc_beta_agent1 = [ep[1] for ep in epoch_concentration_beta_list]
            conc_dirichlet_agent0 = [ep[0] for ep in epoch_concentration_dirichlet_list]
            conc_dirichlet_agent1 = [ep[1] for ep in epoch_concentration_dirichlet_list]
            print(f"  Agent 0 Beta Alpha concentration (mean, std): {np.mean(conc_alpha_agent0):.2f}, {np.std(conc_alpha_agent0):.2f}")
            print(f"  Agent 0 Beta Beta concentration (mean, std): {np.mean(conc_beta_agent0):.2f}, {np.std(conc_beta_agent0):.2f}")
            print(
                f"  Agent 0 Dirichlet concentration (mean, std): "
                f"{np.mean(conc_dirichlet_agent0):.2f}, {np.std(conc_dirichlet_agent0):.2f}"
            )
            print(f"  Agent 1 Beta Alpha concentration (mean, std): {np.mean(conc_alpha_agent1):.2f}, {np.std(conc_alpha_agent1):.2f}")
            print(f"  Agent 1 Beta Beta concentration (mean, std): {np.mean(conc_beta_agent1):.2f}, {np.std(conc_beta_agent1):.2f}")
            print(
                f"  Agent 1 Dirichlet concentration (mean, std): "
                f"{np.mean(conc_dirichlet_agent1):.2f}, {np.std(conc_dirichlet_agent1):.2f}"
            )
