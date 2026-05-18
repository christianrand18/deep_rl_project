import argparse
from typing import Optional


class A2CArgumentBuilder:
	"""Shared CLI argument builder for single- and multi-agent training scripts."""

	@staticmethod
	def _add_common_arguments(
		parser: argparse.ArgumentParser,
		*,
		mode_help: str,
		critic_warmup_default: int,
		model_type_help: str,
		model_type_choices: Optional[list[str]] = None,
	) -> None:
		# Simulator parameters
		parser.add_argument(
			"--seed", type=int, default=10, metavar="S", help="random seed (default: 10)"
		)
		parser.add_argument(
			"--model_type",
			type=str,
			default="running",
			choices=model_type_choices,
			help=model_type_help,
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
			"--mode",
			type=int,
			default=2,
			help=mode_help,
		)

		# Model parameters
		parser.add_argument(
			"--test",
			action="store_true",
			default=False,
			help="activates test mode for agent evaluation",
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
			default=critic_warmup_default,
			help=f"number of episodes to train only critic before training actor (default: {critic_warmup_default})",
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
			help="Use OD-based price scalars (NxN outputs) instead of origin-based (N outputs) (default: False)",
		)
		parser.add_argument(
			"--reward_scalar",
			type=float,
			default=2000.0,
			help="Reward scaling factor (default: 2000.0)",
		)

	@classmethod
	def build_single_agent_parser(cls) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(description="A2C-GNN")
		cls._add_common_arguments(
			parser,
			mode_help=(
				"rebalancing mode. (0:rebalancing only, 1:pricing only, 2:both, 3:baseline no reb/fixed price, "
				"4:baseline uniform reb/fixed price. default 2)"
			),
			critic_warmup_default=1000,
			model_type_help="which checkpoint variant to load: running, test, or sample (default: running)",
			model_type_choices=["running", "test", "sample"],
		)
		parser.add_argument(
			"--beta",
			type=float,
			default=0.5,
			metavar="S",
			help="cost of rebalancing (default: 0.5)",
		)
		parser.add_argument(
			"--fix_baseline",
			action="store_true",
			default=False,
			help="Fix baseline behavior: use base price and initial vehicle distribution (default: False)",
		)
		return parser

	@classmethod
	def build_multi_agent_parser(cls) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(description="A2C-GNN")
		cls._add_common_arguments(
			parser,
			mode_help=(
				"rebalancing mode. (0:manual, 1:pricing, 2:both, 3:baseline (no reb, fixed price), "
				"4:baseline (uniform reb, fixed price). default 2)"
			),
			critic_warmup_default=50,
			model_type_help="Defines the type of model (default: running)",
		)
		parser.add_argument(
			"--load_checkpoint_path",
			type=str,
			default=None,
			help=(
				"name of checkpoint file to load from (if different from checkpoint_path). "
				"If not specified, uses checkpoint_path (default: None)"
			),
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
			"--fix_agent",
			type=int,
			default=2,
			choices=[0, 1, 2],
			help="Fix agent behavior for testing: 0=fix agent 0, 1=fix agent 1, 2=no fixing (default: 2)",
		)
		parser.add_argument(
			"--no_share_info",
			action="store_true",
			default=False,
			help="Don't share competitor pricing info between agents (default: False)",
		)
		parser.add_argument(
			"--use_dynamic_wage_man_south",
			action="store_true",
			default=False,
			help="Enable region-specific wage distributions for NYC Manhattan South (default: False)",
		)
		parser.add_argument(
			"--num_days",
			type=int,
			default=1,
			help="Number of simulated days per episode (default: 1). Values > 1 activate multi-day mode.",
		)
		parser.add_argument(
			"--brand_momentum_lambda",
			type=float,
			default=0.9,
			help="EMA decay for brand momentum (default: 0.9). Higher = slower adaptation.",
		)
		parser.add_argument(
			"--brand_momentum_gamma",
			type=float,
			default=0.0,
			help="Strength of brand momentum in MNL utility (default: 0.0 = disabled, reproduces baseline).",
		)
		return parser

	@classmethod
	def parse_single_agent_args(cls):
		return cls.build_single_agent_parser().parse_args()

	@classmethod
	def parse_multi_agent_args(cls):
		args = cls.build_multi_agent_parser().parse_args()
		if not (0.0 <= args.agent0_vehicle_ratio <= 1.0):
			raise ValueError(
				f"agent0_vehicle_ratio must be between 0.0 and 1.0, got {args.agent0_vehicle_ratio}"
			)
		return args
