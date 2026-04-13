# Competititve Multi-Operator Reinforcement Learning for Joint Pricing and Fleet Rebalancing in Autonomous Mobility-on-Demand Systems

Implementation of A2C-based Graph Convolutional Network operators for joint rebalancing and dynamic pricing in Autonomous Mobility-on-Demand (AMoD) systems, supporting both single-agent and competitive multi-agent settings.

## Prerequisites

You will need a working **IBM CPLEX** installation. If you are a student or academic, IBM offers CPLEX Optimization Studio for free. More info [here](https://community.ibm.com/community/user/datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students).

You will also need a **Weights & Biases** account for experiment tracking. Create a free account at [wandb.ai](https://wandb.ai).

The code is built with **Python 3.10**. Different Python versions may cause errors. To install all required dependencies, run:
```
pip install -r requirements.txt
```
It is recommended to create a virtual environment before installing the packages:
```bash
python -m venv my_env
source my_env/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root with your Weights & Biases API key:
```
WANDB_API_KEY=your_api_key_here
```
This file is loaded automatically at runtime via `python-dotenv`. You can find your API key at [wandb.ai/authorize](https://wandb.ai/authorize).

### Required Directories

Before running any training, create the following directories in the project root:
```bash
mkdir -p saved_files ckpt logs
```
- `saved_files/` — stores intermediate files generated during CPLEX solving
- `ckpt/` — stores model checkpoint weight files
- `logs/` — stores job output and error logs (for HPC batch jobs)

## Contents

* `main_a2c.py`: Single-agent training and evaluation entry point.
* `main_a2c_multi_agent.py`: Multi-agent (duopoly) training and evaluation entry point.
* `src/algos/a2c_gnn.py`: PyTorch + PyG implementation of A2C with Graph Neural Networks (single agent).
* `src/algos/a2c_gnn_multi_agent.py`: Multi-agent A2C-GNN implementation.
* `src/algos/layers.py`: GNN layer definitions.
* `src/algos/reb_flow_solver.py`: Wrapper around the CPLEX Minimum Rebalancing Cost formulation (single agent).
* `src/algos/reb_flow_solver_multi_agent.py`: Multi-agent rebalancing flow solver.
* `src/envs/amod_env.py`: AMoD simulator (single agent).
* `src/envs/amod_env_multi.py`: AMoD simulator (multi-agent with fleet splitting).
* `src/envs/structures.py`: Data structures used by the simulator.
* `src/misc/`: Utility functions.
* `data/`: Scenario JSON files and calibration data for supported cities.

### Supported Cities

| City key | Description |
|---|---|
| `san_francisco` | San Francisco |
| `nyc_man_south` | NYC Manhattan South |
| `washington_dc` | Washington D.C. |

## Experiments

### Arguments

Both `main_a2c.py` (single agent) and `main_a2c_multi_agent.py` (multi-agent) accept the following core arguments:

```bash
cplex arguments:
    --cplexpath         directory of the CPLEX installation

simulator arguments:
    --seed              random seed (default: 10)
    --json_tstep        minutes per timestep (default: 3)
    --city              city to train on (default: nyc_man_south)
    --supply_ratio      supply scaling factor (default: 1.0)
    --jitter            demand jitter (default: 1)
    --maxt              maximum passenger waiting time in time steps (default: 2)

model arguments:
    --test              activates agent evaluation mode (default: False)
    --mode              policy mode: 0=rebalancing only, 1=pricing only, 2=joint,
                        3=baseline (no reb, fixed price), 4=baseline (uniform reb, fixed price)
                        (default: 2)
    --max_episodes      number of training episodes (default: 100000)
    --max_steps         number of steps per episode (default: 20)
    --hidden_size       GNN hidden dimension (default: 256)
    --p_lr              actor learning rate (default: 2e-4)
    --q_lr              critic learning rate (default: 6e-4)
    --gamma             discount factor (default: 0.97)
    --actor_clip        actor gradient clip value (default: 1000)
    --critic_clip       critic gradient clip value (default: 1000)
    --critic_warmup_episodes  episodes to train only critic before actor (default: 1000 single / 50 multi)
    --look_ahead        time steps to look ahead (default: 6)
    --scale_factor      scale factor (default: 0.01)
    --reward_scalar     reward scaling factor (default: 2000.0)
    --checkpoint_path   name of checkpoint file to save/load (default: A2C)
    --model_type        checkpoint variant to load: running, test, or sample (default: running)
    --load              start training from checkpoint (default: False)
    --cuda              enable CUDA training (default: False)
    --directory         directory for intermediate files (default: saved_files)
    --od_price_observe  use OD price matrices for observations (default: False)
    --od_price_actions  use OD-based price scalars (N×N) instead of origin-based (N) (default: False)
```

#### Multi-agent specific arguments (`main_a2c_multi_agent.py` only):

```bash
    --agent0_vehicle_ratio  proportion of vehicles for agent 0 (default: 0.5, range: 0.0–1.0)
    --total_vehicles        total number of vehicles; if None, reads from dataset (default: None)
    --fix_agent             fix agent behaviour for testing: 0=fix agent 0, 1=fix agent 1, 2=none (default: 2)
    --no_share_info         do not share competitor pricing info between agents (default: False)
    --use_dynamic_wage_man_south  enable region-specific wage distributions for NYC Manhattan South (default: False)
```

***Important***: Specify the correct path for your local CPLEX installation. Typical default paths:
```bash
Windows: "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
macOS:   "/Applications/CPLEX_Studio1210/opl/bin/x86-64_osx/"
Linux:   "/opt/ibm/ILOG/CPLEX_Studio1210/opl/bin/x86-64_linux/"
```

### Training

#### Single-agent training
Train a single agent for the joint rebalancing + pricing policy:
```bash
python main_a2c.py --city nyc_man_south --mode 2 --checkpoint_path my_single_agent
```

With OD-level price actions and observations:
```bash
python main_a2c.py --city nyc_man_south --mode 2 --od_price_actions --checkpoint_path my_single_agent_od
```

#### Multi-agent training
Train two competing agents in a duopoly setting:
```bash
python main_a2c_multi_agent.py --city nyc_man_south --mode 2 --checkpoint_path my_dual_agent
```

With OD-level price actions and observations:
```bash
python main_a2c_multi_agent.py --city nyc_man_south --mode 2 --od_price_observe --od_price_actions --checkpoint_path my_dual_agent_od
```

### Evaluation

#### Single-agent evaluation
```bash
python main_a2c.py --city nyc_man_south --test --checkpoint_path my_single_agent
```

#### Multi-agent evaluation
```bash
python main_a2c_multi_agent.py --city nyc_man_south --test --checkpoint_path my_dual_agent
```

When specifying `--checkpoint_path`, do **not** include `ckpt/` in the name — checkpoints are automatically read from/written to the `ckpt/` directory.

### Baseline modes

To run baseline comparisons without learned policies:
```bash
# Mode 3: fixed base price, no rebalancing
python main_a2c.py --city nyc_man_south --mode 3

# Mode 4: fixed base price, uniform rebalancing
python main_a2c.py --city nyc_man_south --mode 4
```