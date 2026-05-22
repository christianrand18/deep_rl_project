#!/bin/sh

### Job Name:
#BSUB -J sanity_bound_d7_g5_meta_one

### Queue Name:
#BSUB -q hpc

### Requesting 4 CPU cores, 4GB memory per core
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"

### Setting a runtime limit of 48 hours
#BSUB -W 48:00

### Email notification when job begins and ends
#BSUB -B
#BSUB -N

### Output and error files
#BSUB -o batch_jobs/logs/Output_%J.out
#BSUB -e batch_jobs/logs/Output_%J.err


### cd to repo dir
cd ~/deep_rl_project

mkdir -p batch_jobs/logs
export CPLEX_PATH="/apps/cplex/cplex1210/opl/bin/x86-64_linux/"

### activate environment
. .venv/bin/activate

### Ablation: bound meta-controlled agent's pre-meta scalar to [0.4, 0.6]
### (factor 0.8x to 1.2x baseline). Tests whether limiting the low-level's
### compensation authority lets the meta-policy learn strategic pricing.
python main_a2c_multi_agent.py \
    --city nyc_man_south \
    --mode 2 \
    --checkpoint_path sanity_bound_d7_g5_meta_one \
    --wandb_group sanity_checks \
    --num_days 7 \
    --meta_policy one \
    --brand_momentum_gamma 5 \
    --low_level_scalar_min 0.4 \
    --low_level_scalar_max 0.6 \
    --max_episodes 25000
