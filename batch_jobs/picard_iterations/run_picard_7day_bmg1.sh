#!/bin/sh

### Job Name:
#BSUB -J picard_7day_bmg1

### Queue Name:
#BSUB -q hpc

### Requesting 4 CPU cores, 4GB memory per core
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"

### Setting a runtime limit
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

### Picard: 7-day episode, bm_gamma=1 — moderate brand momentum coupling
python main_a2c_multi_agent.py \
    --city nyc_man_south \
    --mode 2 \
    --checkpoint_path picard_7day_bmg1 \
    --wandb_group picard_iterations \
    --num_days 7 \
    --meta_policy one \
    --brand_momentum_gamma 1.0 \
    --parallel_days \
    --picard_max_iters 10 \
    --picard_tol 1e-3 \
    --max_episodes 100000
