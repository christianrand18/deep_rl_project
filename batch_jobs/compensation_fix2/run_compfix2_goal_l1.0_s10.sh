#!/bin/sh

### Job Name:
#BSUB -J compfix2_goal_l1.0_s10

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

### run experiment
python main_a2c_multi_agent.py \
    --city nyc_man_south \
    --mode 2 \
    --checkpoint_path compfix2_goal_l1.0_s10 \
    --wandb_group compensation_fix2 \
    --num_days 7 \
    --brand_momentum_gamma 5 \
    --meta_policy one --meta_action_mode goal --meta_align_lambda 1.0 \
    --seed 10 \
    --max_episodes 25000
