#!/bin/sh

### Job Name:
#BSUB -J sanity_normfix_d7_g5_meta_one_unbound

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
module load python3/3.11.13
. .venv/bin/activate

### Unbounded companion to sanity_normfix_d7_g5_meta_one.
### Identical config but without low_level_scalar bounds, so the low-level
### can fully compensate. Direct comparison isolates whether bounded-rho
### changes what strategy the meta learns.
python main_a2c_multi_agent.py \
    --city nyc_man_south \
    --mode 2 \
    --checkpoint_path sanity_normfix_d7_g5_meta_one_unbound \
    --wandb_group sanity_checks \
    --num_days 7 \
    --meta_policy one \
    --brand_momentum_gamma 5 \
    --max_episodes 25000
