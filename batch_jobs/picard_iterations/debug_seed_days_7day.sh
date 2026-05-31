#!/bin/sh

### Job Name:
#BSUB -J debug_seed_days_7day

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

### Debug: sequential with per-day RNG seeding (np.random + torch + env._shuffle_rng).
### Mirrors Picard's prepare_day seeding in isolation. No warm-start, no Picard structure.
### If this matches Picard → per-day seeding (variance reduction) is the cause.
### If this matches baseline → seeding alone does nothing and it's something else.
python main_a2c_multi_agent.py \
    --city nyc_man_south \
    --mode 2 \
    --checkpoint_path debug_seed_days_7day \
    --wandb_group picard_debug \
    --num_days 7 \
    --meta_policy one \
    --brand_momentum_gamma 1.0 \
    --seed_days \
    --max_episodes 100000
