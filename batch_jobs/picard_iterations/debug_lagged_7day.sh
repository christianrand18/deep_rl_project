#!/bin/sh

### Job Name:
#BSUB -J debug_lagged_7day

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

### Debug: sequential with lagged meta-policy observations.
### Meta-policy sees previous episode's daily_state instead of current episode's,
### mirroring what Picard K=1 does with warm-started obs. Isolates the obs-lag effect.
### If this matches Picard → obs lag is the cause. If it matches baseline → it isn't.
python main_a2c_multi_agent.py \
    --city nyc_man_south \
    --mode 2 \
    --checkpoint_path debug_lagged_7day \
    --wandb_group picard_debug \
    --num_days 7 \
    --meta_policy one \
    --brand_momentum_gamma 1.0 \
    --lagged_meta_obs \
    --max_episodes 100000
