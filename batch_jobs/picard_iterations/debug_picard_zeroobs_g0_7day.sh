#!/bin/sh

### Job Name:
#BSUB -J debug_picard_zeroobs_g0

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

### Debug: Picard, meta-policy blinded (zero obs), brand momentum off (gamma=0).
### Pair with debug_seq_zeroobs_g0_7day. With obs blinded and gamma=0, the only
### things left that could differ are the action/reward generation themselves.
### If gap vs sequential vanishes -> the cause was the obs representation.
### If gap persists -> picard's returns are intrinsically lower-variance.
python main_a2c_multi_agent.py \
    --city nyc_man_south \
    --mode 2 \
    --checkpoint_path debug_picard_zeroobs_g0_7day \
    --wandb_group picard_debug \
    --num_days 7 \
    --meta_policy one \
    --brand_momentum_gamma 0 \
    --parallel_days \
    --picard_update_strategy anderson \
    --picard_anderson_m 5 \
    --picard_max_iters 10 \
    --picard_tol 5e-3 \
    --meta_zero_obs \
    --max_episodes 100000
