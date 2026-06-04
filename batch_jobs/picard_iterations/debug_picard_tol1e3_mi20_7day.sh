#!/bin/sh

### Job Name:
#BSUB -J debug_picard_tol1e3

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

### Tighter-tolerance Picard: tol 1e-3 (vs 5e-3) + max_iters 20 (vs 10). Forces the
### fixed point to grind closer to this episode's exact (noisy) trajectory instead of
### stopping at K=1 on the smooth warm-started state. A/B against debug_dump_picard_7day
### (identical except tol 5e-3 / max_iters 10) and debug_dump_seq_7day.
### Prediction: as tol tightens, K rises and meta_critic_loss drifts UP toward the
### sequential level (~0.67) — i.e. tighter convergence => closer to sequential, less
### denoising benefit. Watch debug/picard_K_used and agent0/meta_critic_loss.
python main_a2c_multi_agent.py \
    --city nyc_man_south \
    --mode 2 \
    --checkpoint_path debug_picard_tol1e3_mi20_7day \
    --wandb_group picard_debug \
    --num_days 7 \
    --meta_policy one \
    --brand_momentum_gamma 1.0 \
    --parallel_days \
    --picard_update_strategy anderson \
    --picard_anderson_m 5 \
    --picard_max_iters 20 \
    --picard_tol 1e-3 \
    --max_episodes 100000
