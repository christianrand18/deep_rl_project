#!/bin/sh

### Job Name:
#BSUB -J eval_picardtrained_7day

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

### Fairness check: train the meta-policy with PICARD (the fast, denoised
### scheme), then evaluate it on 100 plain stochastic days. Pair with
### eval_seqtrained_7day (same episode budget, same eval seeds). If Picard's
### eval profit matches or beats sequential -> Picard is a legitimate fast
### scheme. If it underperforms -> the training-time gains were an artifact of
### denoising that does not transfer to real stochastic deployment.
python main_a2c_multi_agent.py \
    --city nyc_man_south \
    --mode 2 \
    --checkpoint_path eval_picardtrained_7day \
    --wandb_group picard_eval \
    --num_days 7 \
    --meta_policy one \
    --brand_momentum_gamma 1.0 \
    --parallel_days \
    --picard_update_strategy anderson \
    --picard_anderson_m 5 \
    --picard_max_iters 10 \
    --picard_tol 5e-3 \
    --max_episodes 20000 \
    --final_stochastic_eval 100
