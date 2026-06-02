#!/bin/sh

### Job Name:
#BSUB -J eval_seqtrained_7day

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

### Fairness check: train the meta-policy SEQUENTIALLY (the honest stochastic
### scheme), then evaluate it on 100 plain stochastic days. Pair with
### eval_picardtrained_7day (same episode budget, same eval seeds). The eval
### profit is the only apples-to-apples measure of deployed policy quality.
python main_a2c_multi_agent.py \
    --city nyc_man_south \
    --mode 2 \
    --checkpoint_path eval_seqtrained_7day \
    --wandb_group picard_eval \
    --num_days 7 \
    --meta_policy one \
    --brand_momentum_gamma 1.0 \
    --max_episodes 20000 \
    --final_stochastic_eval 100
