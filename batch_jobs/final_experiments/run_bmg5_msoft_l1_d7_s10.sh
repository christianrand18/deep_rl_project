#!/bin/sh

#BSUB -J bmg5_msoft_l1_d7_s10
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 48:00
#BSUB -B
#BSUB -N
#BSUB -o batch_jobs/logs/Output_%J.out
#BSUB -e batch_jobs/logs/Output_%J.err

cd ~/deep_rl_project
mkdir -p batch_jobs/logs
export CPLEX_PATH="/apps/cplex/cplex1210/opl/bin/x86-64_linux/"
. .venv/bin/activate

python main_a2c_multi_agent.py \
    --city nyc_man_south \
    --mode 2 \
    --seed 10 \
    --checkpoint_path bmg5_msoft_l1_d7_s10 \
    --wandb_group final_experiments \
    --num_days 7 \
    --meta_policy one \
    --meta_agent 0 \
    --brand_momentum_gamma 5 \
    --meta_action_mode multiplier_soft \
    --meta_reg_lambda 1 \
    --max_episodes 100000 \
    --parallel_days \
    --picard_parallel_workers 7
