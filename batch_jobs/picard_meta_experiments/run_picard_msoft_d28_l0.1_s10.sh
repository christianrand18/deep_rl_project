#!/bin/sh

#BSUB -J picard_msoft_d28_s10
#BSUB -q hpc
#BSUB -n 32
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
    --checkpoint_path picard_msoft_d28_l0.1_s10 \
    --wandb_group picard_meta_experiments \
    --num_days 28 \
    --meta_policy one \
    --meta_agent 0 \
    --fix_agent 1 \
    --max_episodes 200000 \
    --reward_scalar 2000 \
    --meta_action_mode multiplier_soft \
    --meta_reg_lambda 0.1 \
    --parallel_days \
    --picard_update_strategy anderson \
    --picard_anderson_m 5 \
    --picard_tol 0.005 \
    --picard_max_iters 10 \
    --picard_parallel_workers 28
