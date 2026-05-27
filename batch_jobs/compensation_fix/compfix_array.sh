#!/bin/bash

### Job Name (array of 12: 3 seeds x {multiplier, cap, soft@0.1, goal@0.1})
#BSUB -J compfix[1-12]

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

### Output and error files (%J = job id, %I = array index)
#BSUB -o batch_jobs/logs/compfix_%J_%I.out
#BSUB -e batch_jobs/logs/compfix_%J_%I.err


### cd to repo dir
cd ~/deep_rl_project

mkdir -p batch_jobs/logs
export CPLEX_PATH="/apps/cplex/cplex1210/opl/bin/x86-64_linux/"

### activate environment
. .venv/bin/activate

### Compensation-fix screening matrix (Round 1, 12 runs).
### Purpose: pick ONE meta_action_mode to merge. All runs UNBOUNDED (no
### --low_level_scalar_min/max) so compensation is free to act; --meta_action_mode
### is the only structural difference. λ=0.1 fixed; the {0.03, 0.3} sweep is a
### conditional follow-up, run only if soft/goal looks weak/borderline (see
### Investigations/2026-05-25_compensation-fix-spec.md). job_type is derived from
### (mode, λ) in main, so W&B nests group -> job_type -> seed automatically.
IDX=$((LSB_JOBINDEX - 1))

#       1           2           3           4    5    6    7     8     9     10    11    12
MODES=(multiplier multiplier multiplier cap  cap  cap  soft  soft  soft  goal  goal  goal)
SEEDS=(10          20          30          10   20   30   10    20    30    10    20    30)
EXTRA=(""          ""          ""          ""   ""   ""   "--meta_reg_lambda 0.1"   "--meta_reg_lambda 0.1"   "--meta_reg_lambda 0.1"   "--meta_align_lambda 0.1"   "--meta_align_lambda 0.1"   "--meta_align_lambda 0.1")

MODE=${MODES[$IDX]}
SEED=${SEEDS[$IDX]}

case $MODE in
    soft) TAG="soft_l0.1" ;;
    goal) TAG="goal_l0.1" ;;
    *)    TAG="$MODE" ;;
esac
NAME="compfix_${TAG}_s${SEED}"

echo "Array index $LSB_JOBINDEX -> mode=$MODE seed=$SEED name=$NAME"

python main_a2c_multi_agent.py \
    --city nyc_man_south \
    --mode 2 \
    --checkpoint_path "$NAME" \
    --wandb_group compensation_fix \
    --num_days 7 \
    --meta_policy one \
    --brand_momentum_gamma 5 \
    --meta_action_mode "$MODE" \
    ${EXTRA[$IDX]} \
    --seed "$SEED" \
    --max_episodes 25000
