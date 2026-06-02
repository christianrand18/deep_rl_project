#!/bin/sh

### Job Name:
#BSUB -J debug_dump_picard

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

mkdir -p batch_jobs/logs saved_files/meta_dumps
export CPLEX_PATH="/apps/cplex/cplex1210/opl/bin/x86-64_linux/"

### activate environment
. .venv/bin/activate

### Dump every committed meta (obs, action, value, return) tuple to jsonl, so we can
### inspect WHEN and WHY the obs->return structure diverges from sequential. The 100-ep
### local test showed the two are identical early; the critic-loss gap is emergent over
### thousands of episodes. This run captures the emergence. Pair with debug_dump_seq_7day.
### Analyse offline: python batch_jobs/picard_iterations/analyze_dump.py <ep_min> <picard.jsonl> <seq.jsonl>
rm -f saved_files/meta_dumps/picard.jsonl
python main_a2c_multi_agent.py \
    --city nyc_man_south \
    --mode 2 \
    --checkpoint_path debug_dump_picard_7day \
    --wandb_group picard_debug \
    --num_days 7 \
    --meta_policy one \
    --brand_momentum_gamma 1.0 \
    --parallel_days \
    --picard_update_strategy anderson \
    --picard_anderson_m 5 \
    --picard_max_iters 10 \
    --picard_tol 5e-3 \
    --max_episodes 100000 \
    --dump_meta_path saved_files/meta_dumps/picard.jsonl
