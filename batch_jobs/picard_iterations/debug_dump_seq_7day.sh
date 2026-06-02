#!/bin/sh

### Job Name:
#BSUB -J debug_dump_seq

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

### Sequential counterpart of debug_dump_picard_7day. Same config, no Picard. Captures
### the committed (obs, action, value, return) tuples so we can compare the obs->return
### structure against Picard at ep>1000 and see where/why the critic-loss gap emerges.
rm -f saved_files/meta_dumps/seq.jsonl
python main_a2c_multi_agent.py \
    --city nyc_man_south \
    --mode 2 \
    --checkpoint_path debug_dump_seq_7day \
    --wandb_group picard_debug \
    --num_days 7 \
    --meta_policy one \
    --brand_momentum_gamma 1.0 \
    --max_episodes 100000 \
    --dump_meta_path saved_files/meta_dumps/seq.jsonl
