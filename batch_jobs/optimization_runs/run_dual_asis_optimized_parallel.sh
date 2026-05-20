#!/bin/sh

### Job Name:
#BSUB -J run_replication_parallel

### Queue Name:
#BSUB -q hpc

### Requesting 4 CPU cores, 4GB memory per core
#BSUB -n 16
#BSUB -R "rusage[mem=4GB]"

### Setting a runtime limit of 2 hours
#BSUB -W 24:00

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

### multi agent
python main_a2c_multi_agent_parallel.py --n_workers 14 --max_episodes 100000 --city nyc_man_south --mode 2 --checkpoint_path my_dual_agent_parallel

