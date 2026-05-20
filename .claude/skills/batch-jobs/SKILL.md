---
name: batch-jobs
description: Create LSF batch job scripts for running experiments on the HPC cluster. Use whenever the user wants to create a new experiment run, add a batch script, set up a new experiment round, or submit jobs. Also trigger for "write a job script", "new experiment", "run this on the cluster", "add a batch job", or anything about LSF/bsub job submission.
---

# Batch Job Creator

Create LSF batch scripts for running experiments on the DTU HPC cluster.

## Structure

All batch scripts live under `batch_jobs/`, grouped by experiment round:

```
batch_jobs/
├── replication/
├── optimization_runs/
├── experiments_round_1/
├── experiments_round_2/
└── <new_group>/
    ├── README.md
    └── run_<name>.sh
```

When creating a new job, either add to an existing group or create a new one. Each group gets a short README.md listing the scripts and what they test. Look at existing READMEs in `batch_jobs/` for the style — they're brief (5-10 lines).

## Template

```sh
#!/bin/sh

### Job Name:
#BSUB -J <job_name>

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

### run experiment
python main_a2c_multi_agent.py \
    --city nyc_man_south \
    --mode 2 \
    --checkpoint_path <unique_checkpoint_name> \
    --wandb_group <group_name> \
    --max_episodes 100000 \
    <additional flags>
```

## Key rules

- **Always pass `--wandb_group`** matching the experiment group folder name so runs are grouped in W&B.
- **`--checkpoint_path`** must be unique per run — encode the key config in the name (e.g., `dual_meta_one_bmg05_r2`).
- For HRL runs add: `--num_days 7 --meta_policy one --brand_momentum_gamma <value>`.
- Check `python main_a2c_multi_agent.py --help` or read the argparse in `main_a2c_multi_agent.py` if unsure about available flags.
- The wall time (`-W`) is 48h by default. Runs may still hit the limit and crash — that's expected.
