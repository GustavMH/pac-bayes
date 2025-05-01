#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

source venv_cv/bin/activate
python main.py \
    --run_name run_${SLURM_ARRAY_TASK_ID}_svhn \
    --seed ${SLURM_ARRAY_TASK_ID} \
    --seeds_per_job 10 \
    --dataset svhn \
    --folder svhn_models \
    --project "PAC-Bayes NeurIPS 2025"

# Run with --array 10,20,30
