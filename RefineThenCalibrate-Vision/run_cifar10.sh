#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

source venv_cv/bin/activate
python main.py \
    --run_name run_${SLURM_ARRAY_TASK_ID}_cifar10 \
    --seed ${SLURM_ARRAY_TASK_ID} \
    --seeds_per_job 10 \
    --dataset cifar10 \
    --folder cifar10_models \
    --project "PAC-Bayes NeurIPS 2025"
