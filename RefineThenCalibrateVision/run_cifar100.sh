#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu26fl

source ~/pac-bayes/venv/bin/activate
python main.py \
    --run_name run_${SLURM_ARRAY_TASK_ID}_cifar100 \
    --seed ${SLURM_ARRAY_TASK_ID} \
    --seeds_per_job 10 \
    --dataset cifar100 \
    --folder cifar100_models \
    --project "PAC-Bayes NeurIPS 2025"
