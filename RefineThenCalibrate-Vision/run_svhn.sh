#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=hendrixgpu03fl,hendrixgpu04fl,hendrixgpu06fl,hendrixgpu05fl,hendrixgpu08fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl,hendrixgpu14fl,hendrixgpu15fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl,hendrixgpu21fl,hendrixgpu24fl,hendrixgpu26fl

source venv_cv/bin/activate
python main.py \
    --run_name run_${SLURM_ARRAY_TASK_ID}_svhn \
    --seed ${SLURM_ARRAY_TASK_ID} \
    --seeds_per_job 10 \
    --dataset svhn \
    --folder svhn_models \
    --project "PAC-Bayes NeurIPS 2025"

# Run with --array 10,20,30
