#!/bin/bash

#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1

source venv/bin/activate
python kera-synthetic.py -n $SLURM_ARRAY_TASK_ID
