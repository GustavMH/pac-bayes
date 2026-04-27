#!/usr/bin/env sh

#SBATCH --time=07:20:00
#SBATCH --gres=gpu:titanrtx

source ~/venv/bin/activate

python models/eurosat.py
