#!/usr/bin/env sh

#SBATCH --time=01:00:00
#SBATCH --mem=32G

source ~/venv/bin/activate

python models/eurosat.py
