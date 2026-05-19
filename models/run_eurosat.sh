#!/usr/bin/env sh

#SBATCH --time=07:20:00
#SBATCH --gres=gpu:titanrtx

source ~/venv/bin/activate

python models/eurosat.py \
    --model "resnet18"
    --scheduler "tri"
    --n-epochs 50
    --cycle-size 10
    --dataset "EuroSAT"
