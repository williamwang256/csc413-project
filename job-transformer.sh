#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --partition=csc413
#SBATCH --job-name 413-project
#SBATCH --gres=gpu
#SBATCH --output=413-project_%j.out

source venv/bin/activate
export CSC413_PROJECT_DIR="$PWD"
MODEL="model"

time python3 src/transformer.py -t $MODEL -a $MODEL -e $MODEL -p $MODEL
