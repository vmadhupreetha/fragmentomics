#!/bin/bash

#SBATCH --time=30:00:00
#SBATCH -c 12
#SBATCH --partition gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=24G
#SBATCH --job-name=train-model

source activate fragenv
python3 ../test_patients_model.py > ../sbatch_logs/test_all_patients_part_2.log