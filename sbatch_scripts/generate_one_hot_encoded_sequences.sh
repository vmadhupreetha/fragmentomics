#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH -c 12
#SBATCH --partition gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=24G
#SBATCH --job-name=train-model

source activate fragenv
python3 ../generate_one_hot_encoded_sequences.py > ../sbatch_logs/generate_one_hot_encoded_sequences_training_final.log