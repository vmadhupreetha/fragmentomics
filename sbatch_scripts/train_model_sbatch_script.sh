#!/bin/bash

#SBATCH --time=3:00:00
#SBATCH -c 12
#SBATCH --partition gpu
#SBATCH --gpus-per-node=quadro_rtx_6000:1
#SBATCH --mem=35G
#SBATCH --job-name=train-model

source activate fragenv
python3 ../train_combined_model.py > ../sbatch_logs/combined_model_all_tracks.log