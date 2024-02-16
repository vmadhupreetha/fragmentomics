#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH -c 12
#SBATCH --partition gpu
#SBATCH --gpus-per-node=quadro_rtx_6000:1
#SBATCH --mem=25G
#SBATCH --job-name=pretrained-cnn

source activate fragenv
python3 ../run_pretrained_model.py > ../sbatch_logs/run_pretrained_cnn_weight_decay_models.log