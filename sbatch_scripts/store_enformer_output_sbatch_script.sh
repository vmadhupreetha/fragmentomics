#!/bin/bash

#SBATCH --time=50:00:00
#SBATCH -c 12
#SBATCH --partition gpu
#SBATCH --gpus-per-node=7g.79gb:1
#SBATCH --mem=48G
#SBATCH --job-name=train

source activate fragenv
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64

python3 /hpc/compgen/projects/fragclass/analysis/mvivekanandan/script/madhu_scripts/storeEnformerOutput.py > ../sbatch_logs/validation_halfmil_attempt_2.log