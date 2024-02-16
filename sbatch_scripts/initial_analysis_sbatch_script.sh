#!/bin/bash

#SBATCH --time=00:15:00
#SBATCH -c 12
#SBATCH --partition gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=55G
#SBATCH --job-name=initial_analysis

source activate fragenv
python3 ../one_time_use_side_scripts/enformer_output_initial_analysis_tsne_umap.py > ../sbatch_logs/initial_analysis_tsne.log