#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --mem=5G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=madhuvl96@gmail.com

python3 get_data_ver_2.py > make_data_sbatch_output