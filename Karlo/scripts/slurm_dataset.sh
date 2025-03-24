#!/bin/bash
#SBATCH --job-name=dataset_generation
#SBATCH --output=/home/kmrakovcic/Results/LIV/dataset_generation.txt
#SBATCH --partition=computes_thin
#SBATCH --cpus-per-task=24
#SBATCH --mem=1G
#SBATCH --ntasks=1
#SBATCH --time=336:00:00
hostname
srun python3 /home/kmrakovcic/Projects/ANN-VIL/Karlo/scripts/dataset_generation.py \
--examples_num 1000 \
--opacity_file "/home/kmrakovcic/Projects/ANN-VIL/Karlo/extra/Opacity_grid_1000x1000.npz" \
--output_suffix "0"