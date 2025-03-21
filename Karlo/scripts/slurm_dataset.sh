#!/bin/bash
#SBATCH --job-name=dataset_generation
#SBATCH --output=/home/kmrakovcic/Results/LIV/dataset_generation.txt
#SBATCH --partition=computes_thin
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks=1
#SBATCH --time=336:00:00
hostname
srun python3 /home/kmrakovcic/Projects/ANN-VIL/Karlo/scripts/dataset_generation.py \
--E_num 1000 \
--interpolation_file "/home/kmrakovcic/Projects/ANN-VIL/Karlo/extra/Opacity_grid_100x100.npz" \
--output_suffix "0"