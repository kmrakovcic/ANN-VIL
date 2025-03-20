#!/bin/bash
#SBATCH --job-name=opacity_grid
#SBATCH --output=/home/kmrakovcic/Results/LIV/opacity_grid.txt
#SBATCH --partition=computes_thin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=336:00:00
srun hostname
srun date
srun python3 /home/kmrakovcic/Projects/ANN-VIL/scripts/opacity_grid_generation.py \
--E_num 1000 \
--Eqg_num 1000