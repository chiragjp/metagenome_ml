#!/bin/bash
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu
#SBATCH --error=../eo/merge_hyperparameters.err
#SBATCH --output=../eo/merge_hyperparameters.out
#SBATCH --job-name=merge_hyperparameters.job
#SBATCH --mem-per-cpu=1G
#SBATCH -t 15
#SBATCH -c 1
Rscript ../scripts/Merge_hyperparameters.R
