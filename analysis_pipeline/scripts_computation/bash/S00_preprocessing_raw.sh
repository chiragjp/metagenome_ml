#!/bin/bash
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu
#SBATCH --error=../eo/preprocessing_raw.err
#SBATCH --output=../eo/preprocessing_raw.out
#SBATCH --job-name=preprocessing_raw.job
#SBATCH --mem-per-cpu=1G
#SBATCH -t 30
#SBATCH -c 1

Rscript ../scripts/prepare_data_app_associations.R

