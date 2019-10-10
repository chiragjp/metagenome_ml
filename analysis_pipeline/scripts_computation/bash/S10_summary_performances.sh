#!/bin/bash
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu
#SBATCH --error=../eo/summary_performances.err
#SBATCH --output=../eo/summary_performances.out
#SBATCH --job-name=summary_performances.job
#SBATCH --mem-per-cpu=1G
#SBATCH -t 15
#SBATCH -c 1
Rscript ./../scripts/Summary_performances.R
