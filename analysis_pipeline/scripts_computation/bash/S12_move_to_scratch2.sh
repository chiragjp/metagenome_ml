#!/bin/bash
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu
#SBATCH --error=../eo/mts2.err
#SBATCH --output=../eo/mts2.out
#SBATCH --job-name=mts2.job
#SBATCH --mem-per-cpu=1G
#SBATCH -t 600
#SBATCH -c 1
mv ../data/model_* /n/scratch2/al311/Aging/Microbiome/data/

