#!/bin/bash
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu
#SBATCH --error=../eo/compress_results.err
#SBATCH --output=../eo/compress_results.out
#SBATCH --job-name=compress_results.job
#SBATCH --mem-per-cpu=1G
#SBATCH -t 15
#SBATCH -c 1

cd /n/groups/patel/Alan/Aging/Microbiome/

if [ -f cd /n/groups/patel/Alan/Aging/Microbiome/data.tar.gz ]; then
 rm data.tar.gz
fi

tar -czvf data.tar.gz data

echo DONE
