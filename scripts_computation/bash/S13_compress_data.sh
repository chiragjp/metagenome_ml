#!/bin/bash
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu
#SBATCH --error=../eo/compress_data.err
#SBATCH --output=../eo/compress_data.out
#SBATCH --job-name=compress_data.job
#SBATCH --mem-per-cpu=1G
#SBATCH -t 600
#SBATCH -c 1

cd /n/scratch2/al311/Aging/Microbiome/

if [ -f /n/scratch2/al311/Aging/Microbiome/data_intermediate_results.tar.gz ]; then
 rm data_microbiome_intermediate_results.tar.gz
fi

tar -czvf data_microbiome_intermediate_results.tar.gz data
#mv /n/scratch2/al311/Aging/Microbiome/data_microbiome_intermediate_results.tar.gz cd /n/groups/patel/Alan/Aging/Microbiome/data_microbiome_intermediate_results.tar.gz

