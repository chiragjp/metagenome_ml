#!/bin/bash

# $1 = raw reads file 1 
# $2 = raw reads file 2
# $3 = output prefix

metaphlan2.py $1,$2 --bowtie2out $3_bowtie2.bz2 --nproc 14 --input_type fastq > $3_profiled.txt