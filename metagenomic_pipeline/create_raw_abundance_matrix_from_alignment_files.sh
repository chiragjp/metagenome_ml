#!/bin/bash

cd realigned

for file in ./*bam;
do

echo "${file:2:-4}" > "${file}_processed"
echo "${file:2:-4}" > "${file}_misalignments"
samtools idxstats $file | cut -f 3 >> "${file}_processed"
samtools idxstats $file | cut -f 4 >> "${file}_misalignments"

done

echo 'GENENAME  GENELENGTH' > out
samtools idxstats $file | cut -f 1,2 >> out
paste -d '\t' out *processed* > raw_abundance_matrix.csv
head -n-1 raw_abundance_matrix.csv > foo
mv foo raw_abundance_matrix.csv

