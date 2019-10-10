#!/bin/bash

#$1 = basename of input file to be aligned (i.e. SRR1234567) 
#$2 = number of threads to be used

#requires samtools and bowtie2 to be in the path

#actually run the alignment
fasta_filename=$1
read1_filename=${fasta_filename%.*}_1.fastq
read2_filename=${fasta_filename%.*}_2.fastq
fasta_filename=${read1_filename%_*}
num_threads=$2

#
# Run alignment
bam_filename=${fasta_filename}'.catalog.bam'
echo "Starting Alignment." >&2
bowtie2 \
    -p ${num_threads} \
    -D 20 \
    -R 3 \
    -N 1 \
    -L 20 \
    -i S,1,0.50 \
    -x $3 \
    --local \
    -q \
    --quiet \
    --mm \
    '-1' ${read1_filename} \
    '-2' ${read2_filename} \
    | samtools view -T $3 -b -h -o ${bam_filename} -

#
# Sort the bam file
echo 'Sorting the bam file' >&2
samtools sort -l 9 -o ${bam_filename%.*}'.sorted.bam' -O bam -@ ${num_threads} ${bam_filename}

bam_filename_sorted=${bam_filename%.*}'.sorted.bam'

#
# Index the bam
echo 'Indexing the bam file' >&2
bam_index_filename=${bam_filename%.*}'.bai'
samtools index -b ${bam_filename_sorted} ${bam_index_filename}

#rm $1
rm $bam_filename
rm $read1_filename
rm $read2_filename

# Finished.
echo 'Finished' >&2
