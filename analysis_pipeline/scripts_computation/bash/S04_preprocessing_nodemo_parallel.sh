#!/bin/bash
analyses=( "age_at_collection" "abx_usage" "exclusive_bf" "delivery_type" "sex" "country" )
#analyses=( "age_at_collection" )
folds=( "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" )
#folds=( "0" )
memory=300
n_cores=1
time=3
for analysis in "${analyses[@]}"
do
for fold in "${folds[@]}"
do
job_name="pn-$analysis-$fold.job"
out_file="../eo/pn-$analysis-$fold.out"
err_file="../eo/pn-$analysis-$fold.err"
#only run job if the data has not already been saved
if [ ! -f /n/scratch2/al311/Aging/Microbiome/data/preprocessed_data_${analysis}_mixed_${fold}.Rda ]; then
#   echo "preprocessed_data_${analysis}_pathways_${fold}.Rda has not been computed."
   sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -t $time preprocessing_nodemo.sh $analysis $fold
else
   echo ok
#   echo "preprocessed_data_${analysis}_pathways_${fold}.Rda has already been computed."
fi
done
done

