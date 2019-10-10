#!/bin/bash
analyses=( "age_at_collection" "abx_usage" "exclusive_bf" "delivery_type" "sex" "country" )
#analyses=( "age_at_collection" )
predictors=( "demographics" "taxa" "demo+taxa" "cags" "demo+cags" "genes" "demo+genes" "pathways" "demo+pathways" "mixed" "demo+mixed" )
#predictors=( "demo+pathways" )
sets=( "train" "test" )
#sets=( "train" )
for analysis in "${analyses[@]}"
do
for predictor in "${predictors[@]}"
do
for set in "${sets[@]}"
do
job_name="ps-$analysis-$predictor-$set.job"
out_file="./../eo/ps-$analysis-$predictor-$set.out"
err_file="./../eo/ps-$analysis-$predictor-$set.err"
memory=1G
n_cores=1
if [ $analysis == "age_at_collection" ]; then
time=15
else
if [ $predictor = "demographics" ]; then time=80; else time=100; fi
fi
if [ $set = "train" ]; then
time=$(( 2*$time ))
fi
#time=400
sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -t $time  postprocessing.sh $analysis $predictor $set
done
done
done
