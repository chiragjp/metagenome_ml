#!/bin/bash
analyses=( "age_at_collection" "abx_usage" "exclusive_bf" "delivery_type" "sex" "country" )
#analyses=( "age_at_collection" )
sets=( "train" "test" )
#sets=( "test" )
types=( "Performance" "Performance_sd" )
#types=( "Performance" )
metrics_regression=( "R2" )
metrics_classification=( "ROC" "Cross_Entropy" "Mean_Accuracy" "Accuracy" "Sensitivity" "Specificity" )
metrics_country=( "Cross_Entropy" "Mean_Accuracy" "Accuracy" "SWE" "EST" "FIN" "RUS" )
for analysis in "${analyses[@]}"
do
if [ $analysis = "age_at_collection" ]; then
metrics=("${metrics_regression[@]}")
elif [ $analysis = "country" ]; then
metrics=("${metrics_country[@]}")
else
metrics=("${metrics_classification[@]}")
fi
for metric in "${metrics[@]}"
do
for set in "${sets[@]}"
do
for type in "${types[@]}"
do
job_name="mr-$analysis-$metric-$set-$type.job"
out_file="./../eo/mr-$analysis-$metric-$set-$type.out"
err_file="./../eo/mr-$analysis-$metric-$set-$type.err"
memory=200
n_cores=1
time=1
sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -t $time merge_results.sh $analysis $metric $set $type
done
done
done
done

