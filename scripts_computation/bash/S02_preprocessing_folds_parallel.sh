#!/bin/bash
analyses=( "age_at_collection" "abx_usage" "exclusive_bf" "delivery_type" "sex" "country" )
analyses=( "age_at_collection" )
predictors=( "taxa" "cags" "pathways" "glmnet" "glmnet2" "gbm" "gbm2" "rf" "rf2" )
predictors=( "taxa" )
folds=( "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" )
#folds=( "0" )
memory=1G
n_cores=1
for analysis in "${analyses[@]}"
do
echo ANALYSIS
echo $analysis
for predictor in "${predictors[@]}"
do
echo PREDICTORS
echo $predictor
if [ $analysis = "country" ]; then
time=60
elif [ $predictor = "rf2" ] && [ $analysis = "age_at_collection" ]; then
time=60
else
time=15
fi
for fold in "${folds[@]}"
do
#time=100
job_name="pf-$analysis-$predictor-$fold.job"
out_file="./../eo/pf-$analysis-$predictor-$fold.out"
err_file="./../eo/pf-$analysis-$predictor-$fold.err"
#only run job if the data has not already been saved
if [ ! -f /n/scratch2/al311/Aging/Microbiome/data/preprocessed_data_${analysis}_demo+${predictor}_${fold}.Rda ]; then
#   echo "preprocessed_data_${analysis}_demo+${predictor}_${fold}.Rda has not been computed."
   sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -t $time  preprocessing_folds.sh $analysis $predictor $fold
else
#   echo ok
   echo "preprocessed_data_${analysis}_demo+${predictor}_${fold}.Rda has already been computed."
fi
done
done
done

