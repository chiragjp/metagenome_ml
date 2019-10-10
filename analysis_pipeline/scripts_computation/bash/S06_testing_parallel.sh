#!/bin/bash
analyses=( "age_at_collection" "abx_usage" "exclusive_bf" "delivery_type" "sex" "country" )
#analyses=( "age_at_collection" )
predictors=( "demographics" "taxa" "demo+taxa" "cags" "demo+cags" "genes" "demo+genes" "pathways" "demo+pathways" "mixed" "demo+mixed" )
#predictors=( "mixed" "demo+mixed" )
ALGOS=( "glmnet" "glmnet2" "gbm" "gbm2" "rf" "rf2" "svmLinear" "svmPoly" "svmRadial" "knn" "nb" )
#ALGOS=( "glmnet" )
ALGOS_regression=( "glmnet" "glmnet2" "gbm" "gbm2" "rf" "rf2" "svmLinear" "svmPoly" "svmRadial" "knn" )
#ALGOS_regression=( )
ALGOS_genes=( "glmnet" "glmnet2" "gbm" "gbm2" "rf" "rf2" )
#ALGOS_genes=( "gbm2" )
folds=( "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" )
#folds=( "1" )
sets=( "train" "test" )
#sets=( "test" )
for analysis in "${analyses[@]}"
do
for predictor in "${predictors[@]}"
do
#decide which algos will be run
if [ $predictor = "genes" ] || [ $predictor = "demo+genes" ]; then
    algos=("${ALGOS_genes[@]}")
elif [ $analysis = "age_at_collection" ]; then
    algos=("${ALGOS_regression[@]}")
else
    algos=("${ALGOS[@]}")
fi
for algo in "${algos[@]}"
do
#job parameters
memory=200
for fold in "${folds[@]}"
do
for set in "${sets[@]}"
do
if [ $set = "train" ]; then
    if [ $algo = "nb" ] || [ $algo = "knn" ]; then time=30; else time=2; fi
else
    if [ $algo = "nb" ] || [ $algo = "knn" ]; then time=8; else time=1; fi
fi
#time=15
#time=$(( 3*$time ))
job_name="ts-$analysis-$predictor-$algo-$fold-$set.job"
out_file="../eo/ts-$analysis-$predictor-$algo-$fold-$set.out"
err_file="../eo/ts-$analysis-$predictor-$algo-$fold-$set.err"
#only run job if the prediction has not already been saved
if [ ! -f /n/scratch2/al311/Aging/Microbiome/data/pred_class_${set}_${analysis}_${predictor}_${algo}_${fold}.Rda ] && [ ! -f /n/scratch2/al311/Aging/Microbiome/data/pred_${set}_${analysis}_${predictor}_${algo}_${fold}.Rda ]; then
   echo "pred_class_${set}_${analysis}_${predictor}_${algo}_${fold}.Rda has not been computed."
#   echo todo
   sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c 1 -t $time testing.sh $analysis $predictor $algo $fold $set
#else
#   echo ok
#   echo "pred_${set}_${analysis}_${predictor}_${algo}_${fold}.Rda has already been computed."
fi
done
done
done
done
done
echo finished


