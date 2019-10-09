#!/bin/bash
analyses=( "age_at_collection" "abx_usage" "exclusive_bf" "delivery_type" "sex" "country" )
#analyses=( "age_at_collection" )
predictors=( "demographics" "taxa" "demo+taxa" "cags" "demo+cags" "genes" "demo+genes" "pathways" "demo+pathways" "mixed" "demo+mixed" )
#predictors=( "demo+mixed" )
ALGOS=( "glmnet" "glmnet2" "gbm" "gbm2" "rf" "rf2" "svmLinear" "svmPoly" "svmRadial" "knn" "nb" )
#ALGOS=( "gbm" )
ALGOS_regression=( "glmnet" "glmnet2" "gbm" "gbm2" "rf" "rf2" "svmLinear" "svmPoly" "svmRadial" "knn" )
#ALGOS_regression=( )
ALGOS_genes=( "glmnet" "glmnet2" "gbm" "gbm2" "rf" "rf2" )
#ALGOS_genes=( "gbm2" )
folds=( "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" )
#folds=( "0" )
is_priority=false
multiply_time_by=1
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
for fold in "${folds[@]}"
do
#TIME
#time & predictor
if [ $predictor = "demographics" ]; then time=3; else time=10; fi
#time & analysis
if [ $analysis = "age_at_collection" ]; then
 time=$(( 3*$time ))
elif [ $analysis = "country" ]; then
  time=$(( 4*$time ))
fi
#time & algo
if [ $algo = "gbm" ]; then
 time=$(( 2*$time )) 
elif [ $algo = "rf" ] || [ $algo = "svmPoly" ] || [ $algo = "nb" ]; then
 time=$(( 2*$time ))
elif [ $algo = "svmLinear" ] || [ $algo = "svmPoly" ]; then
 time=$(( 2*$time ))
fi
#multiply time for the subset of jobs that hit the time wall
#time=$(( $multiply_time_by*$time ))
#time=6000
#CORES
if [ $algo = "gbm" ] || [ $algo = "svmPoly" ] || [ $algo = "nb" ]; then
 if [ $analysis = "age_at_collection" ] || [ $analysis = "country" ]; then
  n_cores=11; n_cores_R=10
 else
   n_cores=11; n_cores_R=10
 fi
elif [ $algo = "gbm2" ]; then
   n_cores=11; n_cores_R=10
else
 n_cores=1; n_cores_R=1 
fi
#MEMORY
if [ $predictor = "demographics" ]; then
 memory=300
else
 memory=1G
fi
#JOB TYPE
if [ $is_priority = true ]; then type=priority; elif (( $time > 1440 )); then type=medium; else type=short; fi
job_name="tr-$analysis-$predictor-$algo-$fold.job"
out_file="../eo/tr-$analysis-$predictor-$algo-$fold.out"
err_file="../eo/tr-$analysis-$predictor-$algo-$fold.err"
#only run job if the prediction has not already been saved
if [ ! -f /n/scratch2/al311/Aging/Microbiome/data/hyperparameters_${analysis}_${predictor}_${algo}_${fold}.Rda ]; then
#   echo "hyperparameters_${analysis}_${predictor}_${algo}_${fold}.Rda has not been computed."
#   echo todo
   sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -t $time -p $type training.sh $analysis $predictor $algo $fold $n_cores_R
#else
#   echo ok
#   echo "hyperparameters_${analysis}_${predictor}_${algo}_${fold}.Rda has already been computed."
fi
done
done
done
done
echo finished




