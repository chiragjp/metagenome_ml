**Instructions to run the pipeline to regenerate the results and display them.**

*For the manuscript, we ran all analyses in R V3.4.4

The results are generated using the computation pipeline (folder 'scripts_computation') and are displayed using the shiny app (folder 'scripts_shiny_app').

The computation pipeline is constituted of several steps that run several dozens of thousands of jobs in parallel.
The scripts in the 'bash' folder are used to sequentially run these different steps in parallel by calling the R scripts in the 'scripts' folders, to which they are feeding the different parameters (e.g combination of target, algorithm, predictor, cv fold)

The 14 different steps are:

Step 00: Step 0: S00_preprocessing_raw.sh
Preprocesses the data for 'Associations between variables' tab of the shiny app

Step 01: S01_preprocessing_parallel.sh
Preprocesses the data

Step 02: S02_preprocessing_folds_parallel.sh
Preprocesses each CV fold of the data

Step 03: S03_preprocessing_demographics_and_mixed_parallel.sh
Preprocesses the datasets for which predictors=demographics, and without all predictors types combined

Step 04: S04_preprocessing_nodemo_parallel.sh
Preprocesses the datasets without demographics variables

Step 05: S05_training_parallel.sh
Trains and tunes the models

Step 06: S06_testing_parallel.sh
Generates the predictions on both the training and the testing sets

Step 07: S07_postprocessing_parallel.sh
Calculates the R2/accuracy/ROC/cross-entropy/Concordance Index for each fold and on the entire dataset

Step 08: S08_merge_hyperparameters.sh
Summarizes for each fold the hyperparameters selection results, along with the accuracies and sample sizes on each fold. The results will be used in the "Folds tuning" tab of the shiny app

Step 09: S09_merge_results_parallel.sh
Merges the prediction performances for each target, predictors and algorithm

Step 10: S10_summary_performances.sh
Produces summary tables of the results

Step 11: S11_compress_results.sh (optional)
Compresses the data to be displayed using the shiny app

Step 12: S12_move_to_scratch2.sh
Move the voluminous data (the models) to a different folder

Step 13: S13_compress_data.sh (optional)
Compresses the voluminous data that is not useful to the shiny app (large file, > 1TB)


The results can subsequently be given to the shiny app to display the results interactively (put the data generated in the "data" folder of the shiny app)

Libraries used in analysis:
boot
caret
caTools
dplyr
doParallel
DT
dummies
e1071
gbm
ggplot2
glmnet
MLmetrics
nnet
openxlsx
pROC
randomForest
randomForestSRC
shiny
shinyBS
shinydashboard
shinyLP
survcomp
survival
