#Step 05: Trains and tunes the models

#extract and use info from command line
args = commandArgs(trailingOnly=TRUE)
exclude_other_predictors_from_analysis = FALSE

#default, if no command line received
if(length(args) == 0){args <- c("Surv", "mixed", "glmnet2", "0", "1")}
if(length(args) != 5)
{
  stop("use five arguments for analysis")
} else {
  analysis <- args[1]
  predictors <- args[2]
  algo <- args[3]
  i <- args[4]
  n_cores <- as.numeric(args[5])
}

#set path
if(getwd() == "/Users/Alan"){
  path_store <- "~/Desktop/Aging/Microbiome/data/"
  path_compute <- "~/Desktop/Aging/Microbiome/data/"
}else {
  path_store <- "/n/groups/patel/Alan/Aging/Microbiome/data/"
  path_compute <- "/n/scratch2/al311/Aging/Microbiome/data/"}
source(paste(path_store, "../scripts/Helpers_microbiome.R", sep = ""))

#register n_cores for parallel computing. Actually, granted one more core on O2, for master core. n_cores is the number of parallel cores, so n_cores=n_total-1.
registerDoParallel(n_cores)
if(algo== "same"){algo <- predictors} #when analyzing at the gene level, use the genes that were selected for the algorithm.

#set target variable
target <- target_of_analysis(analysis)
#define metric
metric <- define_metric(analysis)
#print
print("Analysis:"); print(analysis); print("Target:"); print(target); print("Predictors:"); print(predictors)
#load data
if (predictors=="genes"){
  string <- algo
} else if (predictors=="demo+genes"){
  string <- paste("demo", sep="+", algo)
} else {
  string <- predictors
}
preprocessed_data <- readRDS(paste(path_compute, "preprocessed_data_", analysis, "_", string, "_", i, ".Rda", sep = ""))
data <- preprocessed_data$data_train
names(data)[which(names(data) == target)] <- "target"
x <- preprocessed_data$x_train
y <- preprocessed_data$y_train
w <- generate_weights(y)
if(!(analysis %in% c("age_at_collection", "country", "HvsFDvsD", "Surv")))
{
  data$target <- make.names(data$target)
  y <- make.names(y)
}


#run analysis
if(analysis=="Surv")
{
  if(algo == "glmnet2") {
    model <- tune_surv_glmnet2(analysis, target, predictors, algo, i, data, x, y, w)
  } else if (algo == "gbm2"){
    model <- tune_surv_gbm2(analysis, target, predictors, algo, i, data, x, y, w, n_cores=n_cores, new_seed=30)
  } else if (algo == "rf2"){
    model <- tune_surv_rf2(analysis, target, predictors, algo, i, data, x, y, w, n_cores=n_cores)
  }
} else if(target == "age_at_collection" & algo == "nb"){
  print("Naive Bayes is not available for regression.")
} else if(algo == "glmnet2") {
  model <- tune_glmnet2(analysis, target, predictors, algo, i, data, x, y, w)
} else if (algo == "gbm2"){
  model <- tune_gbm2(analysis, target, predictors, algo, i, data, x, y, w, n_cores=n_cores)
} else if (algo == "rf2"){
  model <- tune_rf2(analysis, target, predictors, algo, i, data, x, y, w)
} else {
  model <- tune_caret_models(analysis, target, predictors, algo, i, data, x, y, w)
}
if(!(target == "age_at_collection" & algo == "nb"))
{
  saveRDS(model, paste(path_compute, "model", "_", analysis, "_", predictors, "_", algo, "_", i,".Rda", sep = ""))
}

#print and save the best predictors if possible
if(analysis=="Surv")
{
  best_predictors_surv(model, analysis, predictors, algo, i)
} else {
  if(algo %in% c("glmnet", "glmnet2", "gbm", "gbm2", "rf", "rf2"))
  {
    best_predictors(model, analysis, predictors, algo, i)
  }
}
print("done")



