#initiate dataframes to stores the values fo the hyperparameters during processing.
path <- ifelse(getwd() == "/Users/Alan", "~/Desktop/Aging/Microbiome/data/", "/n/groups/patel/Alan/Aging/Microbiome/data/")
source(paste(path, "../scripts/Helpers_microbiome.R", sep = ""))

for (predictors in predictorsS)
{
  for (analysis in analyses)
  {
    if(analysis == "Surv")
    {
      for (algo in algos_surv)
      {
        if(algo=="glmnet") {
          hyperparameters <- initiate_store(seq(0,10), c("lambda"))*NA
        } else if (algo=="gbm") {
          hyperparameters <- initiate_store(seq(0,10), c("n.trees"))*NA
        } else if (algo=="rf") {
          hyperparameters <- initiate_store(seq(0,10), c("mtry", "nodesize"))*NA
        }
        saveRDS(hyperparameters, paste(path, "hyperparameters", "_", analysis, "_", predictors, "_", algo, ".Rda", sep = ""))
      }
    } else if (predictors == "genes"){
      for (algo in algos_genes)
      {
        if(algo=="glmnet") {
          hyperparameters <- initiate_store(seq(0,10), c("lambda", "alpha"))*NA
        } else if (algo=="glmnet2") {
          hyperparameters <- initiate_store(seq(0,10), c("lambda"))*NA
        } else if (algo=="gbm") {
          hyperparameters <- initiate_store(seq(0,10), c("n.trees", "interaction.depth", "shrinkage", "n.minobsinnode"))*NA
        } else if (algo=="gbm2") {
          hyperparameters <- initiate_store(seq(0,10), c("n.trees"))*NA
        } else if (algo=="rf") {
          hyperparameters <- initiate_store(seq(0,10), c("mtry"))*NA
        } else if (algo=="rf2") {
          hyperparameters <- initiate_store(seq(0,10), c("mtry"))*NA
        }
        saveRDS(hyperparameters, paste(path, "hyperparameters", "_", analysis, "_", predictors, "_", algo, ".Rda", sep = ""))
      }
    } else {
      for (algo in algos)
      {
        if(algo=="glmnet") {
          hyperparameters <- initiate_store(seq(0,10), c("lambda", "alpha"))*NA
        } else if (algo=="glmnet2") {
          hyperparameters <- initiate_store(seq(0,10), c("lambda"))*NA
        } else if (algo=="gbm") {
          hyperparameters <- initiate_store(seq(0,10), c("n.trees", "interaction.depth", "shrinkage", "n.minobsinnode"))*NA
        } else if (algo=="gbm2") {
          hyperparameters <- initiate_store(seq(0,10), c("n.trees"))*NA
        } else if (algo=="rf") {
          hyperparameters <- initiate_store(seq(0,10), c("mtry"))*NA
        } else if (algo=="rf2") {
          hyperparameters <- initiate_store(seq(0,10), c("mtry"))*NA
        } else if (algo=="svmLinear") {
          hyperparameters <- initiate_store(seq(0,10), c("C"))*NA
        } else if (algo=="svmPoly") {
          hyperparameters <- initiate_store(seq(0,10), c("sigma", "C"))*NA
        } else if (algo=="svmRadial") {
          hyperparameters <- initiate_store(seq(0,10), c("sigma", "C"))*NA
        } else if (algo=="knn") {
          hyperparameters <- initiate_store(seq(0,10), c("k"))*NA
        } else if (algo=="nb") {
          hyperparameters <- initiate_store(seq(0,10), c("fL", "usekernel", "adjust"))*NA
        }
        saveRDS(hyperparameters, paste(path, "hyperparameters", "_", analysis, "_", predictors, "_", algo, ".Rda", sep = ""))
      }
    }
  }
}

