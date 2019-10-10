#Step 08: Summarizes for each fold the hyperparameters selection results, along with the accuracies and sample sizes on each fold. The results will be used in the "Folds tuning" tab of the shiny app

#set path
if(getwd() == "/Users/Alan"){
  path_store <- "~/Desktop/Aging/Microbiome/data/"
  path_compute <- "~/Desktop/Aging/Microbiome/data/"
}else {
  path_store <- "/n/groups/patel/Alan/Aging/Microbiome/data/"
  path_compute <- "/n/scratch2/al311/Aging/Microbiome/data/"}
source(paste(path_store, "../scripts/Helpers_microbiome.R", sep = ""))

print("start")
#collect and merge the hyperparameters values
for (predictors in predictorsS)
{
  for (analysis in analyses)
  {
    if(analysis == "Surv")
    {
      for (algo in algos_surv)
      {
        if(algo=="glmnet2") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("lambda"))*NA
        } else if (algo=="gbm2") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("n.trees"))*NA
        } else if (algo=="rf2") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("mtry", "nodesize"))*NA
        }
        for (i in seq(0, N_CV_Folds))
        {
          error <- tryCatch(hyperparameters[as.character(i),] <- readRDS(paste(path_compute, "hyperparameters", "_", analysis, "_", predictors, "_", algo, "_", i,".Rda", sep = "")), error=function(err) "error")
          if(grepl("error", error))
          {
            print(paste("Predictors = ", predictors, ", analysis = ", analysis, ", algo = ", algo, ", fold = ", i, ", the hyperparameter(s) could not be found. The fold will not be used.", sep = ""))
            next
          }
        }
        saveRDS(hyperparameters, paste(path_compute, "hyperparameters", "_", analysis, "_", predictors, "_", algo, ".Rda", sep = ""))
      }
    } else if (predictors %in% c("genes", "demo+genes")){
      for (algo in algos_genes)
      {
        if(algo=="glmnet") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("alpha", "lambda"))*NA
        } else if (algo=="glmnet2") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("lambda"))*NA
        } else if (algo=="gbm") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("n.trees", "interaction.depth", "shrinkage", "n.minobsinnode"))*NA
        } else if (algo=="gbm2") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("n.trees"))*NA
        } else if (algo=="rf") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("mtry"))*NA
        } else if (algo=="rf2") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("mtry"))*NA
        }
        for (i in seq(0, N_CV_Folds))
        {
          error <- tryCatch(hyperparameters[as.character(i),] <- readRDS(paste(path_compute, "hyperparameters", "_", analysis, "_", predictors, "_", algo, "_", i,".Rda", sep = "")), error=function(err) "error")
          if(grepl("error", error))
          {
            print(paste("Predictors = ", predictors, ", analysis = ", analysis, ", algo = ", algo, ", fold = ", i, ", the hyperparameter(s) could not be found. The fold will not be used.", sep = ""))
            next
          }
        }
        saveRDS(hyperparameters, paste(path_compute, "hyperparameters", "_", analysis, "_", predictors, "_", algo, ".Rda", sep = ""))
      }
    } else {
      for (algo in algos)
      {
        if(algo=="glmnet") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("alpha", "lambda"))*NA
        } else if (algo=="glmnet2") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("lambda"))*NA
        } else if (algo=="gbm") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("n.trees", "interaction.depth", "shrinkage", "n.minobsinnode"))*NA
        } else if (algo=="gbm2") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("n.trees"))*NA
        } else if (algo=="rf") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("mtry"))*NA
        } else if (algo=="rf2") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("mtry"))*NA
        } else if (algo=="svmLinear") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("C"))*NA
        } else if (algo=="svmPoly") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("degree", "sigma", "C"))*NA
        } else if (algo=="svmRadial") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("sigma", "C"))*NA
        } else if (algo=="knn") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("k"))*NA
        } else if (algo=="nb") {
          hyperparameters <- initiate_store(seq(0, N_CV_Folds), c("fL", "usekernel", "adjust"))*NA
        }
        for (i in seq(0, N_CV_Folds))
        {
          error <- tryCatch(hyperparameters[as.character(i),] <- readRDS(paste(path_compute, "hyperparameters", "_", analysis, "_", predictors, "_", algo, "_", i,".Rda", sep = "")), error=function(err) "error")
          suppressWarnings(if(grepl("error", error))
          {
            print(paste("Predictors = ", predictors, ", analysis = ", analysis, ", algo = ", algo, ", fold = ", i, ", the hyperparameter(s) could not be found. The fold will not be used.", sep = ""))
            next
          })
        }
        #print(hyperparameters) #debug
        saveRDS(hyperparameters, paste(path_compute, "hyperparameters", "_", analysis, "_", predictors, "_", algo, ".Rda", sep = ""))
      }
    }
  }
}

#merge performances
print("MERGE")
for (analysis in analyses)
{
  print("ANALYSIS")
  print(analysis)
  if(analysis=="Surv") {
    algos_analysis <- algos_surv
  } else if (analysis=="age_at_collection") {
    algos_analysis <- algos_regression
  } else {
    algos_analysis <- algos
  }
  # sample_sizes <- readRDS(paste(path_compute, "sample_sizes_", analysis, ".Rda", sep = ""))
  # sample_sizes <- rbind(c(sum(sample_sizes$N_train[-1]), sum(sample_sizes$N_test[-1])), sample_sizes)
  # rownames(sample_sizes) <- c("all", seq(0,N_CV_Folds))
  for (predictors in predictorsS)
  {
    print("PREDICTORS")
    print(predictors)
    for (algo in algos_analysis)
    {
      print("ALGORITHM")
      print(algo)
      if(analysis=="age_at_collection" & algo=="nb"){
        print("The nb algorithm is not available for regressions.")
      } else {
        if (!(predictors %in% c("genes", "demo+genes") & !(algo %in% algos_genes)))
        {
          #bind performances for train, test, and display the sd too.
          Performances <- NULL
          for(set in c("test", "train"))
          {
            for(file_name in Performances_types)
            {
              Perf_set_type <- readRDS(paste(path_compute, file_name, "_", set, "_", analysis, "_", predictors, "_", algo, ".Rda", sep = ""))
              if(file_name=="Performance_sd"){names(Perf_set_type) <- paste(names(Perf_set_type), "_sd", sep="")}
              names(Perf_set_type) <- paste(names(Perf_set_type), set, sep="_")
              ifelse(is.null(Performances), Performances <- Perf_set_type, Performances <- cbind(Performances, Perf_set_type))
            }
            assign(paste("sample_sizes", set, sep="_"), readRDS(paste(path_compute, "sample_sizes_", set, "_", analysis, "_", predictors, "_", algo, ".Rda", sep = "")))
          }
          print(Performances)
          hyperparameters <- readRDS(paste(path_compute, "hyperparameters", "_", analysis, "_", predictors, "_", algo, ".Rda", sep = ""))
          hyperparameters <- rbind(apply(hyperparameters, 2, median), hyperparameters)
          rownames(hyperparameters) <- c("all", seq(0,N_CV_Folds))
          sample_sizes <- cbind.data.frame(sample_sizes_train, sample_sizes_test)
          perf_ss_hyper <- cbind(Performances, sample_sizes, hyperparameters)
          saveRDS(perf_ss_hyper, paste(path_store, "perf_ss_hyper", "_", analysis, "_", predictors, "_", algo, ".Rda", sep = ""))
          print(paste("The performances, sample sizes and hyperparameters values on every CV fold for the prediction of ", analysis, " using ", predictors, " as predictors, and using the algorithm ", algo, " are: " , sep=""))
          print(perf_ss_hyper)
        }
      }
    }
  }
}

