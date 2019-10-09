#Display hyperparameters
path <- ifelse(getwd() == "/Users/Alan", "~/Desktop/Aging/Microbiome/data/", "/n/groups/patel/Alan/Aging/Microbiome/data/")
source(paste(path, "../scripts/Helpers_microbiome.R", sep = ""))


#regression and classification
print("REGRESSION AND CLASSIFICATIONS")
for (algo in algos)
{
  print("ALGORITHM")
  print(algo)
  for (predictors in predictorsS)
  {
    print("PREDICTORS")
    print(predictors)
    for (analysis in analyses[which(!(analyses=="Surv"))])
    {
      print(analysis)
      if (!(predictors == "genes" & !(algo %in% algos_genes)))
      {
        print(paste("The performances, sample sizes and hyperparameters values on every CV fold for the prediction of ", analysis, " using ", predictors, " as predictors, and using the algorithm ", algo, " are: " , sep=""))
        print(readRDS(paste(path, "perf_ss_hyper", "_", analysis, "_", predictors, "_", algo, ".Rda", sep = "")))
      }
    }
  }
}

#survival
print("SURVIVAL")
analysis <- "Surv"
for (algo in algos_surv)
{
  print("ALGORITHM")
  print(algo)
  for (predictors in predictorsS)
  {
    print("PREDICTORS")
    print(predictors)
    print(readRDS(paste(path, "hyperparameters", "_", analysis, "_", predictors, "_", algo, ".Rda", sep = "")))
  }
}


print("done")

