#analysis <- "age_at_collection"
#predictors <- "genes"
#algo <- "rf2"
#i <- "0"

#set path
if(getwd() == "/Users/Alan"){
  path_store <- "~/Desktop/Aging/Microbiome/data/"
  path_compute <- "~/Desktop/Aging/Microbiome/data/"
}else {
  path_store <- "/n/groups/patel/Alan/Aging/Microbiome/data/"
  path_compute <- "/n/scratch2/al311/Aging/Microbiome/data/"}
source(paste(path_store, "../scripts/Helpers_microbiome.R", sep = ""))


for (analysis in c("country", "HvsFDvsD"))
{
  for (predictors in predictorsS)
  {
    for (algo in c("rf2"))
    {
      for (i in seq(0, N_CV_Folds))
      {
        model <- readRDS(paste(path_compute, "model", "_", analysis, "_", predictors, "_", algo, "_", i,".Rda", sep = ""))
        best_predictors(model, analysis, predictors, algo, i)
      }
    }
  }
}

for (analysis in c("Surv"))
{
  for (predictors in c("mixed", "demo+mixed"))
  {
    for (algo in c("glmnet2"))
    {
      for (i in seq(0, N_CV_Folds))
      {
        model <- readRDS(paste(path_compute, "model", "_", analysis, "_", predictors, "_", algo, "_", i,".Rda", sep = ""))
        best_predictors_surv(model, analysis, predictors, algo, i)
      }
    }
  }
}


