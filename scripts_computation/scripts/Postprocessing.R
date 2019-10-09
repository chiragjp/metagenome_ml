#Step 07: Calculates the R2/accuracy/ROC/cross-entropy/Concordance Index for each fold and on the entire dataset

#extract and use info from command line
args = commandArgs(trailingOnly=TRUE)
#default, if no command line received
if(length(args) == 0){args <- c("Surv", "demographics", "test")}
if(length(args) != 3)
{
  stop("use three arguments for analysis")
} else {
  analysis <- args[1]
  predictors <- args[2]
  set <- args[3]
}

#set path
if(getwd() == "/Users/Alan"){
  path_store <- "~/Desktop/Aging/Microbiome/data/"
  path_compute <- "~/Desktop/Aging/Microbiome/data/"
}else {
  path_store <- "/n/groups/patel/Alan/Aging/Microbiome/data/"
  path_compute <- "/n/scratch2/al311/Aging/Microbiome/data/"}
source(paste(path_store, "../scripts/Helpers_microbiome.R", sep = ""))

print("start")
if (analysis == "Surv"){
  post_processing_surv(analysis, predictors, set)
} else if (analysis == "age_at_collection"){
  post_processing_regression(analysis, predictors, set)
} else if (analysis %in% c("country", "HvsFDvsD")){
  post_processing_multinomial(analysis, predictors, set)
} else {
  post_processing_binomial(analysis, predictors, set)
}
print("done")


