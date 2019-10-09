#Step 02: Preprocesses each CV fold of the data

#extract and use info from command line
args = commandArgs(trailingOnly=TRUE)
if(length(args) == 0){args <- c("Surv", "taxa", "9")} #default, if no command line received
if(length(args) != 3)
{
  stop("use three arguments for analysis")
} else {
  analysis <- args[1]
  predictors <- args[2]
  i <- args[3]
}

#set path
if(getwd() == "/Users/Alan"){
  path_store <- "~/Desktop/Aging/Microbiome/data/"
  path_compute <- "~/Desktop/Aging/Microbiome/data/"
}else {
  path_store <- "/n/groups/patel/Alan/Aging/Microbiome/data/"
  path_compute <- "/n/scratch2/al311/Aging/Microbiome/data/"}
source(paste(path_store, "../scripts/Helpers_microbiome.R", sep = ""))

#set target variable
target <- target_of_analysis(analysis)

#preprocess folds
print(target)
print(i)
if (i == "0")
{
  preprocessing_folds_0(analysis, target, predictors)
} else {
  preprocessing_folds(analysis, target, predictors, i)
}
print("done")
