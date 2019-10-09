#Step 01: Preprocesses the data

#extract and use info from command line
args = commandArgs(trailingOnly=TRUE)
#default, if no command line received
if(length(args) == 0){args <- c("abx_usage", "taxa")}
if(length(args) != 2)
{
  stop("use two arguments for analysis")
} else {
  analysis <- args[1]
  predictors <- args[2]
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

#preprocess
print(target)
preprocessing(analysis, target, predictors)
print("done")
