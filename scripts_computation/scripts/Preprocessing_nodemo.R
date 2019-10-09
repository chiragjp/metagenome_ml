#Step 04: Preprocessing: generating the preprocessed dataset for which the predictors do not include the demographics variables

#extract and use info from command line
args = commandArgs(trailingOnly=TRUE)
if(length(args) == 0){args <- c("Surv", "0")} #default, if no command line received
if(length(args) != 2)
{
  stop("use two arguments for analysis")
} else {
  analysis <- args[1]
  i <- args[2]
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
preprocessing_nodemo(analysis, i)
print("done")

