#Step 09: Merges the prediction performances for each target, predictors and algorithm

#extract and use info from command line
args = commandArgs(trailingOnly=TRUE)
#default, if no command line received
if(length(args) == 0){args <- c("Surv", "CI", "test", "Performance")}
print(args)
print(length(args))

if(length(args) != 4)
{
  stop("use 4 arguments for analysis")
} else {
  analysis <- args[1]
  metric <- args[2]
  set <- args[3]
  type <- args[4]
}

#set path
if(getwd() == "/Users/Alan"){
  path_store <- "~/Desktop/Aging/Microbiome/data/"
  path_compute <- "~/Desktop/Aging/Microbiome/data/"
}else {
  path_store <- "/n/groups/patel/Alan/Aging/Microbiome/data/"
  path_compute <- "/n/scratch2/al311/Aging/Microbiome/data/"}
source(paste(path_store, "../scripts/Helpers_microbiome.R", sep = ""))

print("starting")
Performance <- merge_results(type, analysis, metric, set)
print(Performance)
print("done")


