#Microbiome display final results

path <- ifelse(getwd() == "/Users/Alan", "~/Desktop/Aging/Microbiome/data/", "/n/groups/patel/Alan/Aging/Microbiome/data/")
source(paste(path, "../scripts/Helpers_microbiome.R", sep = ""))

sets <- c("test")
Performances_types <- c("Performance")

analyses <- c("HvsFD")


for(analysis in analyses)
{
  metrics <- metricsS[[analysis]]
  for(set in sets)
  {
    for(metric in metrics)
    {
      for(file_name in Performances_types)
      {
        print(paste(file_name, " for the prediction of ", analysis, " on the ", set, "ing set, using the metric ", metric, ".", sep=""))
        print(readRDS(paste(path, file_name, "_", set, "_", analysis, "_", metric, ".Rda", sep = "")))
      }
    }
  }
}


