#Step 00: generate data for shiny app, Associations between variables tab

#set path
if(getwd() == "/Users/Alan"){
  path_store <- "~/Desktop/Aging/Microbiome/data/"
  path_compute <- "~/Desktop/Aging/Microbiome/data/"
}else {
  path_store <- "/n/groups/patel/Alan/Aging/Microbiome/data/"
  path_compute <- "/n/scratch2/al311/Aging/Microbiome/data/"}
source(paste(path_store, "../scripts/Helpers_microbiome.R", sep = ""))

for (predictors in c("demographics", "taxa", "cags", "pathways"))
{
  #load data
  meta <- read.xlsx(paste(path_store, "full_metadata_final_cleaned.xlsx", sep = ""))
  if(predictors=="demographics") {
    data <- read.table(paste(path_store, "species_profile_final.csv", sep = ""), header=TRUE, sep=",")
    rownames(data) <- data[,1]
    data <- data[,-1, drop=FALSE]
    data <- data.frame(t(data))
    data$seqID <- rownames(data)
    data <- data[,which(colnames(data) %in% names_variables),drop=F]
  } else if(predictors == "taxa") {
    data <- read.table(paste(path_store, "species_profile_final.csv", sep = ""), header=TRUE, sep=",")
    rownames(data) <- data[,1]
    data <- data[,-1]
    data <- data.frame(t(data))
    data$seqID <- rownames(data)
  } else if(predictors == "cags") {
    names_data <- t(read.table(paste(path_store, "db_colnames.txt", sep = ""), sep = ","))[,1][-1]
    data <- read.table(paste(path_store, "db_cluster_profiles_cag.txt", sep = ""))
    rownames(data) <- data[,1]
    data <- data[,-1]
    names(data) <- names_data
    data <- data.frame(t(data))
    data$seqID <- rownames(data)
  } else if (predictors == "pathways"){
    data <- read.table(paste(path_store, "pathway_annotation_abMat.csv", sep = ""), header=TRUE, sep = ",")
    rownames(data) <- data[,1]
    data <- data[,-1]
    data <- data.frame(t(data))
    data$seqID <- rownames(data)
  } else { #if predictors are genes associated with the algorithm under "predictors
    data <- read.table(paste(path_store, "genes_", analysis(), "_", algo(), ".csv", sep = ""), header=TRUE, sep = ",")
    rownames(data) <- data[,1]
    data <- data[,-1]
    data <- data.frame(t(data))
    data$seqID <- rownames(data)
  }
  #clean the data
  #select relevant columns
  meta <- meta[, c("index", "subjectID", "seqID", "age_at_collection", "delivery_type", "gender", "country", "exclusive_bf", "abx_usage", "is_mother", "seroconverted_ever", "seroconverted_time")]
  names_togetrid <- c("seqID", "index", "is_mother")
  meta <- meta[!is.na(meta$age_at_collection),]
  meta <- meta[!(is.na(meta$gender) | meta$gender == "UNK" ),]
  names(meta)[which(names(meta)=="gender")] <- "sex"
  meta$sex <- factor(meta$sex)
  meta <- meta[!is.na(meta$delivery_type),]
  meta$delivery_type <- factor(meta$delivery_type)
  meta <- meta[!is.na(meta$exclusive_bf),]
  meta$exclusive_bf <- factor(meta$exclusive_bf)
  meta <- meta[!is.na(meta$seroconverted_ever),]
  meta$D_status <- "H"
  meta$D_status[which(meta$seroconverted_ever==1)] <- "FD"
  meta$D_status[which(meta$seroconverted_time==1)] <- "D"
  meta$D_status <- factor(meta$D_status)
  meta$D_status <- factor(meta$D_status,levels(meta$D_status)[c(3,2,1)])
  meta$seroconverted_ever <- factor(meta$seroconverted_ever)
  meta$seroconverted_time <- factor(meta$seroconverted_time)
  meta <- meta[!is.na(meta$abx_usage),]
  meta$abx_usage <- factor(meta$abx_usage)
  meta <- meta[!(is.na(meta$is_mother) | meta$is_mother == 1),]
  data <- merge(meta, data, by = "seqID")
  data <- data[,which(!(names(data) %in% c("seqID", "index", "subjectID", "is_mother")))]
  rownames(data) <- data$index
  data <- preprocess_data_app(data)
  saveRDS(data, paste(path_store, "data_app_associations_", predictors, ".Rda", sep=""))
}

for (analysis in analyses)
{
  print(analysis)
  ifelse(analysis=="Surv", algos_list <- algos_surv, algos_list <- algos_genes)
  for (algo in algos_list)
  {
    print(algo)
    meta <- read.xlsx(paste(path_store, "full_metadata_final_cleaned.xlsx", sep = ""))
    data <- read.table(paste(path_store, "genes_", analysis, "_", algo, ".csv", sep = ""), header=TRUE, sep = ",")
    rownames(data) <- data[,1]
    data <- data[,-1]
    data <- data.frame(t(data))
    data$seqID <- rownames(data)
    #clean the data
    #select relevant columns
    meta <- meta[, c("index", "subjectID", "seqID", "age_at_collection", "delivery_type", "gender", "country", "exclusive_bf", "abx_usage", "is_mother", "seroconverted_ever", "seroconverted_time")]
    names_togetrid <- c("seqID", "index", "is_mother")
    meta <- meta[!is.na(meta$age_at_collection),]
    meta <- meta[!(is.na(meta$gender) | meta$gender == "UNK" ),]
    names(meta)[which(names(meta)=="gender")] <- "sex"
    meta$sex <- factor(meta$sex)
    meta <- meta[!is.na(meta$delivery_type),]
    meta$delivery_type <- factor(meta$delivery_type)
    meta <- meta[!is.na(meta$exclusive_bf),]
    meta$exclusive_bf <- factor(meta$exclusive_bf)
    meta <- meta[!is.na(meta$seroconverted_ever),]
    meta$D_status <- "H"
    meta$D_status[which(meta$seroconverted_ever==1)] <- "FD"
    meta$D_status[which(meta$seroconverted_time==1)] <- "D"
    meta$D_status <- factor(meta$D_status)
    meta$D_status <- factor(meta$D_status,levels(meta$D_status)[c(3,2,1)])
    meta$seroconverted_ever <- factor(meta$seroconverted_ever)
    meta$seroconverted_time <- factor(meta$seroconverted_time)
    meta <- meta[!is.na(meta$abx_usage),]
    meta$abx_usage <- factor(meta$abx_usage)
    meta <- meta[!(is.na(meta$is_mother) | meta$is_mother == 1),]
    data <- merge(meta, data, by = "seqID")
    data <- data[,which(!(names(data) %in% c("seqID", "index", "subjectID", "is_mother")))]
    rownames(data) <- data$index
    data <- preprocess_data_app(data)
    saveRDS(data, paste(path_store, "data_app_associations_genes_", analysis, "_", algo, ".Rda", sep=""))
  }
}

#generate names of different kinds of predictors
#taxa
data <- read.table(paste(path_store, "species_profile_final.csv", sep = ""), header=TRUE, sep=",")
rownames(data) <- data[,1]
data <- data[,-1]
data <- data.frame(t(data))
saveRDS(names(data), paste(path_store, "names_taxa.Rda", sep=""))
#cags
names_data <- t(read.table(paste(path_store, "db_colnames.txt", sep = ""), sep = ","))[,1][-1]
data <- read.table(paste(path_store, "db_cluster_profiles_cag.txt", sep = ""))
rownames(data) <- data[,1]
data <- data[,-1]
names(data) <- names_data
data <- data.frame(t(data))
saveRDS(names(data), paste(path_store, "names_cags.Rda", sep=""))
#pathways
data <- read.table(paste(path_store, "pathway_annotation_abMat.csv", sep = ""), header=TRUE, sep = ",")
rownames(data) <- data[,1]
data <- data[,-1]
data <- data.frame(t(data))
saveRDS(names(data), paste(path_store, "names_pathways.Rda", sep=""))



#quick fix name
for (predictors in c("demographics", "taxa", "cags", "pathways"))
{
  data <- readRDS(paste(path_store, "data_app_associations_", predictors, ".Rda", sep=""))
  names(data)[which(names(data) == "Seronconverted Ever")] <- "Seroconverted Ever"
  saveRDS(data, paste(path_store, "data_app_associations_", predictors, ".Rda", sep=""))
}

for (analysis in analyses)
{
  print(analysis)
  ifelse(analysis=="Surv", algos_list <- algos_surv, algos_list <- algos_genes)
  for (algo in algos_list)
  {
    data <- readRDS(paste(path_store, "data_app_associations_genes_", analysis, "_", algo, ".Rda", sep=""))
    names(data)[which(names(data) == "Seronconverted Ever")] <- "Seroconverted Ever"
    saveRDS(data, paste(path_store, "data_app_associations_genes_", analysis, "_", algo, ".Rda", sep=""))
  }
}
  
