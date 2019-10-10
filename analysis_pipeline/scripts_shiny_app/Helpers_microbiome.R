#Helpers microbiome

seed <- 0
set.seed(seed)
#libraries
library(openxlsx)
library(dummies)
library(glmnet)
#only load if on computer or cluster (HMS'O2), not on Shiny app
if(!(uploaded_app))
{
  library(caTools)
  library(gbm)
  library(randomForest)
  library(e1071) #svm
  library(randomForestSRC)
  library(caret)
  library(doParallel)
  library(pROC)
  library(MLmetrics)
  library(dplyr)
  library(boot)
  library(nnet)
}


#parameters
genes_analysis = FALSE #run on cags or genes
N_CV_Folds <- 10
#define cv for cross validation in caret. Used seeds = 0 for all subseeds.
cv_seeds <- vector(mode = "list", length = N_CV_Folds)
cv_seeds <- lapply(cv_seeds, function(x) integer(N_CV_Folds+30000)) #use 30,000 as the max number of potential combinaisons of hyperparameters to test, to be safe.
cv_seeds[[length(cv_seeds) + 1]] <- integer(1)
analyses <- c("age_at_collection", "abx_usage", "exclusive_bf", "delivery_type", "sex", "country")
predictorsS <- c("demographics", "taxa", "demo+taxa", "cags", "demo+cags", "genes", "demo+genes", "pathways", "demo+pathways", "mixed", "demo+mixed")
predictorsS_withdemo <- c("demo+taxa", "demo+cags", "demo+genes", "demo+pathways", "demo+mixed")
algos <- c("glmnet", "glmnet2", "gbm", "gbm2", "rf", "rf2", "svmLinear", "svmPoly", "svmRadial", "knn", "nb")
algos_genes <- c("glmnet", "glmnet2", "gbm", "gbm2", "rf", "rf2")
algos_regression <- c("glmnet", "glmnet2", "gbm", "gbm2", "rf", "rf2", "svmLinear", "svmPoly", "svmRadial", "knn")
sets <- c("train", "test")
Performances_types <- c("Performance", "Performance_sd")
metricsS <- list("age_at_collection"=c("R2"), "country"= c("Cross_Entropy", "Mean_Accuracy", "Accuracy", "FIN", "EST", "RUS", "SWE" ))
for (ana in c("abx_usage", "exclusive_bf", "delivery_type", "sex")){metricsS[[ana]] <- c("ROC", "Cross_Entropy", "Mean_Accuracy", "Accuracy", "Sensitivity", "Specificity")}
families <- c("age_at_collection" = "gaussian", "abx_usage"="binomial", "exclusive_bf"="binomial", "delivery_type"="binomial", "sex"="binomial", "country"="multinomial")
names_variables <- c("index", "subjectID", "seqID", "age_at_collection", "delivery_type", "sex", "country", "country.EST", "country.RUS", "country.SWE", "exclusive_bf", "abx_usage", "is_mother", "target")
R_boot <- 1000
#Display variables
#define
analyses_labels <- c("Age", "Antibiotics Usage", "Exclusively Breastfed", "Delivery Type", "Sex", "Country of Origin")
predictors_labels <- c("Demographics", "Taxa", "Taxa + Demographics", "CAGs", "CAGs + Demographics", "Genes", "Genes + Demographics", "Pathways", "Pathways + Demographics", "Mixed Predictors", "Mixed Predictors + Demographics")
algos_labels <- c("Elastic Net (Caret)", "Elastic Net 2", "Gradient Boosted Machine (Caret)", "Gradient Boosted Machine 2", "Random Forest (Caret)", "Random Forest 2", "Support Vector Machine: Linear kernel", "Support Vector Machine: Polynomial kernel of degree 2", "Support Vector Machine: Radial kernel", "K-Nearest Neighbors", "Naive Bayes")
algos_genes_labels <- c("Elastic Net (Caret)", "Elastic Net 2", "Gradient Boosted Machine (Caret)", "Gradient Boosted Machine 2", "Random Forest (Caret)", "Random Forest 2")
metricsS_labels <- list("age_at_collection"=c("R-Squared"), "country"= c("Mean Class Accuracies", "Accuracy", "Finland", "Estonia", "Russia", "Sweden", "Cross Entropy"))
for (ana in c("abx_usage", "exclusive_bf", "delivery_type", "sex")){metricsS_labels[[ana]] <- c("AUC of the ROC", "Mean Class Accuracies", "Accuracy", "Sensitivity", "Specificity", "Cross Entropy")}
sets_labels <- c("Training Set", "Testing Set")
Performances_types_labels <- c("Performance", "Performance +- Confidence Interval")
folds_labels <- c("All samples", seq(N_CV_Folds))
dictionary_analysis <- data.frame("name"=c("age_at_collection", "abx_usage", "exclusive_bf", "delivery_type", "sex", "country"), "label" = c("Age", "Antibiotics Usage", "Exclusively Breastfed", "Delivery Type", "Sex", "Country of Origin"), stringsAsFactors=FALSE)      
dictionary_predictors <- data.frame("name"=c("demographics", "taxa", "demo+taxa", "cags", "demo+cags", "genes", "demo+genes", "pathways", "demo+pathways", "mixed", "demo+mixed"), "label"=c("Demographics", "Taxa", "Taxa + Demographics", "CAGs", "CAGs + Demographics", "Genes", "Genes + Demographics", "Pathways", "Pathways + Demographics", "Mixed Predictors", "Mixed Predictors + Demographics"), stringsAsFactors=FALSE)
dictionary_algo <- data.frame("name"=c("glmnet", "glmnet2", "gbm", "gbm2", "rf", "rf2", "svmLinear", "svmPoly", "svmRadial", "knn", "nb"), "label"=c("Elastic Net (Caret)", "Elastic Net 2", "Gradient Boosted Machine (Caret)", "Gradient Boosted Machine 2", "Random Forest (Caret)", "Random Forest 2", "Support Vector Machine: Linear kernel", "Support Vector Machine: Polynomial kernel of degree 2", "Support Vector Machine: Radial kernel", "K-Nearest Neighbors", "Naive Bayes"), stringsAsFactors=FALSE)
dictionary_set <- data.frame("name"=c("train", "test"), "label"=c("Training Set", "Testing Set"), stringsAsFactors=FALSE)
dictionary_file_name <- data.frame("name"=c("Performance", "Performance_sd"), "label"=c("Performance", "Performance +- Confidence Interval"), stringsAsFactors=FALSE)
dictionary_metric <- data.frame("name"=c("R2", "CI", "ROC", "Cross_Entropy", "Mean_Accuracy", "Accuracy", "Sensitivity", "Specificity", "FIN", "EST", "RUS", "SWE", "X0", "X1"), "label"=c("R-Squared", "Concordance Index", "AUC of the ROC", "Cross Entropy", "Mean Class Accuracies", "Accuracy", "Sensitivity", "Specificity", "Finland", "Estonia", "Russia", "Sweden", "Class 0", "Class 1"), stringsAsFactors=FALSE)
dictionary_fold <- data.frame("name"=seq(0,N_CV_Folds), "label"=c("All samples", seq(N_CV_Folds)), stringsAsFactors=FALSE)
dictionary_multinomial <- data.frame("name"=c("FIN", "EST", "RUS", "SWE"), "label"=c("Finland", "Estonia", "Russia", "Sweden"), stringsAsFactors=FALSE)
n_digits_display <- list("R2"=3, "CI"=3, "ROC"=3, "Cross_Entropy"=5, "Mean_Accuracy"=3,"Accuracy"=3, "Sensitivity"=3, "Specificity"=3, "FIN"=3, "EST"=3, "RUS"=3, "SWE"=3)
names_hyperparameters <- c("lambda", "alpha", "n.trees", "interaction.depth", "shrinkage", "n.minobsinnode", "mtry", "C", "degree", "sigma", "k", "fL", "usekernel", "adjust")
display_performances <- list("R2"=T, "CI"=T, "ROC"=T, "Cross_Entropy"=F, "Mean_Accuracy"=T, "Accuracy"=F, "Sensitivity"=F, "Specificity"=F, "FIN"=F, "EST"=F, "RUS"=F, "SWE"=F)
predictors_types <- c("Demographics", "Taxa", "Genes", "CAGs", "Pathways")

#take subset of variables for the app: do not display diabetes results
analyses_labels <- c("Age", "Antibiotics Usage", "Exclusively Breastfed", "Delivery Type", "Sex", "Country of Origin")
variables_to_remove_associations <- c("Seroconverted Sample", "Seroconverted Ever", "Diabetic Status")

#redefine plot to replace it if necessary so that having a inf value does not crash it.
plot.window.orig <- plot.window
plot.window2 <- function(xlim, ylim, log="", asp=NA, ...)
{
  if (!all(is.finite(xlim))) xlim <- c(0,1)
  if (!all(is.finite(ylim))) ylim <- c(0,1)
  plot.window.orig(xlim, ylim, log="", asp=NA, ...)
}

# #rename input files
# for (analysis in analyses)
# {
#   for (algo in algos_genes)
#   {
#     rename_input_files(analysis, algo)
#   }
# }

#functions
remove_leading_zero <- function(val, n_digits)
{
  return(sub("^(-?)0.", "\\1.", sprintf(paste("%.", n_digits, "f", sep=""), val)))
}

target_of_analysis <- function(analysis)
{
  #set target variable
  target <- analysis
  return(target)
}

predictor_type <- function(name)
{
  if(name %in% names_variables) {
    return("Demographics")
  } else if (name %in% names_taxa) {
    return("Taxa")
  } else if (name %in% names_cags) {
    return("CAGs")
  } else if (name %in% names_pathways) {
    return("Pathways")
  } else {
    return("Genes")
  }
}

define_metric <- function(analysis)
{
  if(analysis=="age_at_collection") {
    metric <- "Rsquared"
  } else if (analysis %in% c("country")) {
    metric <- "Accuracy"
  } else {
    metric <- "ROC" }
}

calculate_r2 <- function(y, prediction)
{
  SSR = sum((y - prediction) ^ 2)
  SST = sum((y - mean(y)) ^ 2)
  r2 <- (1 - SSR/SST)
}

boot_R2 <- function(data, indices)
{
  data = data[indices,]
  y <- data[["y"]]
  prediction <- data[["pred"]]
  SSR = sum((y - prediction) ^ 2)
  SST = sum((y - mean(y)) ^ 2)
  r2 <- (1 - SSR/SST)
  return(r2)
}

boot_mean <- function(data, indices)
{
  data = data[indices]
  return(mean(data))
}

#initiate storing matrices
initiate_store <- function(rows, columns)
{
  data <- data.frame(matrix(0, length(rows), length(columns)))
  rownames(data) <- rows
  names(data) <- columns
  return(data)
}

#generate weights
generate_weights <- function(y)
{
  lgth_w <- length(y)
  w <- seq(1,lgth_w)*0 + 1
  if(!(analysis %in% c("age_at_collection")))
  {
    table_y <- table(y)
    #create weights for training, to correct for classes unbalance
    for (y_class in names(table_y))
    {
      w[which(y == y_class)] <- 1/table(y)[y_class]
    }
  }
  return(w)
}

binomial_convert_to_class <- function(pred)
{
  pred_prob <- pred
  pred_prob[which(pred>.5)] <- "1"
  pred_prob[which(pred <= .5)] <- "0"
  return(pred_prob)
}

rename_input_files <- function(analysis, algo="cags")
{
  name_algo <- ifelse(grepl("glmnet", algo), "glm", algo)
  name_algo <- ifelse(algo == "gbm2", "gbm", name_algo)
  i_algo <- ifelse(grepl(2,algo), 2, 1)
  data <- read.table(paste("~/Desktop/data/", "gene_names_v", i_algo, "_", name_algo, "_p-", analysis, "-", algo, "-0.out_cags_gene_abundance.csv", sep = ""), header=TRUE, sep = ",")
  write.table(data, paste("~/Desktop/data/", "genes_", analysis, "_", algo, ".csv", sep =""), sep = ",")
}

cross_entropy_weighted <- function (y, pred, w) 
{
  if (is.matrix(y) == FALSE){y <- model.matrix(~0 + ., data.frame(as.character(y)))}
  eps <- 1e-10
  N <- nrow(pred)
  pred[pred==0] <- eps
  pred[pred==1] <- 1-eps
  cewl <- (-1/N)*sum(y*log(pred)*w)
  return(cewl)
}

preprocessing <- function(analysis, target, predictors)
{
  #load data
  meta <- read.xlsx(paste(path_store, "full_metadata_final_cleaned.xlsx", sep = ""))
  if(predictors == "taxa")
  {
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
    data <- read.table(paste(path_store, "genes_", analysis, "_", predictors, ".csv", sep = ""), header=TRUE, sep = ",")
    rownames(data) <- data[,1]
    data <- data[,-1]
    data <- data.frame(t(data))
    data$seqID <- rownames(data)
  }
  #clean the data
  #select relevant columns
  meta <- meta[, c("index", "subjectID", "seqID", "age_at_collection", "delivery_type", "gender", "country", "exclusive_bf", "abx_usage", "is_mother", "seroconverted_ever", "seroconverted_time")]
  names_togetrid <- c("seqID", "index", "is_mother")
  #exclude missing ages
  meta <- meta[!is.na(meta$age_at_collection),]
  #exclude missing genders <= if gender ends up being a useless predictor, do not get rid of those samples!
  meta <- meta[!(is.na(meta$gender) | meta$gender == "UNK" ),]
  #exclude cesarian <= if delivery_type ends up being a useless predictor, do not get rid of those samples! Or try to keep cesarian to see if can learn from them, but might just add noise
  meta <- meta[!is.na(meta$delivery_type),]
  delivery_type <- seq(nrow(meta))*0
  delivery_type[which(meta$delivery_type =="cesarean")] <- 1
  meta$delivery_type <- delivery_type
  #exclude missing breastfed <= if exclusive_bf ends up being a useless predictor, do not get rid of those samples!
  meta <- meta[!is.na(meta$exclusive_bf),]
  #exclude diabetics <= if seroconverted_ever ends up being a useless predictor, do not get rid of those samples!
  meta <- meta[!(is.na(meta$seroconverted_ever) | meta$seroconverted_ever == 1),]
  names_togetrid <- c(names_togetrid, "seroconverted_ever", "seroconverted_time")
  #exclude missing antibiotics
  meta <- meta[!is.na(meta$abx_usage),]
  #exclude mothers
  meta <- meta[!(is.na(meta$is_mother) | meta$is_mother == 1),]
  #merge the two datasets
  data <- merge(meta, data, by = "seqID")
  rownames(data) <- data$index
  #explore the data
  #number of samples
  dim(data)
  #number of individuals
  length(table(data$subjectID))
  #distribution of age
  #hist(data$age_at_collection)
  #how many of each gender
  table(data$gender, useNA = "always")
  #how many of each countries?
  table(data$country, useNA = "always")
  #how many cesarian?
  table(data$delivery_type, useNA = "always")
  #how many breastfed?
  table(data$exclusive_bf, useNA = "always")
  #how many took antibiotics
  table(data$abx_usage, useNA = "always")
  #how many are mothers
  table(data$is_mother, useNA = "always")
  data <- data[,-which(names(data) %in% names_togetrid)]
  if(analysis == "country")
  {
    data$country <- as.factor(data$country)
    data$sex <- 0
    data$sex[data$gender == "female"] <- 1
    data_split <- data
    data <- data[,-which(names(data) == "gender")]
  } else {
    dummies <- data.frame(dummy(data$country))
    for (c in c("FIN", "EST", "RUS", "SWE"))
    {
      names(dummies)[grepl(c, names(dummies))] <- paste("country", c, sep = ".")
    }
    dummies <- dummies[,-which(names(dummies) == "country.FIN")]
    dummies$sex <- 0
    dummies$sex[data$gender == "female"] <- 1
    data <- cbind(dummies, data)
    data_split <- data
    data <- data[,-which(names(data) %in% c("gender", "country"))]
  }
  if (!(analysis %in% c("age_at_collection"))){data[,target] <- as.factor(data[,target])}
  if(target=="country"){data[,target] <- factor(data[,target], levels = c("FIN", "EST", "RUS", "SWE"))}
  #generate_folds depending on the analysis
  if (analysis == "age_at_collection")
  {
    #split into training and testing keeping all the sample of each user on the same side
    data_split <- data_split[!duplicated(data_split$subjectID),]
    folds <- createFolds(data_split$country, k = N_CV_Folds)
  } else if (analysis == "country"){
    #split into training and testing keeping all the sample of each user on the same side, stratifying by country
    data_split <- data_split[!duplicated(data_split$subjectID),]
    y <- data_split[, target]
    y_fin <- seq(length(y))[which(y=="FIN")]
    y_est <- seq(length(y))[which(y=="EST")]
    y_rus <- seq(length(y))[which(y=="RUS")]
    y_swe <- seq(length(y))[which(y=="SWE")]
    folds_fin <- createFolds(y_fin, k = N_CV_Folds)
    folds_est <- createFolds(y_est, k = N_CV_Folds)
    folds_rus <- createFolds(y_rus, k = N_CV_Folds)
    folds_swe <- createFolds(y_swe, k = N_CV_Folds)
    folds <- folds_est
    for (fold in names(folds)){folds[[fold]] <- c(y_fin[folds_fin[[fold]]], y_est[folds_est[[fold]]], y_rus[folds_rus[[fold]]], y_swe[folds_swe[[fold]]])}
  } else {
    #split into training and testing keeping all the sample of each user on the same side, stratifying by the target
    data_split <- data_split[!duplicated(data_split$subjectID),]
    y <- data_split[, target]
    y0 <- seq(length(y))[which(y==0)]
    y1 <- seq(length(y))[which(y==1)]
    folds0 <- createFolds(y0, k = N_CV_Folds)
    folds1 <- createFolds(y1, k = N_CV_Folds)
    folds <- folds0
    for (fold in names(folds)){folds[[fold]] <- c(y0[folds0[[fold]]], y1[folds1[[fold]]])}
  }
  #save files
  saveRDS(folds, paste(path_compute, "folds_", analysis, "_", predictors, ".Rda", sep = ""))
  saveRDS(data_split, paste(path_compute, "data_split_", analysis, "_", predictors, ".Rda", sep = ""))
  saveRDS(data, paste(path_compute, "data_", analysis, "_", predictors, ".Rda", sep = ""))
  #generate and save the indices if predictors = cags. other predictors will use the same indices.
  if(predictors=="cags")
  {
    for (set in sets)
    {
      #save fold 0
      for (i in seq(N_CV_Folds))
      {
        index <- folds[[i]]
        ifelse(set=="train", index <- data_split$subjectID[-index], index <- data_split$subjectID[index])
        indices <- rownames(data)[which(data$subjectID %in% index)]
        saveRDS(indices, paste(path_compute, "indices_", set, "_", analysis, "_", predictors, "_", i, ".Rda", sep = ""))
      }
    }
  }
}


preprocessing_folds_0 <- function(analysis, target, predictors)
{
  #generate fold 0: keep all the samples in the training set, to judge the variables' importance.
  #load files
  data_cags <- readRDS(paste(path_compute, "data_", analysis, "_cags.Rda", sep = "")) #load it to make sure the same samples will be used for all pipelines. (predictors wise)
  data_train <- readRDS(paste(path_compute, "data_", analysis, "_", predictors, ".Rda", sep = ""))
  data_train <- data_train[which(rownames(data_train) %in% rownames(data_cags)),] #use the same samples no matter what the predictors are
  #get rid of predictors with only one value
  data_train <- data_train[,sapply(data_train, function(x)(length(unique(x))>1))]
  #display
  print(dim(data_train))
  names_predictors <- names(data_train)[-which(names(data_train) %in% names_variables)]
  names_others <- names(data_train)[-which(names(data_train) %in% names_predictors)]
  names_others <- names_others[-which(names_others == "subjectID")]
  data_train <- data_train[, c(names_others, names_predictors)]
  #normalize
  y <- data_train[,target]
  y_train = data_train[, target]
  x_train = as.matrix(data_train[,-which(names(data_train) %in% names_others)])
  means_x <- sapply(data.frame(x_train), mean)
  sds_x <- sapply(data.frame(x_train), sd)
  x_train <- t((t(x_train)-means_x)/sds_x)
  data_train[,-which(names(data_train) %in% names_others)] <- x_train
  #generate weights
  w_train <- generate_weights(y_train)
  #filter the predictors
  COEFS <- initiate_store(names_predictors, c("coefficients", "p-values"))
  if (analysis == "age_at_collection")
  {
    for (name_predictor in names_predictors)
    {
      x <- data_train[, name_predictor]
      model <- lm(y~x)
      COEFS[name_predictor,] <- summary(model)$coefficients["x",c(1,4)]
    }
    #index of predictors to select
    if(length(which(COEFS$`p-values` < 0.05)) < 1000)
    {
      index <- rownames(COEFS)[order(COEFS$`p-values`, decreasing = FALSE)[1:min(length(names_predictors),1000)]]
    } else {
      index <- rownames(COEFS)[order(abs(COEFS$coefficients), decreasing = TRUE)[1:1000]]
    }
  } else if(analysis %in% c("country"))
  {
    for (name_predictor in names_predictors)
    {
      x <- data_train[, name_predictor]
      model <- multinom(y ~ x, trace = FALSE)
      COEFS[name_predictor, "coefficients"] <- max(abs(summary(model)$coefficients[,"x"]))
      z <- summary(model)$coefficients[,"x"]/summary(model)$standard.errors[,"x"]
      COEFS[name_predictor, "p-values"] <- min(1 - pnorm(abs(z), 0, 1))*2 # 2-tailed Wald z tests to test significance of coefficients
    }
    #index of predictors to select
    if(length(which(COEFS$`p-values` < 0.05)) < 1000)
    {
      index <- rownames(COEFS)[order(COEFS$`p-values`, decreasing = FALSE)[1:min(length(names_predictors),1000)]]
    } else {
      index <- rownames(COEFS)[order(abs(COEFS$coefficients), decreasing = TRUE)[1:1000]]
    }
  } else {
    for (name_predictor in names_predictors)
    {
      x <- data_train[, name_predictor]
      model <- glm(y~x, family = "binomial")
      COEFS[name_predictor,] <- summary(model)$coefficients["x",c(1,4)]
    }
    #COEFS$`p-values` <- COEFS$`p-values`*nrow(COEFS) #do not correct for multiple testing otherwise only 3 cag remain for gender. be less conservative.
    #index of predictors to select
    if(length(which(COEFS$`p-values` < 0.05)) < 1000)
    {
      index <- rownames(COEFS)[order(COEFS$`p-values`, decreasing = FALSE)[1:min(length(names_predictors),1000)]]
    } else {
      index <- rownames(COEFS)[order(abs(COEFS$coefficients), decreasing = TRUE)[1:1000]]
    }
  }
  print(COEFS[index[1:100],])
  print(table(COEFS$`p-values` < 0.05/nrow(COEFS)))
  print(table(COEFS$coefficients < 0))
  data_train <- data_train[, c(names_others, index)]
  var_to_remove <- target
  x_train = as.matrix(data_train[,-which(names(data_train) %in% var_to_remove)])
  y_train <- data_train[, target]
  preprocessed_data <- list("data_train" = data_train, "x_train" = x_train, "y_train" = y_train, "w_train" = w_train, "data_test" = data_train, "x_test" = x_train, "y_test" = y_train, "w_test" = w_train)
  saveRDS(preprocessed_data, paste(path_compute, "preprocessed_data_", analysis, "_", paste("demo", predictors, sep="+"), "_0.Rda", sep = ""))
}

preprocessing_folds <- function(analysis, target, predictors, i)
{
  #load files
  folds <- readRDS(paste(path_compute, "folds_", analysis, "_cags.Rda", sep = ""))
  data_split <- readRDS(paste(path_compute, "data_split_", analysis, "_cags.Rda", sep = "")) #use the folds from the cags on all analyses
  data_cags <- readRDS(paste(path_compute, "data_", analysis, "_cags.Rda", sep = "")) #load it to make sure the same samples will be used for all pipelines. (predictors wise)
  data <- readRDS(paste(path_compute, "data_", analysis, "_", predictors, ".Rda", sep = ""))
  data <- data[which(rownames(data) %in% rownames(data_cags)),] #use the same samples no matter what the predictors are
  print(dim(data))
  index <- folds[[as.numeric(i)]]
  index_train  <- data_split$subjectID[-index]
  index_test <- data_split$subjectID[index]
  data_train <- data[which(data$subjectID %in% index_train),]
  data_test <- data[which(data$subjectID %in% index_test),]
  #get rid of predictors with only one value
  data_test <- data_test[,sapply(data_train, function(x)(length(unique(x))>1))]
  data_train <- data_train[,sapply(data_train, function(x)(length(unique(x))>1))]
  #display
  dim(data_train)
  dim(data_test)
  names_predictors <- names(data_train)[-which(names(data_train) %in% names_variables)]
  names_others <- names(data_train)[-which(names(data_train) %in% names_predictors)]
  names_others <- names_others[-which(names_others == "subjectID")]
  data_train <- data_train[, c(names_others, names_predictors)]
  data_test <- data_test[, c(names_others, names_predictors)]
  #normalize
  y_train = data_train[, target]
  y_test = data_test[, target]
  #create weights
  w_train <- generate_weights(y_train)
  x_train = as.matrix(data_train[,-which(names(data_train) %in% names_others)])
  x_test = as.matrix(data_test[,-which(names(data_test) %in% names_others)])
  means_x <- sapply(data.frame(x_train), mean)
  sds_x <- sapply(data.frame(x_train), sd)
  x_train <- t((t(x_train)-means_x)/sds_x)
  x_test <-  t((t(x_test)-means_x)/sds_x)
  data_train[,-which(names(data_train) %in% names_others)] <- x_train
  data_test[,-which(names(data_test) %in% names_others)] <- x_test
  w_train <- generate_weights(y_train)
  w_test <- generate_weights(y_test)
  #filter predictors
  COEFS <- initiate_store(names_predictors, c("coefficients", "p-values"))
  y <- data_train[,target]
  #test each of the predictors one by one
  if (analysis == "age_at_collection")
  {
    for (name_predictor in names_predictors)
    {
      x <- data_train[, name_predictor]
      model <- lm(y~x)
      COEFS[name_predictor,] <- summary(model)$coefficients["x",c(1,4)]
    }
    table(COEFS$`p-values` < 0.05/nrow(COEFS))
    #index of predictors to select
    if(length(which(COEFS$`p-values` < 0.05)) < 1000)
    {
      index <- rownames(COEFS)[order(COEFS$`p-values`, decreasing = FALSE)[1:min(length(names_predictors),1000)]]
    } else {
      index <- rownames(COEFS)[order(abs(COEFS$coefficients), decreasing = TRUE)[1:1000]]
    }
  } else if(analysis %in% c("country"))
  {
    for (name_predictor in names_predictors)
    {
      x <- data_train[, name_predictor]
      model <- multinom(y ~ x, trace = FALSE)
      COEFS[name_predictor, "coefficients"] <- max(abs(summary(model)$coefficients[,"x"]))
      z <- summary(model)$coefficients[,"x"]/summary(model)$standard.errors[,"x"]
      COEFS[name_predictor, "p-values"] <- min(1 - pnorm(abs(z), 0, 1))*2 # 2-tailed Wald z tests to test significance of coefficients
    }
    #index of predictors to select
    if(length(which(COEFS$`p-values` < 0.05)) < 1000)
    {
      index <- rownames(COEFS)[order(COEFS$`p-values`, decreasing = FALSE)[1:min(length(names_predictors),1000)]]
    } else {
      index <- rownames(COEFS)[order(abs(COEFS$coefficients), decreasing = TRUE)[1:1000]]
    }
  } else {
    for (name_predictor in names_predictors)
    {
      x <- data_train[, name_predictor]
      model <- glm(y~x, family = "binomial")
      COEFS[name_predictor,] <- summary(model)$coefficients["x",c(1,4)]
    }
    #COEFS$`p-values` <- COEFS$`p-values`*nrow(COEFS) #do not correct for multiple testing otherwise only 3 cag remain for gender. be less conservative.
    #index of predictors to select
    if(length(which(COEFS$`p-values` < 0.05)) < 1000)
    {
      index <- rownames(COEFS)[order(COEFS$`p-values`, decreasing = FALSE)[1:min(length(names_predictors),1000)]]
    } else {
      index <- rownames(COEFS)[order(abs(COEFS$coefficients), decreasing = TRUE)[1:1000]]
    }
  }
  print(COEFS[index[1:100],])
  print(table(COEFS$`p-values` < 0.05/nrow(COEFS)))
  print(table(COEFS$coefficients < 0))
  #train
  data_train <- data_train[, c(names_others, index)]
  var_to_remove <- target
  x_train = as.matrix(data_train[,-which(names(data_train) %in% var_to_remove)])
  y_train < data_train[, target]
  y_train <- data_train[, target]
  #test
  data_test <- data_test[, c(names_others, index)]
  x_test = as.matrix(data_test[,-which(names(data_test) %in% var_to_remove)])
  y_test <- data_test[, target]
  preprocessed_data <- list("data_train" = data_train, "x_train" = x_train, "y_train" = y_train, "w_train" = w_train, "data_test" = data_test, "x_test" = x_test, "y_test" = y_test, "w_test" = w_test)
  saveRDS(preprocessed_data, paste(path_compute, "preprocessed_data_", analysis, "_", paste("demo", predictors, sep="+"), "_", i, ".Rda", sep = ""))
}

preprocessing_mixed <- function(analysis, i)
{
  for(set in c("train", "test"))
  {
    data <- readRDS(paste(path_compute, "preprocessed_data_", analysis, "_", "demo+taxa", "_", i, ".Rda", sep = ""))[[paste("data", set, sep="_")]]
    for (predictors in c("cags", "pathways"))
    {
      data_p <- readRDS(paste(path_compute, "preprocessed_data_", analysis, "_", paste("demo", predictors, sep="+"), "_", i, ".Rda", sep = ""))[[paste("data", set, sep="_")]]
      names_predictors <- names(data_p)[which(!(names(data_p) %in% names_variables))]
      data <- cbind(data, data_p[,names_predictors])
    }
    string <- "demo+glmnet"
    data_p <- readRDS(paste(path_compute, "preprocessed_data_", analysis, "_", string, "_", i, ".Rda", sep = ""))[[paste("data", set, sep="_")]]
    names_predictors <- names(data_p)[which(!(names(data_p) %in% names_variables))]
    data <- cbind(data, data_p[,names_predictors])
    assign(paste("data", set, sep="_"), data)
  }
  names_predictors <- names(data_train)[-which(names(data_train) %in% names_variables)]
  names_others <- names(data_train)[-which(names(data_train) %in% names_predictors)]
  names_others <- names_others[which(!(names_others == "subjectID"))]
  y <- readRDS(paste(path_compute, "preprocessed_data_", analysis, "_", "demo+taxa", "_", i, ".Rda", sep = ""))[["y_train"]]
  COEFS <- initiate_store(names_predictors, c("coefficients", "p-values"))
  if (analysis == "age_at_collection")
  {
    for (name_predictor in names_predictors)
    {
      x <- data_train[, name_predictor]
      model <- lm(y~x)
      COEFS[name_predictor,] <- summary(model)$coefficients["x",c(1,4)]
    }
    table(COEFS$`p-values` < 0.05/nrow(COEFS))
    #index of predictors to select
    if(length(which(COEFS$`p-values` < 0.05)) < 1000)
    {
      index <- rownames(COEFS)[order(COEFS$`p-values`, decreasing = FALSE)[1:min(length(names_predictors),1000)]]
    } else {
      index <- rownames(COEFS)[order(abs(COEFS$coefficients), decreasing = TRUE)[1:1000]]
    }
  } else if(analysis %in% c("country"))
  {
    for (name_predictor in names_predictors)
    {
      x <- data_train[, name_predictor]
      model <- multinom(y ~ x, trace = FALSE)
      COEFS[name_predictor, "coefficients"] <- max(abs(summary(model)$coefficients[,"x"]))
      z <- summary(model)$coefficients[,"x"]/summary(model)$standard.errors[,"x"]
      COEFS[name_predictor, "p-values"] <- min(1 - pnorm(abs(z), 0, 1))*2 # 2-tailed Wald z tests to test significance of coefficients
    }
    #index of predictors to select
    if(length(which(COEFS$`p-values` < 0.05)) < 1000)
    {
      index <- rownames(COEFS)[order(COEFS$`p-values`, decreasing = FALSE)[1:min(length(names_predictors),1000)]]
    } else {
      index <- rownames(COEFS)[order(abs(COEFS$coefficients), decreasing = TRUE)[1:1000]]
    }
  } else {
    for (name_predictor in names_predictors)
    {
      x <- data_train[, name_predictor]
      model <- glm(y~x, family = "binomial")
      COEFS[name_predictor,] <- summary(model)$coefficients["x",c(1,4)]
    }
    #COEFS$`p-values` <- COEFS$`p-values`*nrow(COEFS) #do not correct for multiple testing otherwise only 3 cag remain for gender. be less conservative.
    #index of predictors to select
    if(length(which(COEFS$`p-values` < 0.05)) < 1000)
    {
      index <- rownames(COEFS)[order(COEFS$`p-values`, decreasing = FALSE)[1:min(length(names_predictors),1000)]]
    } else {
      index <- rownames(COEFS)[order(abs(COEFS$coefficients), decreasing = TRUE)[1:1000]]
    }
  }
  print(COEFS[index[1:100],])
  print(table(COEFS$`p-values` < 0.05/nrow(COEFS)))
  print(table(COEFS$coefficients < 0))
  to_remove <- target
  data_train <- data_train[, c(names_others, index)]
  data_test <- data_test[, c(names_others, index)]
  preprocessed_data <- readRDS(paste(path_compute, "preprocessed_data_", analysis, "_", "demo+taxa", "_", i, ".Rda", sep = ""))
  preprocessed_data[["data_train"]] <- data_train
  preprocessed_data[["x_train"]] = as.matrix(data_train[,-which(names(data_train) %in% to_remove)])
  preprocessed_data[["data_test"]] <- data_test
  preprocessed_data[["x_test"]] = as.matrix(data_test[,-which(names(data_test) %in% to_remove)])
  saveRDS(preprocessed_data, paste(path_compute, "preprocessed_data_", analysis, "_", "demo+mixed", "_", i, ".Rda", sep = ""))
}

preprocessing_nodemo <- function(analysis, i)
{
  target <- target_of_analysis(analysis)
  to_remove <- names_variables[which(!(names_variables==target))]
  for (predictors in predictorsS_withdemo)
  {
    if (predictors == "demo+genes")
    {
      algos_genes_list <- algos_genes
      for (algo in algos_genes_list)
      {
        preprocessed_data <- readRDS(paste(path_compute, "preprocessed_data_", analysis, "_", paste("demo", algo, sep="+"), "_", i, ".Rda", sep = ""))
        preprocessed_data$data_train <- preprocessed_data$data_train[,-which(names(preprocessed_data$data_train) %in% to_remove)]
        preprocessed_data$data_test <- preprocessed_data$data_test[,-which(names(preprocessed_data$data_test) %in% to_remove)]
        index_x <- which(names(data.frame(preprocessed_data$x_train)) %in% to_remove)
        preprocessed_data$x_train <- preprocessed_data$x_train[,-index_x]
        preprocessed_data$x_test <- preprocessed_data$x_test[,-index_x]
        saveRDS(preprocessed_data, paste(path_compute, "preprocessed_data_", analysis, "_", gsub("\\+.*","", algo), "_", i, ".Rda", sep = ""))
      }
    } else {
      preprocessed_data <- readRDS(paste(path_compute, "preprocessed_data_", analysis, "_", predictors, "_", i, ".Rda", sep = ""))
      preprocessed_data$data_train <- preprocessed_data$data_train[,-which(names(preprocessed_data$data_train) %in% to_remove)]
      preprocessed_data$data_test <- preprocessed_data$data_test[,-which(names(preprocessed_data$data_test) %in% to_remove)]
      index_x <- which(names(data.frame(preprocessed_data$x_train)) %in% to_remove)
      preprocessed_data$x_train <- preprocessed_data$x_train[,-index_x]
      preprocessed_data$x_test <- preprocessed_data$x_test[,-index_x]
      saveRDS(preprocessed_data, paste(path_compute, "preprocessed_data_", analysis, "_", gsub(".*\\+","", predictors), "_", i, ".Rda", sep = ""))
    }
  }
}

index_common_rows <- function(df1, df2)
{
  index <- c()
  for (i in seq(nrow(df1)))
  {
    if (nrow(suppressMessages(inner_join(df1[i,], df2)))==0){index <- c(index, i)}
  }
  return(index)
}

filter_results_hyperparameter <- function(name_hyperparameter, hyperparameters_list, results)
{
  results_hyperparameter <- c()
  for (hyperparameter in hyperparameters_list)
  {
    results_hyperparameter <- c(results_hyperparameters, max(results[which(results[,name_hyperparameter]==hyperparameter),metric]))
  }
  return(results_hyperparameter)
}

generate_new_hyperparameters_list <- function(name_hyperparameter, old_hyperparameters, old_performances, number_hyperparameters, number_extra_hyperparameters)
{
  #number_hyperparameters = numbers of values between the two best values, those two values excluded. Total number of parameters = number_hyperparameters + number_extra_hyperparameters + 1.
  best_hyperparameter <- old_hyperparameters[which.max(old_performances)]
  #if several values gave the best prediction, then return one of these values, the one that gives the simpler model.
  if(table(old_performances)[as.character(max(old_performances))]>1)
  {
    old_hyperparameters <- old_hyperparameters[which(old_performances==max(old_performances))]
    if (name_hyperparameter %in% c("lambda", "alpha")){
      return(max(old_hyperparameters))
    } else  { #mtries, C, sigma, K, fL, adjust
      return(min(old_hyperparameters)) #deal with gbm, ...
    }
  }
  second_hyperparameter <- old_hyperparameters[which.max(old_performances[old_performances!=max(old_performances)])]
  stride <- floor((second_hyperparameter - best_hyperparameter)/(number_hyperparameters+2-number_extra_hyperparameters))
  new_hyperparameters <- seq(best_hyperparameter-number_extra_hyperparameters*stride, second_hyperparameter, by=stride)
  new_hyperparameters <- new_hyperparameters[-which(new_hyperparameters %in% old_hyperparameters)]
  return(new_hyperparameters)
}

boot_performance_binomial <- function(data, indices)
{
  data = data[indices,]
  y <- data$y
  w <- data$w
  pred_class <- data$pred_class
  pred_prob <- data[,-which(names(data) %in% c("y", "w", "pred_class"))]
  #some algorithms (nb so far) sometimes generate NaN for the pred_prob. Clean the data.
  index_tokeep <- complete.cases(pred_prob)
  y_prob <- y[index_tokeep]
  pred_prob <- pred_prob[index_tokeep,]
  if(length(y_prob)>0)
  {
    if(length(y_prob) < length(y)){print(paste(length(y) - length(y_prob), "samples had NaN probabilities predictions.", sep =""))}
    Cross_Entropy <- cross_entropy_weighted(dummy(y,drop=F), pred_prob, w)
    pred_prob_1c <- pred_prob[,2]
    error <- tryCatch(ROC <- as.numeric(auc(roc(y, pred_prob_1c))), error=function(err) "error")
    if(grepl("error", error))
    {
      ROC <- NA
    }
  } else {
    print("The fold has only missing values for the probabilities predictions.")
    Cross_Entropy <- NA
    ROC <- NA
  }
  TP <- sum(pred_class == "X1" & y == "X1")
  TN <- sum(pred_class == "X0" & y == "X0")
  FP <- sum(pred_class == "X1" & y == "X0")
  FN <- sum(pred_class == "X0" & y == "X1")
  Accuracy <- (TP + TN)/(TP + TN + FP + FN)
  Sensitivity <- TP/(TP + FN)
  Specificity <- TN/(TN+FP)
  Mean_Accuracy <- (Sensitivity + Specificity)/2
  return(c(ROC, Cross_Entropy, Mean_Accuracy, Accuracy, Sensitivity, Specificity))
}

merge_results <- function(file_name, analysis, metric, set)
{
  if (analysis=="age_at_collection") {
    algos_analysis <- algos_regression
  } else {
    algos_analysis <- algos
  }
  Performance <- initiate_store(algos_analysis, predictorsS)*NA
  for(predictors in predictorsS)
  {
    col_perf <-  readRDS(paste(path_compute, file_name, "_", set, "_", analysis, "_", predictors, ".Rda", sep = ""))[[metric]]
    Performance[1:length(col_perf),predictors] <- col_perf
  }
  saveRDS(Performance, paste(path_store, file_name, "_", set, "_", analysis, "_", metric, ".Rda", sep = ""))
  return(Performance)
}

boot_performance_country <- function(data, indices)
{
  data = data[indices,]
  y <- data$y
  w <- data$w
  pred_class <- data$pred_class
  pred_prob <- data[,-which(names(data) %in% c("y", "w", "pred_class"))]
  #some algorithms (nb so far) sometimes generate NaN for the pred_prob. Clean the data.
  index_tokeep <- complete.cases(pred_prob)
  y_prob <- y[index_tokeep]
  pred_prob <- pred_prob[index_tokeep,]
  if(length(y_prob)>0)
  {
    Cross_Entropy <- cross_entropy_weighted(dummy(y,drop=F), pred_prob, w)
    if(length(y_prob) < length(y)){print(paste(length(y) - length(y_prob), "samples had NaN probabilities predictions.", sep =""))}
  } else {
    Cross_Entropy <- NA
    print("The fold has only missing values for the probabilities predictions.")
  }
  y <- as.character(y)
  Accuracy <- mean(y == pred_class)
  FIN <- sum(pred_class == "FIN" & y == "FIN") /table(y)["FIN"]
  EST <- sum(pred_class == "EST" & y == "EST")/table(y)["EST"]
  RUS <- sum(pred_class == "RUS" & y == "RUS") /table(y)["RUS"]
  SWE <- sum(pred_class == "SWE" & y == "SWE") /table(y)["SWE"]
  Mean_Accuracy <- (FIN + EST + RUS + SWE)/4
  return(c(Cross_Entropy, Mean_Accuracy, Accuracy, FIN, EST, RUS, SWE))
}

post_processing_regression <- function(analysis, predictors, set)
{
  ifelse(predictors %in% c("genes", "demo+genes"), algos_list_pp <- algos_genes, algos_list_pp <- algos_regression)
  Performance <- initiate_store(algos_list_pp , c("R2"))
  Performance[,] <- NA
  Performance_sd <- Performance
  for (algo in rownames(Performance))
  {
    print(paste("Computing performance for algorithm ", algo, ".", sep=""))
    if(predictors=="demographics") {
      string <- "cags"
    } else if (predictors=="genes") {
      string <- algo
    } else if (predictors=="demo+genes") {
      string <- paste("demo", sep="+", algo)
    } else {
      string <- predictors}
    y <- c()
    pred <- c()
    Performance_algo <- initiate_store(c("all", seq(0,N_CV_Folds)) , c("R2"))
    Performance_algo[,] <- NA
    Performance_algo_sd <- Performance_algo
    for (i in seq(0,N_CV_Folds))
    {
      print(paste("Computing performance for fold ", i, ".", sep=""))
      y_i <- readRDS(paste(path_compute, "preprocessed_data_", analysis, "_", string, "_", i, ".Rda", sep = ""))[[paste("y", set, sep = "_")]]
      pred_i <- readRDS(paste(path_compute, "pred_", set, "_", analysis, "_", predictors, "_", algo, "_", i, ".Rda", sep = ""))
      index_not_NA <- which(!is.na(pred_i))
      y_i <- y_i[index_not_NA]
      w_i <- seq(length(y_i))*0+1
      pred_i <- pred_i[index_not_NA]
      if(!(i==0))
      {
        y <- c(y, y_i)
        pred <- c(pred, pred_i)
      }
      perf_boot <- boot(data = data.frame("y"=y_i, "pred"=pred_i), statistic = boot_R2, R = R_boot)
      Performance_algo[as.character(i),] <- perf_boot$t0
      Performance_algo_sd[as.character(i),] <- sapply(data.frame(perf_boot$t), sd, na.rm=T)
      if(i==0){sample_sizes <- initiate_store(c("all", seq(0,N_CV_Folds)), paste("N", set, sep="_"))}
      sample_sizes[as.character(i),] <- length(y_i)
    }
    w <- generate_weights(y)
    perf_boot <- boot(data = data.frame("y"=y, "pred"=pred), statistic = boot_R2, R = R_boot)
    mean_boot <- perf_boot$t0
    sd_boot <- sapply(data.frame(perf_boot$t), sd, na.rm=T)
    Performance_algo["all",] <- mean_boot
    Performance_algo_sd["all",] <- sd_boot
    sample_sizes["all",] <- length(y)
    Performance[algo,] <- mean_boot
    Performance_sd[algo,] <- sd_boot
    saveRDS(Performance_algo, paste(path_compute, "Performance_", set, "_", analysis, "_", predictors, "_", algo, ".Rda", sep = ""))
    saveRDS(Performance_algo_sd, paste(path_compute, "Performance_sd_", set, "_", analysis, "_", predictors, "_", algo, ".Rda", sep = ""))
    saveRDS(sample_sizes, paste(path_compute, "sample_sizes_", set, "_", analysis, "_", predictors, "_", algo, ".Rda", sep = ""))
    print(paste("The performance for the ", algo, " algorithm is:", sep=""))
    print(Performance_algo)
    print(Performance_algo_sd)
  }
  saveRDS(Performance, paste(path_compute, "Performance_", set, "_", analysis, "_", predictors, ".Rda", sep = ""))
  saveRDS(Performance_sd, paste(path_compute, "Performance_sd_", set, "_", analysis, "_", predictors, ".Rda", sep = ""))
  print("Performance summary:")
  print(Performance)
  print(Performance_sd)
  return(Performance)
}

post_processing_binomial <- function(analysis, predictors, set)
{
  ifelse(predictors %in% c("genes", "demo+genes"), algos_list_pp <- algos_genes, algos_list_pp <- algos)
  Performance <- initiate_store(algos_list_pp , c("ROC", "Cross_Entropy", "Mean_Accuracy", "Accuracy","Sensitivity", "Specificity"))
  Performance[,] <- NA
  Performance_sd <- Performance
  for (algo in rownames(Performance))
  {
    print(paste("Computing performance for algorithm ", algo, ".", sep=""))
    if(predictors=="demographics") {
      string <- "cags"
    } else if (predictors=="genes") {
      string <- algo
    } else if (predictors=="demo+genes") {
      string <- paste("demo", sep="+", algo)
    } else {
      string <- predictors}
    y <- c()
    pred_class <- c()
    pred_prob <- c()
    Performance_algo <- initiate_store(c("all", seq(0,N_CV_Folds)) , c("ROC", "Cross_Entropy", "Mean_Accuracy", "Accuracy","Sensitivity", "Specificity"))
    Performance_algo[,] <- NA
    Performance_algo_sd <- Performance_algo
    for (i in seq(0,N_CV_Folds))
    {
      print(paste("Computing performance for fold ", i, ".", sep=""))
      y_i <- make.names(readRDS(paste(path_compute, "preprocessed_data_", analysis, "_", string, "_", i, ".Rda", sep = ""))[[paste("y", set, sep = "_")]])
      pred_class_i <- as.character(readRDS(paste(path_compute, "pred_class_", set, "_", analysis, "_", predictors, "_", algo, "_", i, ".Rda", sep = "")))
      pred_prob_i <- readRDS(paste(path_compute, "pred_prob_", set, "_", analysis, "_", predictors, "_", algo, "_", i, ".Rda", sep = ""))
      index_not_NA <- which(!is.na(pred_class_i))
      y_i <- y_i[index_not_NA]
      w_i <- generate_weights(y_i)
      pred_class_i <- pred_class_i[index_not_NA]
      pred_prob_i <- pred_prob_i[index_not_NA,]
      if(!(i==0))
      {
        y <- c(y, y_i)
        pred_class <- c(pred_class, pred_class_i)
        pred_prob <- rbind(pred_prob, pred_prob_i)
      }
      perf_boot <- boot(data = cbind(data.frame("y"=y_i, "w"=w_i, "pred_class"=pred_class_i), pred_prob_i), statistic = boot_performance_binomial, R = R_boot)
      Performance_algo[as.character(i),] <- perf_boot$t0
      Performance_algo_sd[as.character(i),] <- sapply(data.frame(perf_boot$t), sd, na.rm=T)
      if(i==0){sample_sizes <- initiate_store(c("all", seq(0,N_CV_Folds)), c(paste("N", set, sep="_"), paste("N", set, levels(y_i), sep="_")))}
      sample_sizes[as.character(i),] <- c(length(y_i), table(y_i))
    }
    w <- generate_weights(y)
    perf_boot <- boot(data = cbind(data.frame("y"=y, "w"=w, "pred_class"=pred_class), pred_prob), statistic = boot_performance_binomial, R = R_boot)
    mean_boot <- perf_boot$t0
    sd_boot <- sapply(data.frame(perf_boot$t), sd, na.rm=T)
    sample_sizes["all",] <- c(length(y), table(y)[levels(y_i)])
    Performance_algo["all",] <- mean_boot
    Performance_algo_sd["all",] <- sd_boot
    Performance[algo,] <- mean_boot
    Performance_sd[algo,] <- sd_boot
    saveRDS(Performance_algo, paste(path_compute, "Performance_", set, "_", analysis, "_", predictors, "_", algo, ".Rda", sep = ""))
    saveRDS(Performance_algo_sd, paste(path_compute, "Performance_sd_", set, "_", analysis, "_", predictors, "_", algo, ".Rda", sep = ""))
    saveRDS(sample_sizes, paste(path_compute, "sample_sizes_", set, "_", analysis, "_", predictors, "_", algo, ".Rda", sep = ""))
    print(paste("The performance for the ", algo, " algorithm is:", sep=""))
    print(Performance_algo)
    print(Performance_algo_sd)
  }
  saveRDS(Performance, paste(path_compute, "Performance_", set, "_", analysis, "_", predictors, ".Rda", sep = ""))
  saveRDS(Performance_sd, paste(path_compute, "Performance_sd_", set, "_", analysis, "_", predictors, ".Rda", sep = ""))
  print("Performance summary:")
  print(Performance)
  print(Performance_sd)
  return(Performance)
}

post_processing_multinomial <- function(analysis, predictors, set)
{
  classes_target <- c("FIN", "EST", "RUS", "SWE")
  ifelse(predictors %in% c("genes", "demo+genes"), algos_list_pp <- algos_genes, algos_list_pp <- algos)
  Performance <- initiate_store(algos_list_pp , c("Cross_Entropy", "Mean_Accuracy", "Accuracy", classes_target))
  Performance[,] <- NA
  Performance_sd <- Performance
  for (algo in rownames(Performance))
  {
    print(paste("Computing performance for algorithm ", algo, ".", sep=""))
    if(predictors=="demographics") {
      string <- "cags"
    } else if (predictors=="genes") {
      string <- algo
    } else if (predictors=="demo+genes") {
      string <- paste("demo", sep="+", algo)
    } else {
      string <- predictors}
    y <- c()
    pred_class <- c()
    pred_prob <- c()
    Performance_algo <- initiate_store(c("all", seq(0,N_CV_Folds)), c("Cross_Entropy", "Mean_Accuracy", "Accuracy", classes_target))
    Performance_algo[,] <- NA
    Performance_algo_sd <- Performance_algo
    for (i in seq(0,N_CV_Folds))
    {
      print(paste("Computing performance for fold ", i, ".", sep=""))
      y_i <- readRDS(paste(path_compute, "preprocessed_data_", analysis, "_", string, "_", i, ".Rda", sep = ""))[[paste("y", set, sep = "_")]]
      pred_class_i <- as.character(readRDS(paste(path_compute, "pred_class_", set, "_", analysis, "_", predictors, "_", algo, "_", i, ".Rda", sep = "")))
      pred_prob_i <- readRDS(paste(path_compute, "pred_prob_", set, "_", analysis, "_", predictors, "_", algo, "_", i, ".Rda", sep = ""))
      index_not_NA <- which(!is.na(pred_class_i))
      y_i <- y_i[index_not_NA]
      w_i <- generate_weights(y_i)
      pred_class_i <- pred_class_i[index_not_NA]
      pred_prob_i <- pred_prob_i[index_not_NA,]
      if(!(i==0))
      {
        y <- c(y, y_i)
        pred_class <- c(pred_class, pred_class_i)
        pred_prob <- rbind(pred_prob, pred_prob_i)
      }
      #generate bootstrapped summary statistics
      perf_boot <- boot(data = cbind(data.frame("y"=y_i, "w"=w_i, "pred_class"=pred_class_i), pred_prob_i), statistic = get(paste("boot_performance", analysis, sep = "_")), R = R_boot)
      Performance_algo[as.character(i),] <- perf_boot$t0
      Performance_algo_sd[as.character(i),] <- sapply(data.frame(perf_boot$t), sd, na.rm=T)
      if(i==0){sample_sizes <- initiate_store(c("all", seq(0,N_CV_Folds)), c(paste("N", set, sep="_"), paste("N", set, levels(y_i), sep="_")))}
      sample_sizes[as.character(i),] <- c(length(y_i), table(y_i))
    }
    #generate confusion matrix
    print(algo)
    y <- levels(y_i)[as.numeric(y)]
    w <- generate_weights(y)
    perf_boot <- boot(data = cbind(data.frame("y"=y, "w"=w, "pred_class"=pred_class), pred_prob), statistic = get(paste("boot_performance", analysis, sep = "_")), R = R_boot)
    mean_boot <- perf_boot$t0
    sd_boot <- sapply(data.frame(perf_boot$t), sd, na.rm=T)
    Performance_algo["all",] <- mean_boot
    Performance_algo_sd["all",] <- sd_boot
    sample_sizes["all",] <- c(length(y), table(y)[levels(y_i)])
    Performance[algo,] <- mean_boot
    Performance_sd[algo,] <- sd_boot
    print(confusionMatrix(factor(pred_class, levels = classes_target), factor(y, levels = classes_target)))
    print(Performance_algo)
    print(Performance_algo_sd)
    saveRDS(Performance_algo, paste(path_compute, "Performance_", set, "_", analysis, "_", predictors, "_", algo, ".Rda", sep = ""))
    saveRDS(Performance_algo_sd, paste(path_compute, "Performance_sd_", set, "_", analysis, "_", predictors, "_", algo, ".Rda", sep = ""))
    saveRDS(sample_sizes, paste(path_compute, "sample_sizes_", set, "_", analysis, "_", predictors, "_", algo, ".Rda", sep = ""))
    print(paste("The performance for the ", algo, " algorithm is:", sep=""))
    print(Performance_algo)
    print(Performance_algo_sd)
  }
  saveRDS(Performance, paste(path_compute, "Performance_", set, "_", analysis, "_", predictors, ".Rda", sep = ""))
  saveRDS(Performance_sd, paste(path_compute, "Performance_sd_", set, "_", analysis, "_", predictors, ".Rda", sep = ""))
  print("Performance summary:")
  print(Performance)
  print(Performance_sd)
  return(Performance)
}

transform_hyper <- function(list_hyper, hyper_fun)
{
  transformed_hyper <- list()
  for (hyper in names(list_hyper)){transformed_hyper[[hyper]] <- hyper_fun[[hyper]](list_hyper[[hyper]])}
  return(transformed_hyper)
}

ten_power_x <- function(x){return(10**x)}

index_different_rows <- function(df1, df2)
{
  index <- c()
  for (i in seq(nrow(df1)))
  {
    if (nrow(suppressMessages(inner_join(df1[i,,drop=FALSE], df2)))==0){index <- c(index, i)}
  }
  return(index)
}

testing <- function(analysis, target, predictors, algo, i, data, x, y, w, set)
{
  model <- readRDS(paste(path_compute, "model", "_", analysis, "_", predictors, "_", algo, "_", i,".Rda", sep = ""))
  args_predict <- list("object"=model, "newdata"=data)
  if (analysis == "age_at_collection")
  {
    if (algo=="glmnet2")
    {
      args_predict[["newdata"]] <- NULL
      args_predict[["newx"]] <- x
      args_predict[["s"]] <- "lambda.min"
    }
    pred <- do.call(predict, args=args_predict)
    saveRDS(pred, paste(path_compute, "pred_", set, "_", analysis, "_", predictors, "_", algo, "_", i, ".Rda", sep = ""))
  } else if (analysis %in% c("country")) {
    if (algo=="glmnet2") {
      pred_prob <- data.frame(predict(model, x, s="lambda.min", type ="response")[,,1])
      pred_class <- predict(model, x, s="lambda.min", type ="class")
    } else if (algo=="gbm2") {
      pred_prob <- data.frame(predict.gbm(model, data, type="response")[,,1])
      pred_class <- names(pred_prob)[apply(pred_prob, 1, which.max)]
    } else if (algo=="rf2") {
      pred_prob <- data.frame(predict(model, data, type="prob"))
      pred_class <- as.character(predict(model, data, type="class"))
    } else {
      args_predict[["type"]] <- "prob"
      pred_prob <- do.call(predict, args=args_predict)
      args_predict[["type"]] <- "raw"
      pred_class <- as.character(do.call(predict, args=args_predict))
      #how to handle NA values generated
      if(any(is.na(pred_class)) | any(is.na(pred_prob)))
      {
        pred_prob <- initiate_store(seq(nrow(data)), levels(predict(model, data[1,], type="raw")))
        pred_class <- c()
        for (j in seq(nrow(data)))
        {
          p_prob <- predict(model, data[j,], type="prob")
          ifelse(anyNA(p_prob), pred_prob[j,] <- NA, pred_prob[j,] <- p_prob)
          p_class <- predict(model, data[j,], type="raw")
          ifelse(is.na(p_class), pred_class <- c(pred_class,NA), pred_class <- c(pred_class, levels(p_class)[p_class]))
        }
      }
    }
    saveRDS(pred_prob, paste(path_compute, "pred_prob_", set, "_", analysis, "_", predictors, "_", algo, "_", i, ".Rda", sep = ""))
    saveRDS(pred_class, paste(path_compute, "pred_class_", set, "_", analysis, "_", predictors, "_", algo, "_", i, ".Rda", sep = ""))
  } else {
    if (algo=="glmnet2") {
      pred_prob <- predict(model, x, s="lambda.min", type ="response")[,1]
      pred_prob <- data.frame("X0"=1-pred_prob, "X1"=pred_prob)
      pred_class <- make.names(predict(model, x, s="lambda.min", type ="class"))
    } else if (algo=="gbm2") {
      #assignInNamespace("plot.window", plot.window, "graphics")
      pred_prob <- predict.gbm(object = model, newdata = data, type = "response") #n.trees = n_trees by default
      pred_prob <- data.frame("X0"=1-pred_prob, "X1"=pred_prob)
      pred_class <- names(pred_prob)[apply(pred_prob, 1, which.max)]
    } else if (algo=="rf2") {
      pred_prob <- data.frame(predict(model, data, type="prob"))
      pred_class <- as.character(predict(model, data, type="class"))
    } else {
      args_predict[["type"]] <- "prob"
      pred_prob <- do.call(predict, args=args_predict)
      args_predict[["type"]] <- "raw"
      pred_class <- as.character(do.call(predict, args=args_predict))
    }
    names(pred_prob) <- c("X0", "X1")
    saveRDS(pred_prob, paste(path_compute, "pred_prob_", set, "_", analysis, "_", predictors, "_", algo, "_", i, ".Rda", sep = ""))
    saveRDS(pred_class, paste(path_compute, "pred_class_", set, "_", analysis, "_", predictors, "_", algo, "_", i, ".Rda", sep = ""))
  }
}

predict_debug <- function(model,data)
{
  args_predict <- list("object"=model, "newdata"=data)
  if (analysis == "age_at_collection")
  {
    if (algo=="glmnet2")
    {
      args_predict[["newdata"]] <- NULL
      args_predict[["newx"]] <- x
      args_predict[["s"]] <- "lambda.min"
    }
    pred <- do.call(predict, args=args_predict)
  } else if (analysis %in% c("country")) {
    if (algo=="glmnet2") {
      pred_prob <- data.frame(predict(model, x, s="lambda.min", type ="response")[,,1])
      pred_class <- predict(model, x, s="lambda.min", type ="class")
    } else if (algo=="gbm2") {
      pred_prob <- data.frame(predict.gbm(model, data)[,,1])
      pred_class <- names(pred_prob)[apply(pred_prob, 1, which.max)]
    } else if (algo=="rf2") {
      pred_prob <- data.frame(predict(model, data, type="prob"))
      pred_class <- as.character(predict(model, data, type="class"))
    } else {
      args_predict[["type"]] <- "prob"
      pred_prob <- do.call(predict, args=args_predict)
      args_predict[["type"]] <- "raw"
      pred_class <- as.character(do.call(predict, args=args_predict))
    }
  } else {
    if (algo=="glmnet2") {
      pred_prob <- predict(model, x, s="lambda.min", type ="response")[,1]
      pred_prob <- data.frame("X0"=1-pred_prob, "X1"=pred_prob)
      pred_class <- make.names(predict(model, x, s="lambda.min", type ="class"))
    } else if (algo=="gbm2") {
      #assignInNamespace("plot.window", plot.window, "graphics")
      pred_prob <- predict.gbm(object = model, newdata = data, type = "response") #n.trees = n_trees by default
      pred_prob <- data.frame("X0"=1-pred_prob, "X1"=pred_prob)
      pred_class <- names(pred_prob)[apply(pred_prob, 1, which.max)]
    } else if (algo=="rf2") {
      pred_prob <- data.frame(predict(model, data, type="prob"))
      pred_class <- as.character(predict(model, data, type="class"))
    } else {
      args_predict[["type"]] <- "prob"
      pred_prob <- do.call(predict, args=args_predict)
      args_predict[["type"]] <- "raw"
      pred_class <- as.character(do.call(predict, args=args_predict))
    }
    names(pred_prob) <- c("X0", "X1")
  }
  ifelse(analysis=="age_at_collection", return(pred), return(pred_class))
}

glmnet2_results_fxn <- function(lambdas, cvm, metric)
{
  args_results <- list("lambda"=lambdas)
  args_results[[metric]] <- cvm
  return(do.call(data.frame, args_results))
}

rf2_results_fxn <- function(mtry, err_cv, metric)
{
  args_results <- list("mtry"=mtry)
  args_results[[metric]] <- 1-err_cv
  return(do.call(data.frame, args_results))
}

best_predictors <- function(model, analysis, predictors, algo, i)
{
  if (algo=="glmnet") {
    if(analysis %in% c("country"))
    {
      variables_importance <- data.frame(as.matrix(coef(model$finalModel, s=model$bestTune$lambda)[[1]]))[-1,,drop=F]*0
      names(variables_importance) <- "Mean_Absolute_Coefficient"
      y_classes <- c("FIN", "EST", "RUS", "SWE")
      for (y_class in y_classes)
      {
        variables_importance_class <- coef(model$finalModel, s=model$bestTune$lambda)[[y_class]][-1]
        variables_importance$Mean_Absolute_Coefficient <- variables_importance$Mean_Absolute_Coefficient + abs(variables_importance_class)
        variables_importance[[dictionary_multinomial$label[which(dictionary_multinomial$name == y_class)]]] <- variables_importance_class
      }
      variables_importance$Mean_Absolute_Coefficient <- variables_importance$Mean_Absolute_Coefficient/length(y_classes)
      variables_importance <- variables_importance[order(variables_importance$Mean_Absolute_Coefficient,decreasing = T),,drop=F]
    } else {
      variables_importance <- data.frame(as.matrix(coef(model$finalModel, s=model$bestTune$lambda)))[,,drop=F]
      variables_importance <- variables_importance[-which(rownames(variables_importance)=="(Intercept)"),,drop=F]
      variables_importance <- variables_importance[order(abs(variables_importance),decreasing = T),,drop=F]
      names(variables_importance) <- "Regression_Coefficients"
    }
  } else if(algo=="glmnet2") {
    ifelse(analysis %in% c("country"), variables_importance <- data.frame(as.matrix(coefficients(model)[[1]])), variables_importance <- data.frame(as.matrix(coef(model))))
    variables_importance <- variables_importance[order(abs(variables_importance[,1]),decreasing = TRUE ),, drop = FALSE]
    variables_importance <- variables_importance[-which(rownames(variables_importance)=="(Intercept)"),,drop=F]
    names(variables_importance) <- "Regression_Coefficients"
  } else if (algo=="gbm") {
    variables_importance <- summary(model)
    rownames(variables_importance) <- variables_importance[,1]
    variables_importance <- variables_importance[,2,drop=F]
    names(variables_importance) <- "Relative_Influences"
  } else if(algo=="gbm2") {
    variables_importance <- summary(model)[,2,drop=FALSE]
    names(variables_importance) <- "Relative_Influences"
  } else if (algo=="rf") {
    variables_importance <- data.frame(model$finalModel$importance)
    variables_importance <- variables_importance[order(abs(variables_importance),decreasing = T),,drop=F]
  } else if(algo=="rf2") {
    variables_importance <- data.frame(model$importance)
    if(!(analysis=="age_at_collection"))
    {
      variables_importance <- variables_importance[order(variables_importance$MeanDecreaseAccuracy, decreasing = T),]
      remaining_names <- names(variables_importance)[which(!(names(variables_importance) %in% c("MeanDecreaseAccuracy", "MeanDecreaseGini")))]
      variables_importance <- variables_importance[,c("MeanDecreaseAccuracy", "MeanDecreaseGini", remaining_names)]
      names(variables_importance)[which(names(variables_importance) %in% remaining_names)] <- dictionary_metric[which(dictionary_metric$name %in% remaining_names),"label"]
    } else {
      variables_importance <- variables_importance[order(variables_importance$X.IncMSE, decreasing = T),]
      names(variables_importance) <- c("MSE_Increase", "Node_Purity_Increase")
    }
  }
  #print(head(variables_importance, 100))
  saveRDS(variables_importance, paste(path_store, "variables_importance_", analysis, "_", predictors, "_", algo, "_", i, ".Rda", sep = ""))
}

tune_glmnet2 <- function(analysis, target, predictors, algo, i, data, x, y, w)
{
  alpha <- 1 #default value for the package, corresponds to ridge regression.
  names_hyper <- c("lambda")
  strides_hyper <- list("lambda"=.5)
  n_extra_leftright <- list("lambda"=2)
  n_fine_tuning <- list("lambda"=5)
  min_hyper <- list("lambda"=-Inf)
  max_hyper <- list("lambda"=Inf)
  is_int_hyper <- list("lambda"=F)
  list_hyper <- list("lambda"=seq(-3,3,by=strides_hyper[["lambda"]]))
  hyper_fun <- list("lambda"=ten_power_x)
  hyper_fun_inv <- list("lambda"=log10)
  transformed_hyper <- transform_hyper(list_hyper, hyper_fun)
  hyper_grid <- do.call(expand.grid, args = transformed_hyper)
  hyper_grid_all <- hyper_grid
  if (analysis %in% c("age_at_collection", "country")) {
    metriC <- "mse"
  } else {
    metriC <- "auc"}
  flds <- createFolds(y, k = N_CV_Folds, list = TRUE, returnTrain = FALSE)
  foldids = rep(1,length(y))
  for (j in seq(N_CV_Folds)){foldids[flds[[paste("Fold",sprintf("%02d", j),sep="")]]] <- j}
  model <- cv.glmnet(x = x, y = y, weights = w, family = families[analysis], lambda = transformed_hyper[["lambda"]], alpha = alpha, type.measure = metriC, nfolds = N_CV_Folds, foldid = foldids, parallel = FALSE, standardize = FALSE)
  results <- glmnet2_results_fxn(model$lambda, model$cvm, metric)
  best_tune <- list("lambda"=model$lambda.min)
  top_performance <- max(results[,metric],na.rm=TRUE)
  print("Model:"); print(model); print("Results:"); print(results); print("Best tune:"); print(best_tune)
  #COARSE GRID SEARCH: keep shifting the grid until the best hyperparameters are not an extremum value
  print("Now starting COARSE GRID SEARCH.")
  coarse_search <- TRUE
  N_cs <- 0 #keep track of the number of iterations when looking for the abscence of extremum values.
  while(coarse_search)
  {
    N_cs <- N_cs+1; print(paste("Current iteration of the COARSE SEARCH, going through every parameter: ", N_cs, ".", sep=""))
    coarse_search <- FALSE #if no extremum is detected for this "epoch", skip the while() the next time
    #for each hyperparameter, check if the best value is an extremum.
    for (hyper in names_hyper)
    {
      results_filtered <- results
      #filter the results and prepare grid: for each other hyperparameter, only consider the values that were involved in performances as good as the top performance.
      for (hyper_other in names_hyper[-which(names_hyper==hyper)])
      {
        best_hyper_other <- results_filtered[,hyper_other][which(results_filtered[,metric]==top_performance)]
        list_hyper[[hyper_other]] <- hyper_fun_inv[[hyper_other]](best_hyper_other)
        results_filtered <- results_filtered[which(results_filtered[,hyper_other] %in% best_hyper_other),]
      }
      #check if the top performance was reached for another value of the explored hyperparameter
      n_equal_best <- nrow(results_filtered[which(results_filtered[[metric]]==top_performance & !(results_filtered[[hyper]]==best_tune[[hyper]])),])
      coarse_search_hyper <- best_tune[[hyper]] %in% c(min(results_filtered[,hyper]), max(results_filtered[,hyper])) & n_equal_best == 0
      if(!coarse_search_hyper){print(paste("This hyperparameter is not (only) an extremum, so no coarse search for this coarse_search iteration: ", hyper, sep =""))}
      N_h <- 0 #keep track of the number of iterations for this hyperparameter.
      deadend <- list("min"=FALSE, "max"=FALSE) #keep track of the directions in which a limit was reached during the search.
      while(coarse_search_hyper)
      {
        N_h <- N_h+1; print(paste("Current iteration for search of HYPERPARAMETER: ", hyper, ". Iteration is: ", N_h, sep=""))
        coarse_search_hyper <- TRUE
        keep_csh <- TRUE
        if(best_tune[[hyper]] == min(results_filtered[,hyper]) & min(hyper_fun_inv[[hyper]](results_filtered[,hyper])) > min_hyper[[hyper]] & !deadend[["min"]])
        {
          print(paste("Best model selected the lowest ", hyper, " value: ", best_tune[[hyper]], sep =""))
          from_seq <- min(hyper_fun_inv[[hyper]](results_filtered[,hyper]))- strides_hyper[[hyper]]
          if(from_seq==Inf){from_seq <- 20}
          new_hyper <- seq(from=from_seq, by=-strides_hyper[[hyper]], length.out=n_extra_leftright[[hyper]])
          new_hyper[which(new_hyper < min_hyper[[hyper]])] <- min_hyper[[hyper]]
          search_direction <- "min"
        } else if (best_tune[[hyper]] == max(results_filtered[,hyper]) & max(hyper_fun_inv[[hyper]](results_filtered[,hyper])) < max_hyper[[hyper]] & !deadend[["max"]])
        {
          print(paste("Best model selected the highest ", hyper, " value: ", best_tune[[hyper]], sep =""))
          from_seq <- max(hyper_fun_inv[[hyper]](results_filtered[,hyper])) + strides_hyper[[hyper]]
          if(from_seq==-Inf){from_seq <- -20}
          new_hyper <- seq(from=from_seq, by=strides_hyper[[hyper]], length.out=n_extra_leftright[[hyper]])
          new_hyper[which(new_hyper > max_hyper[[hyper]])] <- max_hyper[[hyper]]
          search_direction <- "max"
        } else {
          coarse_search_hyper <- FALSE #to keep track of the big csh loop.
          keep_csh <- FALSE #to keep track of different steps in this iteration of the loop. Not the same as "coarse_search_hyper", because can search smaller, then bigger.
        }
        if(keep_csh)
        {
          list_hyper[[hyper]] <- unique(new_hyper)
          transformed_hyper <- transform_hyper(list_hyper, hyper_fun)
          hyper_grid <- do.call(expand.grid, args = transformed_hyper)
          hyper_grid <- hyper_grid[index_different_rows(hyper_grid, hyper_grid_all),,drop=FALSE]
          hyper_grid_all <- rbind(hyper_grid_all, hyper_grid)
          if(nrow(hyper_grid) < 1)
          {
            keep_csh <- FALSE
            print("The grid corresponding to this new value of the hyperparameter has already been explored and found to be suboptimal.")
            deadend[[search_direction]] <- TRUE
          }
        }
        if(keep_csh) #only search for other hypervalues if nrow(hyper_grid) > 0
        {
          error <- tryCatch(model2 <- cv.glmnet(x = x, y = y, weights = w, family = families[analysis], lambda = transformed_hyper[["lambda"]], alpha = alpha, type.measure = metriC, nfolds = N_CV_Folds, foldid = foldids, parallel = FALSE, standardize = FALSE), error=function(err) "error")
          while(grepl("error", error)) #if the model aborts, set new hyperparameter limit
          {
            print("One of the new hyperparameters values returned an error. Therefore the limit for this hyperparameter has been reset.")
            if(min(hyper_grid[[hyper]]) < best_tune[[hyper]])
            {
              toremove_hyper <- min(hyper_grid[[hyper]])
              toremove_inv <- hyper_fun_inv[[hyper]](toremove_hyper)
              min_hyper[[hyper]] <- toremove_inv + strides_hyper[[hyper]]
            } else {
              toremove_hyper <- max(hyper_grid[[hyper]])
              toremove_inv <- hyper_fun_inv[[hyper]](toremove_hyper)
              min_hyper[[hyper]] <- toremove_inv - strides_hyper[[hyper]]
            }
            hyper_grid <- hyper_grid[-which(hyper_grid[[hyper]]==toremove_hyper),,drop=FALSE]
            if(nrow(hyper_grid)>0)
            {
              print("Re-searching after removing one (more) of the new hyperparameter values.")
              error <- tryCatch(model2 <- cv.glmnet(x = x, y = y, weights = w, family = families[analysis], lambda = transformed_hyper[["lambda"]], alpha = alpha, type.measure = metriC, nfolds = N_CV_Folds, foldid = foldids, parallel = FALSE, standardize = FALSE), error=function(err) "error")
            } else {
              print("All new values have been removed. None are left to test.")
              break #break out of the "error-check" loop
            }
          } #end of error checking
          if(nrow(hyper_grid)>0)
          {
            results2 <- glmnet2_results_fxn(model2$lambda, model2$cvm, metric)
            results <- rbind(results, results2)
            results_filtered <- rbind(results_filtered, results2) #to check in next coarse_search_hyper iteration if a better value has been found.
            print(paste("More values were tried for the hyperparameter: ", hyper, ".", sep="")); print("The results for these hyperparameter values are:"); print(model2$cvm)
            if(max(model2$cvm,na.rm=TRUE) <= top_performance){
              print(paste("No better value was found for ", hyper, ".", sep =""))
            } else {
              model <- model2
              best_tune <- list("lambda"=model$lambda.min)
              top_performance <- max(results[,metric],na.rm=TRUE)
              coarse_search <- TRUE #Keep looping the great coarse_search loop because new parameters ranges were introduced, so it is possible that another hyperparameter is now on an extremum.
              print(paste("A better hypervalue was found for the model changing the hyperparameter: ", hyper, ".", sep="")); print("Best tune:"); print(best_tune); print(paste("Performance using metric ", metric, " = ", round(top_performance,4), sep=""))
              #check if the new best hyperparameter is an extremum, of if looping can stop for this hyperparameter.
              results_filtered <- results
              for (hyper_other in names_hyper[-which(names_hyper==hyper)]) #filter the results: only consider the rows of the results for which the other hyperparameters are best tuned.
              {
                best_hyper_other <- results_filtered[,hyper_other][which(results_filtered[,hyper_other]==best_tune[[hyper_other]])]
                results_filtered <- results_filtered[which(results_filtered[,hyper_other] %in% best_hyper_other),]
              }
              n_equal_best <- nrow(results_filtered[which(results_filtered[[metric]]==top_performance & !(results_filtered[[hyper]]==best_tune[[hyper]])),])
              coarse_search_hyper <- best_tune[[hyper]] %in% c(min(results_filtered[,hyper]), max(results_filtered[,hyper])) & n_equal_best == 0
            } #end of ifelse a better model is found => update best model. otherwise go to next iteration.
          } #end of "if nrow>1"
        } else {
          print(paste("The coarse search stopped for ", hyper, " because no other values could/had to be explored in this direction.", sep =""))
        } #end of if nrows > 0 => update. Otherwise stop coarse search.
      } #end of coarse search while loop for a specific hyperparameter.
    } #end of coarse search for each hyperparameter.
  } #end of coarse search while loop
  print(paste("END of COARSE SEARCH. Best tune is: ", best_tune, sep=""))
  #FINE GRID SEARCH: zoom around the best hyperparameters values to fine tune the grid
  print("Now STARTING FINE GRID search.")
  for (hyper in names_hyper)
  {
    center_hyper <- hyper_fun_inv[[hyper]](best_tune[[hyper]])
    best_hypers <- hyper_fun_inv[[hyper]](results[,hyper][which(results[,metric]==top_performance)])
    min_seq <- min(best_hypers)
    max_seq <- max(best_hypers)
    #if there is a plateau of best parameters, explore it. Otherwise, look for best second value. In the other direction, explore half a stride.
    if (min_seq == max_seq)
    {
      results_filtered_without <- results[-which(results[,hyper]==best_tune[[hyper]]),]
      hyper_2nd_bests <- hyper_fun_inv[[hyper]](results_filtered_without[,hyper][which(results_filtered_without[,metric]==max(results_filtered_without[,metric],na.rm=TRUE))])
      hyper_2nd_best <- hyper_2nd_bests[which.min(abs(hyper_2nd_bests-center_hyper))]
      if(hyper_2nd_best > center_hyper)
      {
        min_seq <- max(min_hyper[[hyper]], center_hyper - strides_hyper[[hyper]]/2)
        max_seq <- hyper_2nd_best
      } else {
        min_seq <- hyper_2nd_best
        max_seq <- min(max_hyper[[hyper]], center_hyper + strides_hyper[[hyper]]/2)
      }
    }
    if(min_seq==-Inf) min_seq <- -20
    if(max_seq==Inf) max_seq <- 20
    list_hyper[[hyper]] <- seq(min_seq, max_seq, by=(max_seq-min_seq)/(n_fine_tuning[[hyper]]+1))
    if(is_int_hyper[[hyper]]){list_hyper[[hyper]] <- unique(round(list_hyper[[hyper]]))}
  }
  transformed_hyper <- transform_hyper(list_hyper, hyper_fun)
  hyper_grid <- do.call(expand.grid, args = transformed_hyper)
  hyper_grid <- hyper_grid[index_different_rows(hyper_grid, hyper_grid_all),,drop=FALSE]
  if(nrow(hyper_grid) > 0)
  {
    print("The grid values tested during the fine search are: "); print(hyper_grid)
    hyper_grid_all <- rbind(hyper_grid_all, hyper_grid)
    error <- tryCatch(model2 <- cv.glmnet(x = x, y = y, weights = w, family = families[analysis], lambda = transformed_hyper[["lambda"]], alpha = alpha, type.measure = metriC, nfolds = N_CV_Folds, foldid = foldids, parallel = FALSE, standardize = FALSE), error=function(err) "error")
    if(grepl("error", error)) #if the model aborts, set new hyperparameter limit
    {
      print("At least one of values of the lambda lead to a computing error. Rerunning the values one by one.")
      for (j in seq(length(transformed_hyper[["lambda"]])))
      {
        print(paste("Trying the following lambda value: ", transformed_hyper[["lambda"]][j], sep=""))
        error <- tryCatch(model2 <- cv.glmnet(x = x, y = y, weights = w, family = families[analysis], lambda = c(best_tune[["lambda"]], transformed_hyper[["lambda"]][j]), alpha = alpha, type.measure = metriC, nfolds = N_CV_Folds, foldid = foldids, parallel = FALSE, standardize = FALSE), error=function(err) "error")
        if(grepl("error", error))
        {
          print("This lambda value made the model crash.")
        } else {
          results <- rbind(results, glmnet2_results_fxn(model2$lambda, model2$cvm, metric)[2,])
          print(paste("The performance for this lambda value is: ", model2$cvm[2], sep=""))
          if(!is.na(model2$cvm[2]) & model2$cvm[2] > top_performance)
          {
            model <- model2
            best_tune <- list("lambda"=transformed_hyper[["lambda"]][j])
            top_performance <- model2$cvm[2]
            print(paste("The performance of the model improved. Performance using metric ", metric, " = ", round(top_performance,4), sep=""))
          } else {
            print("This hypervalue did not improve the performance.")
          }
        }
      }
    }
    else {
      results <- rbind(results, glmnet2_results_fxn(model2$lambda, model2$cvm, metric))
      print("More values were tried during the fine search. The results for these hyperparameter values are: "); print(model2$cvm)
      if(max(model2$cvm,na.rm=TRUE) <= top_performance){
        print("No better value was found during the fine search.")
      } else {
        model <- model2
        best_tune <- list("lambda"=model$lambda.min)
        top_performance <- max(results[,metric],na.rm=TRUE)
        print("A better hypervalue was found during the fine search."); print("Best tune:"); print(best_tune); print(paste("Performance using metric ", metric, " = ", round(top_performance,4), sep=""))
      }
    }
  } else {
    print("No better value was found during the fine search.")
  }
  saveRDS(best_tune, paste(path_compute, "hyperparameters", "_", analysis, "_", predictors, "_", algo, "_", i,".Rda", sep = ""))
  print("Results:"); print(results); print("Best tune:"); print(best_tune); print(paste("Performance using metric ", metric, " = ", round(top_performance,4), sep=""))
  return(model)
}

tune_gbm2 <- function(analysis, target, predictors, algo, i, data, x, y, w)
{
  if(!(analysis %in% c("age_at_collection", "country")))
  {
    y <- substring(y,2)
    data$target <- substring(data$target,2)
  }
  class_stratify_cv <- ifelse(analysis=="age_at_collection", FALSE, TRUE)
  n_trees <- 1000
  print(paste("number of trees = ", n_trees, sep=""))
  model <- gbm(formula = target ~., weights=w, data = data, n.trees = n_trees, interaction.depth = 1, n.minobsinnode = 10, shrinkage = 0.1, cv.folds = N_CV_Folds, class.stratify.cv = class_stratify_cv, n.cores = n_cores)
#  print(model) #this line crashes if there is a NaN somewhere in model
  mean_89 <- mean(model$cv.error[ceiling(0.8*n_trees):floor(0.9*n_trees)])
  mean_910 <- mean(model$cv.error[ceiling(0.9*n_trees):n_trees])
  while(which.min(model$cv.error) > floor(0.9*n_trees) & (mean_89-mean_910)/mean_910 > 0.0001)
  {
    print("WARNING! doubling the number of trees")
    n_trees<- n_trees*2
    print(paste("number of trees = ", n_trees, sep=""))
    model <- gbm(formula = target ~., weights=w, data = data, n.trees = n_trees, interaction.depth = 1, n.minobsinnode = 10, shrinkage = 0.1, cv.folds = N_CV_Folds, class.stratify.cv = class_stratify_cv, n.cores = n_cores)
#    print(model) #this line crashes if there is a NaN somewhere in model
    mean_89 <- mean(model$cv.error[ceiling(0.8*n_trees):floor(0.9*n_trees)])
    mean_910 <- mean(model$cv.error[ceiling(0.9*n_trees):n_trees])
  }
  error <- tryCatch(best_tune <- gbm.perf(model, method = "cv"), error=function(err) "error")
  if(grepl("error", error))
  {
    print("There is at least one NA value that made the script the command best_tune <- gbm.perf(model, method = cv) fail because of the plot.window function. Replacing the plot function with an amended function, and rerunning the command.")
    assignInNamespace("plot.window", plot.window, "graphics")
    error <- tryCatch(best_tune <- gbm.perf(model, method = "cv"), error=function(err) "error")
  }
  if(grepl("error", error))
  {
    print("Even after replacing the plot.window function, the command best_tune <- gbm.perf(model, method = cv) failed. The model will be returned and saved, but the hyperparameters need to be extracted later.")
  } else {
    saveRDS(best_tune, paste(path_compute, "hyperparameters", "_", analysis, "_", predictors, "_", algo, "_", i,".Rda", sep = ""))
    print("Results"); print(model$cv.error[best_tune]); print("Best tune:"); print(best_tune)
  }
  return(model)
}

tune_rf2 <- function(analysis, target, predictors, algo, i, data, x, y, w)
{
  print("starting algo")
  if(!(analysis %in% c("age_at_collection", "country")))
  {
    y <- factor(y)
    data$target <- factor(data$target)
  }
  n_trees <- 1001
  metric <- "Accuracy"
  ifelse(analysis=="age_at_collection", samp_size <- ceiling(.632*nrow(x)), samp_size <- vector(mode="numeric", length= length(table(y)))+ min(table(y)))
  args_rfcv <- list("trainx"=x, "trainy"=y, "cv.fold"=N_CV_Folds, "ntree"=n_trees, "sampsize"=samp_size, step=0.9)
  if(!(analysis=="age_at_collection")){args_rfcv[["strata"]] <- y}
  parameters_rf <- do.call(rfcv, args=args_rfcv)
  args_model <- list("form"= formula(target ~.), "data"=data, "sampsize"=samp_size, "ntree"=n_trees, "ntreeTry"=as.numeric(names(parameters_rf$error.cv)[which.min(parameters_rf$error.cv)]), "importance"=T)
  if(!(analysis=="age_at_collection")){args_model[["strata"]] <- y}
  model <- do.call(randomForest, args=args_model)
  results <- rf2_results_fxn(parameters_rf$n.var, parameters_rf$error.cv, metric)
  best_tune <- list("mtry"=results[["mtry"]][which.max(results[[metric]])])
  top_performance <- max(results[,metric],na.rm=TRUE)
  saveRDS(best_tune, paste(path_compute, "hyperparameters", "_", analysis, "_", predictors, "_", algo, "_", i,".Rda", sep = ""))
  print("Results:"); print(results); print("Best tune:"); print(best_tune); print(paste("Performance using metric ", metric, " = ", round(top_performance,4), sep=""))
  return(model)
}

tune_caret_models <- function(analysis, target, predictors, algo, i, data, x, y, w)
{
  if(algo=="glmnet") {
    names_hyper <- c("lambda", "alpha")
    strides_hyper <- list("lambda"=.5, "alpha"=1)
    n_extra_leftright <- list("lambda"=2, "alpha"=2)
    n_fine_tuning <- list("lambda"=10, "alpha"=10)
    min_hyper <- list("lambda"=-Inf, "alpha"=-Inf)
    max_hyper <- list("lambda"=Inf, "alpha"=Inf)
    is_int_hyper <- list("lambda"=F, "alpha"=F)
    list_hyper <- list("lambda"=seq(-3,3,by=strides_hyper[["lambda"]]), "alpha"=c(-Inf, seq(-3,3,by = strides_hyper[["alpha"]]), Inf))
    hyper_fun <- list("lambda"=ten_power_x, "alpha"=inv.logit)
    hyper_fun_inv <- list("lambda"=log10, "alpha"=logit)
    transformed_hyper <- transform_hyper(list_hyper, hyper_fun)
  } else if (algo=="gbm") {
    names_hyper <- c("shrinkage", "interaction.depth", "n.minobsinnode")
    n_trees <- 1000
    strides_hyper <- list("shrinkage"=1, "interaction.depth"=1, "n.minobsinnode"=5)
    n_extra_leftright <- list("shrinkage"=1, "interaction.depth"=1, "n.minobsinnode"=1)
    n_fine_tuning <- list("shrinkage"=1, "interaction.depth"=1, "n.minobsinnode"=1)
    min_hyper <- list("shrinkage"=-Inf, "interaction.depth"=1, "n.minobsinnode"=1)
    max_hyper <- list("shrinkage"=Inf, "interaction.depth"=6, "n.minobsinnode"=floor(nrow(x)/3))
    is_int_hyper <- list("shrinkage"=F, "interaction.depth"=T, "n.minobsinnode"=T)
    list_hyper <- list("shrinkage"=c(-5,-4,-3), "interaction.depth"=c(1,2), "n.minobsinnode"=c(10))
    hyper_fun <- list("shrinkage"=inv.logit, "interaction.depth"=identity, "n.minobsinnode"=identity)
    hyper_fun_inv <- list("shrinkage"=logit, "interaction.depth"=identity, "n.minobsinnode"=identity)
    transformed_hyper <- transform_hyper(list_hyper, hyper_fun)
    transformed_hyper$n.trees <- n_trees
  } else if (algo=="rf") {
    names_hyper <- c("mtry")
    n_partition <- 5
    strides_hyper <- list("mtry"=ceiling(ncol(x)/n_partition))
    n_extra_leftright <- list("mtry"=1)
    n_fine_tuning <- list("mtry"=4)
    min_hyper <- list("mtry"=1)
    max_hyper <- list("mtry"=ncol(x))
    is_int_hyper <- list("mtry"=T)
    list_hyper <- list("mtry"=seq(strides_hyper[["mtry"]],strides_hyper[["mtry"]]*(n_partition-1),by=strides_hyper[["mtry"]]))
    hyper_fun <- list("mtry"=identity)
    hyper_fun_inv <- list("mtry"=identity)
    transformed_hyper <- transform_hyper(list_hyper, hyper_fun)
  } else if (algo=="svmLinear") {
    names_hyper <- c("C")
    strides_hyper <- list("C"=1)
    n_extra_leftright <- list("C"=3)
    n_fine_tuning <- list("C"=10)
    min_hyper <- list("C"=-Inf)
    max_hyper <- list("C"=Inf)
    is_int_hyper <- list("C"=F)
    list_hyper <- list("C"=seq(-7,3,by=strides_hyper[["C"]]))
    hyper_fun <- list("C"=ten_power_x)
    hyper_fun_inv <- list("C"=log10)
    transformed_hyper <- transform_hyper(list_hyper, hyper_fun)
  } else if (algo=="svmPoly") {
    names_hyper <- c("C", "scale")
    n_poly <- 2
    strides_hyper <- list("C"=1, "scale"=1)
    n_extra_leftright <- list("C"=2, "scale"=2)
    n_fine_tuning <- list("C"=3, "scale"=3)
    min_hyper <- list("C"=-Inf, "scale"=-Inf)
    max_hyper <- list("C"=Inf, "scale"=Inf)
    is_int_hyper <- list("C"=F, "scale"=F)
    list_hyper <- list("C"=seq(-2,0,by=strides_hyper[["C"]]), "scale"=c(-Inf, seq(-2,1,by = strides_hyper[["scale"]]), Inf))
    hyper_fun <- list("C"=ten_power_x, "scale"=inv.logit)
    hyper_fun_inv <- list("C"=log10, "scale"=logit)
    transformed_hyper <- transform_hyper(list_hyper, hyper_fun)
    transformed_hyper$degree <- n_poly
  } else if (algo=="svmRadial") {
    names_hyper <- c("C", "sigma")
    strides_hyper <- list("C"=1, "sigma"=1)
    n_extra_leftright <- list("C"=2, "sigma"=2)
    n_fine_tuning <- list("C"=3, "sigma"=3)
    min_hyper <- list("C"=-Inf, "sigma"=-Inf)
    max_hyper <- list("C"=Inf, "sigma"=Inf)
    is_int_hyper <- list("C"=F, "sigma"=F)
    list_hyper <- list("C"=seq(-2,0,by=strides_hyper[["C"]]), "sigma"=c(-Inf, seq(-2,1,by = strides_hyper[["sigma"]]), Inf))
    hyper_fun <- list("C"=ten_power_x, "sigma"=inv.logit)
    hyper_fun_inv <- list("C"=log10, "sigma"=logit)
    transformed_hyper <- transform_hyper(list_hyper, hyper_fun)
  } else if (algo=="knn") {
    names_hyper <- c("k")
    n_partition <- 5
    strides_hyper <- list("k"=4)
    n_extra_leftright <- list("k"=3)
    n_fine_tuning <- list("k"=3)
    min_hyper <- list("k"=1)
    max_hyper <- list("k"=ncol(x))
    is_int_hyper <- list("k"=T)
    list_hyper <- list("k"=c(1, seq(strides_hyper[["k"]],strides_hyper[["k"]]*(n_partition-1),by=strides_hyper[["k"]])))
    hyper_fun <- list("k"=identity)
    hyper_fun_inv <- list("k"=identity)
    transformed_hyper <- transform_hyper(list_hyper, hyper_fun)
  } else if (algo=="nb") {
    names_hyper <- c("adjust", "fL")
    use_kernel = c(TRUE)
    strides_hyper <- list("adjust"=1, "fL"=1)
    n_extra_leftright <- list("adjust"=2, "fL"=2)
    n_fine_tuning <- list("adjust"=2, "fL"=2)
    min_hyper <- list("adjust"=-Inf, "fL"=-Inf)
    max_hyper <- list("adjust"=Inf, "fL"=Inf)
    is_int_hyper <- list("adjust"=F, "fL"=F)
    list_hyper <- list("adjust"=seq(-1,1,by=strides_hyper[["adjust"]]), "fL"= seq(-1,1,by = strides_hyper[["fL"]]))
    hyper_fun <- list("adjust"=ten_power_x, "fL"=ten_power_x)
    hyper_fun_inv <- list("adjust"=log10, "fL"=log10)
    transformed_hyper <- transform_hyper(list_hyper, hyper_fun)
    transformed_hyper$usekernel <- use_kernel
  }
  hyper_grid <- do.call(expand.grid, args = transformed_hyper)
  hyper_grid_all <- hyper_grid
  print("starting algo")
  args_trctrl <- list(method = "cv", number = N_CV_Folds, seeds = cv_seeds)
  if(!(analysis %in% c("age_at_collection")))
  {
    args_trctrl[["classProbs"]] <- TRUE
    if(!(analysis %in% c("country"))){args_trctrl[["summaryFunction"]] <- twoClassSummary}
  }
  #parallelize the inner cross validation
  if(analysis %in% c("age_at_collection", "country") & algo %in% c("gbm", "svmPoly", "nb")){args_trctrl[["allowParallel"]] <- T}
  trctrl <- do.call(trainControl, args_trctrl)
  hyper_grid <- do.call(expand.grid, args = transformed_hyper)
  args_model <- list("form"= formula(target ~.), "data"=data, "weights"=w, "method"=algo, "trControl"=trctrl, "metric"=metric, "tuneGrid"=hyper_grid)
  if(algo=="gbm"){args_model[["verbose"]] <- FALSE}
  model <- do.call(train, args=args_model)
  results <- model$results
  best_tune <- model$bestTune
  top_performance <- max(results[,metric],na.rm=TRUE)
  print("Model:"); print(model); print("Results:"); print(results); print("Best tune:"); print(best_tune)
  #COARSE GRID SEARCH: keep shifting the grid until the best hyperparameters are not an extremum value
  print("Now starting COARSE GRID SEARCH.")
  coarse_search <- TRUE
  N_cs <- 0 #keep track of the number of iterations when looking for the absence of extremum values.
  while(coarse_search)
  {
    N_cs <- N_cs+1; print(paste("Current iteration of the COARSE SEARCH, going through every parameter: ", N_cs, ".", sep=""))
    coarse_search <- FALSE #if no extremum is detected for this "epoch", skip the while() the next time
    #for each hyperparameter, check if the best value is an extremum.
    for (hyper in names_hyper)
    {
      results_filtered <- results
      #filter the results and prepare grid: for each other hyperparameter, only consider the values that were involved in performances as good as the top performance.
      for (hyper_other in names_hyper[-which(names_hyper==hyper)])
      {
        best_hyper_other <- results_filtered[,hyper_other][which(results_filtered[,metric]==top_performance)]
        list_hyper[[hyper_other]] <- hyper_fun_inv[[hyper_other]](best_hyper_other)
        results_filtered <- results_filtered[which(results_filtered[,hyper_other] %in% best_hyper_other),]
      }
      #check if the top performance was reached for another value of the explored hyperparameter
      n_equal_best <- nrow(results_filtered[which(results_filtered[[metric]]==top_performance & !(results_filtered[[hyper]]==best_tune[[hyper]])),])
      coarse_search_hyper <- best_tune[[hyper]] %in% c(min(results_filtered[,hyper]), max(results_filtered[,hyper])) & n_equal_best == 0
      if(!coarse_search_hyper){print(paste("This hyperparameter is not (only) an extremum, so no coarse search for this coarse_search iteration: ", hyper, sep=""))}
      N_h <- 0 #keep track of the number of iterations for this hyperparameter.
      deadend <- list("min"=FALSE, "max"=FALSE) #keep track of the directions in which a limit was reached during the search.
      while(coarse_search_hyper)
      {
        N_h <- N_h+1; print(paste("Current iteration for search of HYPERPARAMETER: ", hyper, ". Iteration is: ", N_h, sep=""))
        coarse_search_hyper <- TRUE
        keep_csh <- TRUE
        if(best_tune[[hyper]] == min(results_filtered[,hyper]) & min(hyper_fun_inv[[hyper]](results_filtered[,hyper])) > min_hyper[[hyper]] & !deadend[["min"]])
        {
          print(paste("Best model selected the lowest ", hyper, " value: ", best_tune[[hyper]], sep =""))
          from_seq <- min(hyper_fun_inv[[hyper]](results_filtered[,hyper]))- strides_hyper[[hyper]]
          if(from_seq==Inf){from_seq <- 20}
          new_hyper <- seq(from=from_seq, by=-strides_hyper[[hyper]], length.out=n_extra_leftright[[hyper]])
          new_hyper[which(new_hyper < min_hyper[[hyper]])] <- min_hyper[[hyper]]
          search_direction <- "min"
        } else if (best_tune[[hyper]] == max(results_filtered[,hyper]) & max(hyper_fun_inv[[hyper]](results_filtered[,hyper])) < max_hyper[[hyper]] & !deadend[["max"]])
        {
          print(paste("Best model selected the highest ", hyper, " value: ", best_tune[[hyper]], sep =""))
          from_seq <- max(hyper_fun_inv[[hyper]](results_filtered[,hyper])) + strides_hyper[[hyper]]
          if(from_seq==-Inf){from_seq <- -20}
          new_hyper <- seq(from=from_seq, by=strides_hyper[[hyper]], length.out=n_extra_leftright[[hyper]])
          new_hyper[which(new_hyper > max_hyper[[hyper]])] <- max_hyper[[hyper]]
          search_direction <- "max"
        } else {
          coarse_search_hyper <- FALSE #to keep track of the big csh loop.
          keep_csh <- FALSE #to keep track of different steps in this iteration of the loop. Not the same as "coarse_search_hyper", because can search smaller, then bigger.
        }
        if(keep_csh)
        {
          list_hyper[[hyper]] <- unique(new_hyper)
          transformed_hyper <- transform_hyper(list_hyper, hyper_fun)
          if(algo=="gbm"){
            transformed_hyper$n.trees <- n_trees
          } else if (algo=="svmPoly") {
            transformed_hyper$degree <- n_poly
          } else if (algo=="nb") {
            transformed_hyper$usekernel <- use_kernel}
          hyper_grid <- do.call(expand.grid, args = transformed_hyper)
          hyper_grid <- hyper_grid[index_different_rows(hyper_grid, hyper_grid_all),,drop=FALSE]
          hyper_grid_all <- rbind(hyper_grid_all, hyper_grid)
          if(nrow(hyper_grid) < 1)
          {
            keep_csh <- FALSE
            print("The grid corresponding to this new value of the hyperparameter has already been explored and found to be suboptimal.")
            deadend[[search_direction]] <- TRUE
          }
        }
        if(keep_csh) #only search for other hypervalues if nrow(hyper_grid) > 0
        {
          args_model <- list("form"= formula(target ~.), "data"=data, "weights"=w, "method"=algo, "trControl"=trctrl, "metric"=metric, "tuneGrid"=hyper_grid)
          if(algo=="gbm"){args_model[["verbose"]] <- FALSE}
          error <- tryCatch(model2 <- do.call(train, args=args_model), error=function(err) "error")
          while(grepl("error", error)) #if the model aborts, set new hyperparameter limit
          {
            print("One of the new hyperparameters values returned an error. Therefore the limit for this hyperparameter has been reset.")
            if(min(hyper_grid[[hyper]]) < best_tune[[hyper]])
            {
              toremove_hyper <- min(hyper_grid[[hyper]])
              toremove_inv <- hyper_fun_inv[[hyper]](toremove_hyper)
              min_hyper[[hyper]] <- toremove_inv + strides_hyper[[hyper]]
            } else {
              toremove_hyper <- max(hyper_grid[[hyper]])
              toremove_inv <- hyper_fun_inv[[hyper]](toremove_hyper)
              min_hyper[[hyper]] <- toremove_inv - strides_hyper[[hyper]]
            }
            hyper_grid <- hyper_grid[-which(hyper_grid[[hyper]]==toremove_hyper),,drop=FALSE]
            if(nrow(hyper_grid)>0)
            {
              args_model <- list("form"= formula(target ~.), "data"=data, "weights"=w, "method"=algo, "trControl"=trctrl, "metric"=metric, "tuneGrid"=hyper_grid)
              if(algo=="gbm"){args_model[["verbose"]] <- FALSE}
              print("Re-searching after removing one (more) of the new hyperparameter values.")
              error <- tryCatch(model2 <- do.call(train, args=args_model), error=function(err) "error")
            } else {
              print("All new values have been removed. None are left to test.")
              break #break out of the "error-check" loop
            }
          } #end of error checking
          if(nrow(hyper_grid)>0)
          {
            results <- rbind(results, model2$results)
            results_filtered <- rbind(results_filtered, model2$results) #to check in next coarse_search_hyper iteration if a better value has been found.
            print(paste("More values were tried for the hyperparameter: ", hyper, ".", sep="")); print("The results for these hyperparameter values are:"); print(model2$results)
            if(max(model2$results[,metric],na.rm=TRUE) <= top_performance){
              print(paste("No better value was found for ", hyper, ".", sep =""))
            } else if (length(predict_debug(model2, data))==0) {
              print(paste("The model performed better in term of metrics, but it was unable to generate predictions, so it was discarded."))
            } else {
              model <- model2
              best_tune <- model$bestTune
              top_performance <- max(results[,metric],na.rm=TRUE)
              coarse_search <- TRUE #Keep looping the great coarse_search loop because new parameters ranges were introduced, so it is possible that another hyperparameter is now on an extremum.
              print(paste("A better hypervalue was found for the model changing the hyperparameter: ", hyper, ".", sep="")); print("Best tune:"); print(best_tune); print(paste("Performance using metric ", metric, " = ", round(top_performance,4), sep=""))
              #check if the new best hyperparameter is an extremum, of if looping can stop for this hyperparameter.
              results_filtered <- results
              for (hyper_other in names_hyper[-which(names_hyper==hyper)]) #filter the results: only consider the rows of the results for which the other hyperparameters are best tuned.
              {
                best_hyper_other <- results_filtered[,hyper_other][which(results_filtered[,hyper_other]==best_tune[[hyper_other]])]
                results_filtered <- results_filtered[which(results_filtered[,hyper_other] %in% best_hyper_other),]
              }
              n_equal_best <- nrow(results_filtered[which(results_filtered[[metric]]==top_performance & !(results_filtered[[hyper]]==best_tune[[hyper]])),])
              coarse_search_hyper <- best_tune[[hyper]] %in% c(min(results_filtered[,hyper]), max(results_filtered[,hyper])) & n_equal_best == 0
            } #end of ifelse a better model is found => update best model. otherwise go to next iteration.
          } #end of "if nrow>1"
        } else {
          print(paste("The coarse search stopped for ", hyper, " because no other values could/had to be explored in this direction.", sep =""))
        } #end of if nrows > 0 => update. Otherwise stop coarse search.
      } #end of coarse search while loop for a specific hyperparameter.
    } #end of coarse search for each hyperparameter.
  } #end of coarse search while loop
  print("END of COARSE SEARCH. Best tune is: "); print(best_tune)
  #FINE GRID SEARCH: zoom around the best hyperparameters values to fine tune the grid
  print("Now STARTING FINE GRID search.")
  for (hyper in names_hyper)
  {
    center_hyper <- hyper_fun_inv[[hyper]](best_tune[[hyper]])
    best_hypers <- hyper_fun_inv[[hyper]](results[,hyper][which(results[,metric]==top_performance)])
    min_seq <- min(best_hypers)
    max_seq <- max(best_hypers)
    #if there is a plateau of best parameters, explore it. Otherwise, look for best second value. In the other direction, explore half a stride.
    if (min_seq == max_seq)
    {
      results_filtered_without <- results[-which(results[,hyper]==best_tune[[hyper]]),]
      hyper_2nd_bests <- hyper_fun_inv[[hyper]](results_filtered_without[,hyper][which(results_filtered_without[,metric]==max(results_filtered_without[,metric],na.rm=TRUE))])
      hyper_2nd_best <- hyper_2nd_bests[which.min(abs(hyper_2nd_bests-center_hyper))]
      if(hyper_2nd_best > center_hyper)
      {
        min_seq <- max(min_hyper[[hyper]], center_hyper - strides_hyper[[hyper]]/2)
        max_seq <- hyper_2nd_best
      } else {
        min_seq <- hyper_2nd_best
        max_seq <- min(max_hyper[[hyper]], center_hyper + strides_hyper[[hyper]]/2)
      }
    }
    if(min_seq==-Inf) min_seq <- -20
    if(max_seq==Inf) max_seq <- 20
    list_hyper[[hyper]] <- seq(min_seq, max_seq, by=(max_seq-min_seq)/(n_fine_tuning[[hyper]]+1))
    if(is_int_hyper[[hyper]]){list_hyper[[hyper]] <- unique(round(list_hyper[[hyper]]))}
  }
  transformed_hyper <- transform_hyper(list_hyper, hyper_fun)
  if(algo=="gbm"){
    transformed_hyper$n.trees <- n_trees
  } else if (algo=="svmPoly") {
    transformed_hyper$degree <- n_poly
  } else if (algo=="nb"){
    transformed_hyper$usekernel <- use_kernel}
  hyper_grid <- do.call(expand.grid, args = transformed_hyper)
  hyper_grid <- hyper_grid[index_different_rows(hyper_grid, hyper_grid_all),,drop=FALSE]
  if(nrow(hyper_grid) > 0)
  {
    print("The grid values tested during the fine search are: "); print(hyper_grid)
    hyper_grid_all <- rbind(hyper_grid_all, hyper_grid)
    args_model <- list("form"= formula(target ~.), "data"=data, "weights"=w, "method"=algo, "trControl"=trctrl, "metric"=metric, "tuneGrid"=hyper_grid)
    if(algo=="gbm"){args_model[["verbose"]] <- FALSE}
    error <- tryCatch(model2 <- do.call(train, args=args_model), error=function(err) "error")
    if(grepl("error", error)) #if the model aborts, set new hyperparameter limit
    {
      print("At least one of the lines of the hyper_grid lead to a computing error. Rerunning the hypervalues combinations one by one.")
      for (j in seq(nrow(hyper_grid)))
      {
        print("Trying the following hypervalues combination: "); print(hyper_grid[j,,drop=F])
        args_model <- list("form"= formula(target ~.), "data"=data, "weights"=w, "method"=algo, "trControl"=trctrl, "metric"=metric, "tuneGrid"=hyper_grid[j,,drop=F])
        if(algo=="gbm"){args_model[["verbose"]] <- FALSE}
        error <- tryCatch(model2 <- do.call(train, args=args_model), error=function(err) "error")
        if(grepl("error", error))
        {
          print("This hypervalues combination made the model crash.")
        } else {
          results <- rbind(results, model2$results)
          print(paste("The performance for this hypervalue combination is: ", model2$results[,metric], sep=""))
          if(is.na(model2$results[,metric]) | model2$results[,metric] <= top_performance)
          {
            print("This hypervalues combination did not improve the performance.")
          } else if (length(predict_debug(model2, data))==0) {
            print(paste("The model performed better in term of metrics, but it was unable to generate predictions, so it was discarded."))
          } else {
            model <- model2
            best_tune <- hyper_grid[j,,drop=F]
            top_performance <- model2$results[,metric]
            print(paste("The performance of the model improved. Performance using metric ", metric, " = ", round(top_performance,4), sep=""))
          }
        }
      }
    } else {
      results <- rbind(results, model2$results)
      print("More values were tried during the fine search. The results for these hyperparameter values are: "); print(model2$results)
      if(max(model2$results[,metric],na.rm=TRUE) <= top_performance){
        print("No better value was found during the fine search.")
      } else if (length(predict_debug(model2, data))==0) {
        print(paste("The model performed better in term of metrics, but it was unable to generate predictions, so it was discarded."))
      } else {
        model <- model2
        best_tune <- model$bestTune
        top_performance <- max(results[,metric],na.rm=TRUE)
        print("A better hypervalue was found during the fine search."); print("Best tune:"); print(best_tune); print(paste("Performance using metric ", metric, " = ", round(top_performance,4), sep=""))
      }
    }
  } else {
    print("No better value was found during the fine search.")
  }
  saveRDS(model$bestTune, paste(path_compute, "hyperparameters", "_", analysis, "_", predictors, "_", algo, "_", i,".Rda", sep = ""))
  print("Results:"); print(results); print("Best tune:"); print(best_tune); print(paste("Performance using metric ", metric, " = ", round(top_performance,4), sep=""))
  return(model)
}

best_performances <- function()
{
  Best_performances <- initiate_store(analyses, c("Analysis", "Best Performance", "Control", "Metric", "Predictors", "Algorithm", "Algorithm-Control"))
  for (analysis in analyses)
  {
    if (analysis=="age_at_collection") {
      metric <- "R2"
    } else if (analysis %in% c("country")) {
      metric <- "Mean_Accuracy"
    } else {
      metric <- "ROC"}
    data <- readRDS(paste(path_store, "Performance", "_", "test", "_", analysis, "_", metric, ".Rda", sep = ""))
    data_sd <- readRDS(paste(path_store, "Performance_sd", "_", "test", "_", analysis, "_", metric, ".Rda", sep = ""))
    analysis_label <- dictionary_analysis[which(dictionary_analysis$name==analysis),"label"]
    metric_label <- dictionary_metric[which(dictionary_metric$name==metric),"label"]
    #best performance
    row_i <- which.max(apply(data,MARGIN=1,max,na.rm=T))
    col_j <- which.max(apply(data,MARGIN=2,max,na.rm=T))
    best_algo <- dictionary_algo[which(dictionary_algo$name==rownames(data)[row_i]),"label"]
    best_predictors <- dictionary_predictors[which(dictionary_predictors$name==names(data)[col_j]), "label"]
    best_perf <- remove_leading_zero(data[row_i, col_j], n_digits=n_digits_display[[metric]])
    best_perf_sd <- remove_leading_zero(data_sd[row_i, col_j], n_digits=n_digits_display[[metric]])
    best_perf <- paste(best_perf, best_perf_sd, sep="+-")
    #best_control
    row_i <- which.max(data$demographics)
    best_algo_demo <- dictionary_algo[which(dictionary_algo$name==rownames(data)[row_i]),"label"]
    best_perf_demo <- remove_leading_zero(data$demographics[row_i], n_digits=n_digits_display[[metric]])
    best_perf_demo_sd <- remove_leading_zero(data_sd$demographics[row_i], n_digits=n_digits_display[[metric]])
    best_perf_demo <- paste(best_perf_demo, best_perf_demo_sd, sep="+-")
    #store results
    Best_performances[which(rownames(Best_performances)==analysis),] <- c(analysis_label, best_perf, best_perf_demo, metric_label, best_predictors, best_algo, best_algo_demo)
  }
  rownames(Best_performances) <- Best_performances$Analysis
  Best_performances <- Best_performances[,which(!(names(Best_performances) %in% c("Analysis")))]
  print(Best_performances)
  saveRDS(Best_performances, paste(path_store, "Best_performances.Rda", sep = ""))
}

preprocess_data_app <- function(data)
{
  names_demographics <- c("age_at_collection", "abx_usage", "exclusive_bf", "delivery_type", "sex", "country")
  data[,seq(length(names_demographics))] <- data[,names_demographics]
  names(data)[1:9] <- names_demographics
  levels(data$abx_usage) <- c("Never", "Already")
  levels(data$exclusive_bf) <- c("No", "Yes")
  levels(data$delivery_type) <- c("Cesarean", "Vaginal")
  data$delivery_type <- factor(data$delivery_type, levels=c("Vaginal", "Cesarean"))
  levels(data$sex) <- c("Female", "Male")
  data$sex <- factor(data$sex, levels=c("Male", "Female"))
  data$country <- as.factor(data$country)
  levels(data$country) <- c("Estonia", "Finland", "Russia", "Sweden")
  data$country <- factor(data$country, levels=c("Finland", "Estonia", "Russia", "Sweden"))
  levels(data$seroconverted_time) <- c("No", "Yes")
  levels(data$seroconverted_ever) <- c("No", "Yes")
  names(data)[which(names(data) %in% c("age_at_collection", "abx_usage", "exclusive_bf", "delivery_type", "sex", "country"))] <- c("Age", "Antibiotics Usage", "Exclusively Breastfed", "Delivery Type", "Sex", "Country of Origin")
  return(data)
}

  


