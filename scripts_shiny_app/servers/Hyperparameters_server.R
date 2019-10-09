analysis <- reactive({dictionary_analysis[dictionary_analysis$label==input$analysis,"name"]})

predictors <- reactive({dictionary_predictors[dictionary_predictors$label==input$predictors,"name"]})

algo <- reactive({dictionary_algo[dictionary_algo$label==input$algo,"name"]})


output$algos <- renderUI({
  if (analysis()=="Surv") {
    algos_list <- algos_surv
  } else if (predictors() %in% c("genes", "demo+genes")) {
    algos_list <- algos_genes
  } else if(analysis()=="age_at_collection") {
    algos_list <- algos_regression
  } else {
    algos_list <- algos }
  algos_list <- dictionary_algo$label[which(dictionary_algo$name %in% algos_list)]
  selectInput("algo", label = "Select the algorithm used for prediction:", choices = c(Choose = '', algos_list), selectize = TRUE, selected=algos_list[1])
})

output$display_performances_CI <- renderUI({
  if(input$display_performances){checkboxInput("display_performances_CI", "Include the confidence intervals for the performances", value = F)}
})

output$help_text_list_performances <- renderUI({
  if(input$display_performances){HTML("<b>Display the following performance metrics:</b>")}
})

output$list_performances <- renderUI({
  if(input$display_performances)
  {
    lapply(1:length(metricsS[[analysis()]]), function(i) {
      metric_i <- metricsS[[analysis()]][i]
      div(style="display:inline-block", checkboxInput(metric_i, dictionary_metric[dictionary_metric$name==metric_i,"name"], value=display_performances[[metric_i]]))
    })
  }
})

output$table_Hyperparameters <- renderDataTable({
  data <- readRDS(paste(path, "perf_ss_hyper", "_", analysis(), "_", predictors(), "_", algo(), ".Rda", sep = ""))
  to_format <- names(data)[unique(grep("R2|CI|ROC|Accuracy|Sensitivity|Specificity|alpha|sigma", names(data)))]
  data[,to_format] <- sapply(data[,to_format], remove_leading_zero, n_digits=3)
  to_format <- names(data)[unique(grep("Cross_Entropy", names(data)))]
  data[,to_format] <- sapply(data[,to_format], remove_leading_zero, n_digits=5)
  to_format <- names(data)[unique(grep("lambda|shrinkage|C|fL|adjust", names(data)))]
  data[,to_format] <- sapply(data[,to_format], formatC, format = "e", digits = 2)
  rownames(data) <- c("Folds average", "All samples", seq(N_CV_Folds))
  if(!input$display_performances){data <- data[,unique(which(!(grepl("R2|CI|ROC|Cross_Entropy|Accuracy|Sensitivity|Specificity", names(data))))), drop=F]}
  if(!input$display_performances_CI){data <- data[,unique(which(!(grepl("_sd", names(data))))),drop=F]}
  if(!input$display_sample_sizes){data <- data[,unique(which(!(grepl("N", names(data))))),drop=F]}
  if(!input$display_hyperparameters){data <- data[,which(!(names(data) %in% names_hyperparameters)),drop=F]}
  if(input$display_performances){for (metric in metricsS[[analysis()]]){if(!input[[metric]]){data <- data[,which(!(gsub("_sd|_train|_test", "", names(data))==metric))]}}}
  if(input$display_performances | input$display_sample_sizes)
  {
    if(input$set=="Training"){
      data <- data[,which(!(grepl("_test", names(data)))),drop=F]
      names(data) <- gsub("_train", "", names(data))
    } else if (input$set=="Testing") {
      data <- data[,which(!(grepl("_train", names(data)))),drop=F]
      names(data) <- gsub("_test", "", names(data))
    }
  }
  data
}, rownames = TRUE, options = list(pageLength = N_CV_Folds+2, lengthChange = FALSE, dom = "t"))

