analysis <- reactive({dictionary_analysis[dictionary_analysis$label==input$analysis,"name"]})

predictors <- reactive({dictionary_predictors[dictionary_predictors$label==input$predictors,"name"]})

algo <- reactive({dictionary_algo[dictionary_algo$label==input$algo,"name"]})

fold <- reactive({dictionary_fold[dictionary_fold$label==input$fold,"name"]})

output$algos <- renderUI({
  if (analysis()=="Surv") {
    algos_list <- algos_surv
  } else {
    algos_list <- algos_genes}
  algos_list <- dictionary_algo$label[which(dictionary_algo$name %in% algos_list)]
  selectInput("algo", label = "Select the algorithm used for prediction:", choices = c(Choose = '', algos_list), selectize = TRUE, selected=algos_list[1])
})

output$list_cols_text <- renderUI({
  if(ncol(data())>1)
  {
    HTML("<b>Select the metrics to be displayed:</b>")
  }
})

output$list_cols <- renderUI({
  if(ncol(data())>1)
  {
    lapply(1:ncol(data()), function(i) {
      div(style="display:inline-block", checkboxInput(names(data())[i], names(data())[i], value=T))
    })
  }
})

output$col_order <- renderUI({
  if(ncol(data2())>1){radioButtons("col_order", "Select the metric to order the predictors:", choices = names(data2()), selected = names(data2())[1])}
})

output$abs_reg_coefs <- renderUI({
  if(algo() %in% c("glmnet", "glmnet2")){checkboxInput("abs_reg_coefs", "Order by absolute values", value=T)}
})

output$n_top_predictors <- renderUI({
  if(predictors() %in% c("mixed", "demo+mixed")){sliderInput("n_top_predictors", "Select N to classify the top N predictors by type", min = 1, max = nrow(data2()), value = min(100, nrow(data2())))}
})

data <- reactive({
  data <- readRDS(paste(path, "variables_importance", "_", analysis(), "_", predictors(), "_", algo(), "_",  fold(), ".Rda", sep = ""))
  data
})

data2 <- reactive({
  data2 <- data()
  names_col <- names(data2)
  if(ncol(data2)>1){for (name_col in names_col){if(!input[[name_col]]){data2 <- data2[,which(!(names(data2)==name_col)),drop=F]}}}
  data2
})


output$table_best_predictors <- renderDataTable({
  if(predictors() %in% c("mixed", "demo+mixed"))
  {
    data <- data2()
    data <- data.frame(data)
    ifelse(ncol(data)>1, index_col <- input$col_order, index_col <- names(data))
    ifelse(algo() %in% c("glmnet", "glmnet2") & input$abs_reg_coefs, index_pred <- order(abs(data[[index_col]]),decreasing = T), index_pred <- order(data[[index_col]],decreasing = T))
    data <- data[index_pred[1:input$n_top_predictors],index_col,drop=F]
    data$type <- sapply(rownames(data), predictor_type)
    ifelse(predictors()=="demo+mixed", pred_types <- predictors_types, pred_types <- predictors_types[-1])
    #pred_types <- predictors_types
    data$type <- factor(data$type, levels=pred_types)
    tp_n <- as.vector(table(data$type))
    tp_p <- formatC(tp_n*100/sum(tp_n), format="f", digits=1)
    tp_wn <- aggregate(abs(data[[index_col]]), by=list(Category=data$type), FUN=sum, drop=F)
    rownames(tp_wn) <- tp_wn$Category
    tp_wn <- tp_wn[pred_types,"x"]
    tp_wn[is.na(tp_wn)] <- 0
    tp_wp <- formatC(tp_wn*100/sum(tp_wn), format="f", digits=1)
    top_preds <- data.frame(rbind(tp_n, tp_p, tp_wp))
    names(top_preds) <- pred_types
    rownames(top_preds) <- c("Number of best predictors", "Percentage of best predictors", "Weighted percentage of best predictors")
    top_preds
  }
}, rownames = TRUE, options = list(lengthChange = FALSE, dom = "t"), caption="Distribution of the best predictors between the different predictors categories")

output$table_predictors <- renderDataTable({
  data <- data2()
  ifelse(ncol(data)>1, index_col <- input$col_order, index_col <- names(data))
  ifelse(algo() %in% c("glmnet", "glmnet2") & input$abs_reg_coefs, index_pred <- order(abs(data[[index_col]]),decreasing = T), index_pred <- order(data[[index_col]],decreasing = T))
  data <- data[index_pred,,drop=F]
  data[,] <- sapply(data, formatC, format = "e", digits = 2)
  data
}, rownames = TRUE, options = list(pageLength = 100), caption="Ordered list of the most important predictors")

