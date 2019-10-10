analysis <- reactive({dictionary_analysis[dictionary_analysis$label==input$analysis,"name"]})

metric <- reactive({dictionary_metric[dictionary_metric$label==input$metric,"name"]})

set <- reactive({dictionary_set[dictionary_set$label==input$set,"name"]})

file_name <- reactive({dictionary_file_name[dictionary_file_name$label==input$file_name,"name"]})

best_performance <- reactive({
  data <- data_num()
  if(metric()=="Cross_Entropy") {
    name_row <- rownames(data)[which.max(apply(data,MARGIN=1,min,na.rm=T))]
    name_col <- names(data)[which.max(apply(data,MARGIN=2,min,na.rm=T))]
    best_perf <- remove_leading_zero(data[which(rownames(data)==name_row), name_col], n_digits=n_digits_display[[metric()]]) 
  } else {
    name_row <- rownames(data)[which.max(apply(data,MARGIN=1,max,na.rm=T))]
    name_col <- names(data)[which.max(apply(data,MARGIN=2,max,na.rm=T))]
    best_perf <- remove_leading_zero(data[which(rownames(data)==name_row), name_col], n_digits=n_digits_display[[metric()]])
  }
  list("name_row"=name_row, "name_col"=name_col, "best_perf"= best_perf)
})

data_num <- reactive({
  data_num <- readRDS(paste(path, "Performance", "_", set(), "_", analysis(), "_", metric(), ".Rda", sep = ""))
  rownames(data_num) <- dictionary_algo$label[which(dictionary_algo$name %in% rownames(data_num))]
  names(data_num) <- predictors_labels
  data_num
})

data <- reactive({
  print(input$metric)
  print(metric())
  print(dictionary_metric)
  if(file_name()=="Performance")
  {
    data <- readRDS(paste(path, "Performance", "_", set(), "_", analysis(), "_", metric(), ".Rda", sep = ""))
    data <- sapply(data, remove_leading_zero, n_digits=n_digits_display[[metric()]])
  } else {
    data <- readRDS(paste(path, "Performance", "_", set(), "_", analysis(), "_", metric(), ".Rda", sep = ""))
    data_sd <- readRDS(paste(path, "Performance_sd", "_", set(), "_", analysis(), "_", metric(), ".Rda", sep = ""))
    data <- sapply(data, remove_leading_zero, n_digits=n_digits_display[[metric()]])
    data_sd <- sapply(data_sd, remove_leading_zero, n_digits=n_digits_display[[metric()]])
    data <- matrix(paste(data, data_sd, sep="+-"), nrow=nrow(data))
  }
  data <- data.frame(data)
  rownames(data) <- rownames(data_num())
  names(data) <- predictors_labels
  data
})

output$list_metrics <- renderUI({
  selectInput(
    inputId = "metric",
    label = "Select the metric",
    choices = metricsS_labels[[analysis()]],
    selected = NULL)
})

output$text_performances <- renderText({
  paste("<h4> Best performance </h4> Best performance for <b>", input$analysis, "</b> prediction using the ", input$metric, " metric was <b>", best_performance()[["best_perf"]], "</b> and was obtained using the <b>", best_performance()[["name_col"]], "</b> predictors, and the <b>", best_performance()[["name_row"]], "</b> algorithm. <br> <br> ", sep="")
})

output$plot_performances <- renderPlot({
  data <- data_num()
  data[data < 0] <- 0
  data_stack <- stack(data)
  data_stack$rows <- rep(rownames(data), times = ncol(data))
  data_stack <- data_stack[,c(3,2,1)]
  names(data_stack) <- c("Algorithms", "Predictors", "Values")
  levels_algos <- rownames(data)
  data_stack$Algorithms <- factor(data_stack$Algorithms, levels = levels_algos)
  data_stack$Predictors <- factor(data_stack$Predictors, levels = predictors_labels)
  print(head(data_stack))
  if(input$group_by=="Predictors"){
    levels(data_stack$Predictors) <- gsub(" ", "\n", levels(data_stack$Predictors))
    p <- ggplot(data_stack, aes(fill=Algorithms, y=Values, x=Predictors))
  } else {
    levels(data_stack$Algorithms) <- gsub(" ", "\n", levels(data_stack$Algorithms))
    p <- ggplot(data_stack, aes(fill=Predictors, y=Values, x=Algorithms))
  }
  p <- p + geom_bar(position="dodge", stat="identity") + ggtitle("Comparison of the prediction performances between the different predictors and algorithms") + ylab(input$metric) + theme(plot.title = element_text(size=20), axis.title.x = element_text(size = 18), axis.title.y = element_text(size = 18), axis.text.x = element_text(size=15), axis.text.y = element_text(size=15), legend.position="bottom")
  p
})

output$table_performances <- renderDataTable({
  data <- data()
  data
}, rownames = TRUE, options = list(pageLength = length(algos), lengthChange = FALSE, dom = "t"))

