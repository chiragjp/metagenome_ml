analysis <- reactive({dictionary_analysis[dictionary_analysis$label==input$analysis,"name"]})

predictors <- reactive({dictionary_predictors[dictionary_predictors$label==input$predictors,"name"]})

algo <- reactive({dictionary_algo[dictionary_algo$label==input$algo,"name"]})

biomarker1 <- reactive({input$Biomarker1})

biomarker2 <- reactive({input$Biomarker2})

output$analysis <- renderUI({if(predictors() == "genes"){radioButtons("analysis", "Phenotype", choices = analyses_labels)}})

output$algo <- renderUI({if(predictors() == "genes"){radioButtons("algo", "Algorithm", choices = algos_genes_labels)}})

output$Biomarker1 <- renderUI({
  selectInput(
    inputId = "Biomarker1", 
    label = "Select the 1st variable",
    choices = colnames(data()),
    selected = NULL)
})

output$Biomarker2 <- renderUI({
  selectInput(
    inputId = "Biomarker2", 
    label = "Select the 2nd variable",
    choices = colnames(data())[-which(colnames(data()) == input$Biomarker1)],
    selected = NULL)
})

data <- reactive({
  predictors <- predictors()
  #load data
  if(predictors=="genes")
  {
    data <- readRDS(paste(path, "data_app_associations_genes_", analysis(), "_", algo(), ".Rda", sep=""))
  } else {
    data <- readRDS(paste(path, "data_app_associations_", predictors, ".Rda", sep=""))
  }
  #get rid of diabetes variables
  data <- data[, which(!(names(data) %in% variables_to_remove_associations))]
  data
})

output$plot_associations <- renderPlot({
  data <- data()
  biomarker1 <- biomarker1()
  biomarker2 <- biomarker2()
  data <- data.frame("biomarker1"=data[[biomarker1]], "biomarker2"=data[[biomarker2]])
  n_factors <- (class(data$biomarker1)=="factor") + (class(data$biomarker2)=="factor")
  if (n_factors == 0)
  {
    cor <- cor(data$biomarker1, data$biomarker2)
    model <- lm(biomarker2 ~ biomarker1, data)
    coef <- summary(model)$coefficients["biomarker1", "Estimate"]
    pv <- summary(model)$coefficients["biomarker1", "Pr(>|t|)"]
    intercept <- summary(model)$coefficients["(Intercept)", "Estimate"]
    x_name <- input$Biomarker1
    y_name <- input$Biomarker2
    if(x_name %in% dictionary_analysis$name){x_name <- dictionary_analysis$label[which(dictionary_analysis$name==x_name)]}
    if(y_name %in% dictionary_analysis$name){y_name <- dictionary_analysis$label[which(dictionary_analysis$name==y_name)]}
    title_p <- paste(x_name, " VS ", y_name, ", Correlation = ", round(cor,3), ", Coefficient = ", formatC(coef, format = "e", digits = 1), ", p-value = ", formatC(pv, format = "e", digits = 1), sep = "")
    p <- ggplot(data = data, aes(biomarker1, biomarker2)) + geom_point(size = size_dots, alpha=.5) + geom_abline(intercept = intercept, slope = coef, color="red", size=size_lines) + ggtitle(title_p) + xlab(input$Biomarker1) + ylab(input$Biomarker2) + theme(plot.title = element_text(size=size_titles), axis.title.x = element_text(size = size_axis), axis.title.y = element_text(size = size_axis), axis.text.x = element_text(size=size_ticks), axis.text.y = element_text(size=size_ticks))
  } else if (n_factors == 2){
    bartable = table(data$biomarker2, data$biomarker1)  ## get the cross tab
    pv <- chisq.test(bartable)$p.value
    x_name <- input$Biomarker1
    y_name <- input$Biomarker2
    title_p <- paste(x_name, " VS ", y_name, ", p-value = ", formatC(pv, format = "e", digits = 1), sep = "")
    p <- barplot(bartable, beside = TRUE, names.arg = levels(unique(data$biomarker1)), legend = levels(unique(data$biomarker2)), main = title_p)
  } else {
    x_name <- input$Biomarker1
    y_name <- input$Biomarker2
    if(!(class(data$biomarker1)=="factor"))
    {
      names(data) <- c("biomarker2", "biomarker1")
      x_name <- input$Biomarker2
      y_name <- input$Biomarker1
    }
    x <- data[["biomarker1"]]
    y <- data[["biomarker2"]]
    ifelse(length(table(x)) > 2, pv <- (anova(lm(y~x))$"Pr(>F)")[1], pv <- (t.test(y~x))$p.value)
    title_p <- paste(x_name, " VS ", y_name, ", p-value = ", formatC(pv, format = "e", digits = 1), sep = "")
    p <- boxplot(biomarker2~biomarker1,data=data, main=title_p, xlab=x_name, ylab=y_name)
  }
  p
})



