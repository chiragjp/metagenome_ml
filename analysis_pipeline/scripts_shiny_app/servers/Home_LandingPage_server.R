output$table_home <- renderDataTable({
  data <- readRDS(paste(path, "Best_performances.Rda", sep = ""))
  data <- data[which(rownames(data) %in% analyses_labels),]   #modify table to take the relevant subset of phenotypes
  data
}, rownames = TRUE, options = list(pageLength = length(analyses), lengthChange = FALSE, dom = "t"))

output$text_home <- renderText({
  "<h4> Best performances </h4> The table above summarizes the best performances in term of prediction. For each phenotype, the value of the highest prediction accuracy that we obtained is reported (Best Performance), along the highest prediction accuracy that we obtained using only the demographics variables as predictors (Control). The metric used to evaluate the performance is also specified, as well as the type of predictors and the algorithm that yielded the best prediction on the testing set. The algorithm that performed the best using the control predictors is also reported. <br> <br> <br>"
})

