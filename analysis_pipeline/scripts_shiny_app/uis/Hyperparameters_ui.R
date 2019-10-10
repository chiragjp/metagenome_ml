fluidPage(
  titlePanel("Folds: Hyperparameters, Sample Sizes and Prediction Performances"),
  sidebarLayout(
    sidebarPanel(
      helpText("Explore the hyperparameters values, the performances and the sample sizes for each model."),
      selectInput("analysis", label = "Select the predicted phenotype:", choices = c(Choose = '', analyses_labels), selectize = TRUE, selected=analyses_labels[1]),
      selectInput("predictors", label = "Select the predictors:", choices = c(Choose = '', predictors_labels), selectize = TRUE, selected="Mixed Predictors + Demographics"),
      uiOutput("algos"),
      radioButtons("set", "Display the results for the datasets:", choices = c("Training and Testing", "Training", "Testing"), selected = "Training and Testing"),
      checkboxInput("display_performances", "Display the performances", value = T),
      uiOutput("display_performances_CI"),
      checkboxInput("display_sample_sizes", "Display the sample sizes", value = T),
      checkboxInput("display_hyperparameters", "Display the hyperparameters values", value = T),
      uiOutput("help_text_list_performances"),
      uiOutput("list_performances")
    ),
    
    mainPanel(dataTableOutput("table_Hyperparameters"),
              tags$style(type="text/css",
                         ".shiny-output-error { visibility: hidden; }",
                         ".shiny-output-error:before { visibility: hidden; }"))
  )
)
