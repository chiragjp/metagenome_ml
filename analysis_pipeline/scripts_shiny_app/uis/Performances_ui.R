fluidPage(
  titlePanel("Prediction performances"),
  sidebarLayout(
    sidebarPanel(
      helpText("Explore the prediction performances."),
      selectInput("analysis", label = "Select the predicted phenotype", choices = c(Choose = '', analyses_labels), selectize = TRUE, selected=analyses_labels[1]),
      selectInput("set", label = "Select training or testing", choices = sets_labels, selectize = TRUE, selected="Testing Set"),
      selectInput("file_name", label = "Display the performances with or without confidence interval", choices = Performances_types_labels, selectize = TRUE, selected="Performance +- Confidence Interval"),
      uiOutput("list_metrics"),
      radioButtons("group_by", "Group barplots by:", choices = c("Predictors", "Algorithms"), selected = "Predictors")
    ),
    
    mainPanel(htmlOutput("text_performances"),
              plotOutput("plot_performances"),
              dataTableOutput("table_performances"),
              tags$style(type="text/css",
                         ".shiny-output-error { visibility: hidden; }",
                         ".shiny-output-error:before { visibility: hidden; }")
              )
  )
)

