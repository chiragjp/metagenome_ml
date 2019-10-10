fluidPage(
  titlePanel("Best predictors"),
  sidebarLayout(
    sidebarPanel(
      helpText("Explore the list of the most important predictors for each phenotype, depending on the algorithm (elastic net, gradient boosted machine, random forest)."),
      selectInput("analysis", label = "Select the predicted phenotype:", choices = c(Choose = '', analyses_labels), selectize = TRUE, selected=analyses_labels[1]),
      selectInput("predictors", label = "Select the predictors:", choices = c(Choose = '', predictors_labels), selectize = TRUE, selected="Mixed Predictors + Demographics"),
      uiOutput("algos"),
      selectInput("fold", label = "Select the cross validation fold:", choices = c(Choose = '', folds_labels), selectize = TRUE, selected = "All samples"),
      uiOutput("list_cols_text"),
      uiOutput("list_cols"),
      uiOutput("col_order"),
      uiOutput("abs_reg_coefs"),
      uiOutput("n_top_predictors")
    ),
    
    mainPanel(
      dataTableOutput("table_best_predictors"),
      dataTableOutput("table_predictors"),
      tags$style(type="text/css",
                 ".shiny-output-error { visibility: hidden; }",
                 ".shiny-output-error:before { visibility: hidden; }")
      )
  )
)

