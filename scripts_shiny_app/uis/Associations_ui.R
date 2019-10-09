fluidPage(
titlePanel("Associations"),
  
  sidebarLayout(
    sidebarPanel(
      helpText("Vizualize the distribution of the different biomarkers in different demographic groups."),
      radioButtons("predictors", "Predictors", choices = c("Demographics", "Taxa", "CAGs", "Genes", "Pathways"), selected = "Demographics"),
      uiOutput("analysis"),
      uiOutput("algo"),
      htmlOutput("Biomarker1"),
      htmlOutput("Biomarker2")
      ),
    
    mainPanel(plotOutput("plot_associations"),
              tags$style(type="text/css",
                         ".shiny-output-error { visibility: hidden; }",
                         ".shiny-output-error:before { visibility: hidden; }")              
              )
  )
)
