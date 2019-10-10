#APP VERSION

source("Helpers_microbiome_app.R")
source("Helpers_microbiome.R")

ui <- fluidPage(
  navbarPage("Infants, Microbiome and Machine Learning", id = "Home",
             tabPanel("Home", value = "Home_LandingPage"
             ),
             tabPanel("Prediction performances", value = "Performances"),
             tabPanel("Best predictors", value = "Predictors"),
             tabPanel("Associations between variables", value = "Associations"),
             tabPanel("Folds tuning", value = "Hyperparameters")
  ),
  uiOutput("container"))


server = function(input, output, session) {
  output$container <- renderUI({
    folder <- gsub( "_.*$", "", input$Home)
    source(paste("servers", "/", input$Home, "_server", ".R", sep = ""), local = TRUE)
    source(paste("uis", "/", input$Home, "_ui", ".R", sep = ""), local=TRUE)$value
  })
}


shinyApp(ui, server)


