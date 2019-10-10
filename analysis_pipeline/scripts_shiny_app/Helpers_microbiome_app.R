#Helpers for apps

set.seed(0)
path <- "data/"

app_version = "other phenotypes"
uploaded_app = TRUE

#libraries
# library(rsconnect)
# options(rsconnect.max.bundle.size=3145728000)
# options(rsconnect.max.bundle.files=200000)
# options(warn=1)
library(shiny)
library(shinydashboard)
library(shinyBS)
library(shinyLP)
library(DT)
Sys.setenv(RGL_USE_NULL = TRUE)
library(ggplot2)

#only load if on computer or O2, not on Shiny app
# if(!(uploaded_app))
# {
#   library(rgl)
#   library(plotly)
#   library(metafor)
#   library(shinyWidgets)
#   library(reshape2)
#   library(pvclust)
#   library(caTools)
#   library(plotrix)
#   library(openxlsx)
#   library(weights)
# }

folder = "Preprocessing"
#figures parameters
generate_figures = TRUE
size_dots = 5
size_lines = 1
size_bars = 1
size_titles = 15
size_axis = 30
size_ticks = 20
size_titles_flatheatmaps <- 20
size_axis_flatheatmaps <- 10
size_ticks_flatheatmaps_x <- 5
size_ticks_flatheatmaps_y <- 6


#variables
#decompress data if not already done (makes it easier to upload on the server)
if(!dir.exists("data"))
{
  untar('data.tar.gz')
  file.remove('data.tar.gz')
  message("done")
}

names_taxa <- readRDS(paste(path, "names_taxa.Rda", sep=""))
names_cags <- readRDS(paste(path, "names_cags.Rda", sep=""))
names_pathways <- readRDS(paste(path, "names_pathways.Rda", sep=""))

if(generate_figures)
{
  size_axis = 30
  size_labels_heatmaps = 8
  
} else {
  size_axis = 25
  size_labels_heatmaps = 5
}

#loading functions
file_name <- function(name = "", analysis = "", age_group = "", demographic_group = "", half = "", side = "", other_arg1 = "", other_arg2 = "", other_arg3 = "", other_arg4 = "")
{
  args = c(name, analysis, age_group, demographic_group, half, side, other_arg1, other_arg2, other_arg3, other_arg4)
  args <- args[which(args != "")]
  file_name <- args[1]
  for (arg in args[-1])
  {
    file_name <- paste(file_name, arg, sep = "_")
  }
  return(file_name)
}

redRDS <- function(folder = "", name = "", analysis = "", age_group = "", demographic_group = "", half = "", side = "", other_arg1 = "", other_arg2 = "", other_arg3 = "", other_arg4 = "")
{
  return(readRDS(paste("data/max_data/", folder, "/", file_name(name, analysis, age_group, demographic_group, half, side, other_arg1, other_arg2, other_arg3, other_arg4), ".Rda", sep = "")))
}

redRDS_equalbins <- function(folder = "", name = "", analysis = "", age_group = "", demographic_group = "", half = "", side = "", other_arg1 = "", other_arg2 = "", other_arg3 = "", other_arg4 = "")
{
  return(readRDS(paste("data/equal_bins/", folder, "/", file_name(name, analysis, age_group, demographic_group, half, side, other_arg1, other_arg2, other_arg3, other_arg4), ".Rda", sep = "")))
}

savtiff <- function(p, name = "", analysis = "", age_group = "", demographic_group = "", half = "", side = "", other_arg1 = "", other_arg2 = "", other_arg3 = "", other_arg4 = "")
{
  tiff(paste("figures/", file_name(name, analysis, age_group, demographic_group, half, side, other_arg1, other_arg2, other_arg3, other_arg4), ".tiff", sep = ""))
  p
  dev.off()
}

#initiate storing matrices
initiate_store <- function(rows, columns)
{
  data <- data.frame(matrix(0, length(rows), length(columns)))
  rownames(data) <- rows
  names(data) <- columns
  return(data)
}

#function to convert the matrix of coefficients into a vector
concatenate_coefficients <- function(coefs)
{
  concatenated_coefficients <- c()
  for (i in seq(length(biomarkers_hc)))
  {
    concatenated_coefficients <- c(concatenated_coefficients, coefs[-i,i])
  }
  return(concatenated_coefficients)
}

jumbotron_size <- function(header , content, size_header=1, buttonID, button = TRUE, ...)
{
  if(!(size_header%%1==0)){print("size is the html size of the headar and must be an integer. 1 is the tallest.")}
  button_label = c(...)
  if (button){
    HTML(paste("<div class='jumbotron'>
                <h", size_header, "> ", header, " </h", size_header, ">
                <p> ", content ," </p> ",
                "<p><a class='btn btn-primary btn-lg' button id=", buttonID,"> ", button_label, " </a></p>
                </div>", sep = "") )
  } else {
    HTML(paste("<div class='jumbotron'>
                <h", size_header, "> ", header, " </h", size_header, ">
                <p> ", content ," </p> ",
                "</div>", sep = ""))
  }
}
