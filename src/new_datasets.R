library(tidyverse)
source("src/functions.R")

current_user <- Sys.info()[["user"]]

if (current_user == "olive") {
  setwd("C:/Users/olive/Documents/R/special_course_spring2026")
} else if (current_user == "mj607") {
  setwd("//wsl$/Ubuntu/home/hilarius/special_course_spring2026")
} else if (current_user == "hilarius") {
  setwd("/Users/hilarius/Desktop/DTU/special_course_spring2026")
} else if (current_user == "Hilarius") {
  setwd("C:/Users/Hilarius/OneDrive - Danmarks Tekniske Universitet/Skrivebord/special_course_spring2026/special_course_spring2026")
} else {
  stop("Unknown user. Please set working directory manually.")
}

df_raw <- read.csv("data/iedb_522_EL_epitopes.csv")

df_clean <- df_raw |>
  select(where(~ !all(is.na(.)))) |>
  mutate(
    pep_length = nchar(Epitope...Name)
  )