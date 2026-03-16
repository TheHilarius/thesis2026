library(tidyverse)
source("src/functions.R")
set_working_directory()

df_raw <- read_csv("data/processed/pos_EL_all_epitopes_hla0201.csv")