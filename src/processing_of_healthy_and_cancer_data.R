library(tidyverse)
library(stringr)
source("src/functions.R")
set_working_directory()

# load and clean data
df_raw <- read.csv("data/iedb_200K_EL_epitopes.csv", 
                   sep = ";", stringsAsFactors = FALSE)
colnames(df_raw) <- clean_names(colnames(df_raw))

df_clean <- df_raw |>
  select(where(~ !all(is.na(.)))) |>
  mutate(across(where(is.character), ~ na_if(.x, ""))) |>
  mutate(pep_length = nchar(epitope_name))

disease_list <- df_clean |>
  select(in_vivo_process_disease) |>
  distinct() 

cancer_patterns <- "cancer|tumor|leukemia|carcinoma|melanoma|sarcoma|lymphoma|myeloma|glioma|glioblastoma|neuroblastoma|mesothelioma"

cancer_list <- disease_list |>
  filter(str_detect(
    in_vivo_process_disease, regex(cancer_patterns, ignore_case = TRUE)))

non_cancer_list <- disease_list |>
  filter(!str_detect(
    in_vivo_process_disease, regex(cancer_patterns, ignore_case = TRUE)))

df_healthy <- df_clean |>
  filter(in_vivo_process_disease=="healthy")

df_cancer <- df_clean |>
  filter(in_vivo_process_disease %in% cancer_list$in_vivo_process_disease)

  