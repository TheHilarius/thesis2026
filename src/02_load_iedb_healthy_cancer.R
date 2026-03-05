library(tidyverse)
source("src/functions.R")
set_working_directory()

# Load and clean IEDB 200K EL epitopes
df_raw <- read.csv("data/raw/iedb_200K_EL_epitopes.csv",
                   sep = ";", stringsAsFactors = FALSE)
colnames(df_raw) <- clean_names(colnames(df_raw))

df_clean <- df_raw |>
  select(where(~ !all(is.na(.)))) |>
  mutate(across(where(is.character), ~ na_if(.x, ""))) |>
  mutate(pep_length = nchar(epitope_name)) |>
  select(c(
    epitope_name, epitope_modifications, pep_length,
    epitope_starting_position, epitope_ending_position,
    epitope_source_molecule, epitope_molecule_parent, epitope_molecule_parent_iri,
    in_vivo_process_process_type, in_vivo_process_disease
  )) |>
  rename_with(~ str_remove(.x, "^epitope_"), starts_with("epitope_")) |>
  rename(
    start  = starting_position,
    end    = ending_position,
    peptide = name
  ) |>
  mutate(uniprot_id = str_extract(molecule_parent_iri, "[^/]+$")) |>
  select(-molecule_parent_iri) |>
  distinct(peptide, uniprot_id, .keep_all = TRUE) |>
  filter(is.na(modifications)) |>
  select(-modifications)

# Split by disease category
df_healthy <- df_clean |>
  filter(in_vivo_process_disease == "healthy")

cancer_patterns <- "cancer|tumor|leukemia|carcinoma|melanoma|sarcoma|lymphoma|myeloma|glioma|glioblastoma|neuroblastoma|mesothelioma"

cancer_list <- df_clean |>
  distinct(in_vivo_process_disease) |>
  filter(str_detect(in_vivo_process_disease, regex(cancer_patterns, ignore_case = TRUE)))

df_cancer <- df_clean |>
  filter(in_vivo_process_disease %in% cancer_list$in_vivo_process_disease)

# Export
write_csv(df_clean,   "data/processed/pos_EL_all_epitopes_hla0201.csv")
write_csv(df_healthy, "data/processed/pos_EL_healthy_epitopes_hla0201.csv")
write_csv(df_cancer,  "data/processed/pos_EL_cancer_epitopes_hla0201.csv")

# Next step: run src/export_fasta_files_from_iedb.py on data/processed/pos_EL_all_epitopes_hla0201.csv