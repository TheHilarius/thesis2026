library(tidyverse)
source("src/pipeline/r/functions.R")
set_working_directory()

cat("=== STEP 1: Loading Raw IEDB Data ===\n")
# Load and clean IEDB 200K EL epitopes
df_raw <- read.csv("data/raw/iedb_200K_EL_epitopes.csv",
                   sep = ";", stringsAsFactors = FALSE) # Oliver should use delimter = ","
colnames(df_raw) <- clean_names(colnames(df_raw))

cat("Raw assays (rows) loaded from IEDB: ", nrow(df_raw), "\n")
cat("Total unique peptides in raw data:  ", n_distinct(df_raw$epitope_name), "\n\n")

cat("=== STEP 2: Formatting and De-duplication ===\n")
# Initial formatting and selection
df_formatted <- df_raw |>
  select(where(~ !all(is.na(.)))) |>
  mutate(across(where(is.character), ~ na_if(.x, ""))) |>
  mutate(pep_length = nchar(epitope_name)) |>
  select(c(
    epitope_name, epitope_modifications, pep_length,
    epitope_starting_position, epitope_ending_position,
    epitope_source_molecule, epitope_molecule_parent,
    epitope_source_molecule_iri,
    epitope_molecule_parent_iri,
    in_vivo_process_process_type, in_vivo_process_disease
  )) |>
  rename_with(~ str_remove(.x, "^epitope_"), starts_with("epitope_")) |>
  rename(
    start   = starting_position,
    end     = ending_position,
    peptide = name
  ) |>
  mutate(
      # Step 1: Extract from Source
      source_uniprot_id = str_extract(source_molecule_iri, "(?<=uniprot/)[A-Z0-9]+(?:-[0-9]+)?"),
      
      # Step 2: Extract from Parent
      parent_uniprot_id = str_extract(molecule_parent_iri, "(?<=uniprot/)[A-Z0-9]+(?:-[0-9]+)?"),
      
      # Step 3: Combine them (Prioritize Source, fall back to Parent)
      uniprot_id = coalesce(source_uniprot_id, parent_uniprot_id)
  ) |>
  select(-source_molecule_iri, -molecule_parent_iri, -source_uniprot_id, -parent_uniprot_id)

# De-duplicate to unique peptide + Uniprot ID combinations
df_dedup <- df_formatted |>
  distinct(peptide, uniprot_id, .keep_all = TRUE)

cat("Rows after grouping unique peptide + UniProt ID pairs: ", nrow(df_dedup), "\n")
cat("Redundant assays (duplicates) collapsed:               ", nrow(df_formatted) - nrow(df_dedup), "\n\n")


cat("=== STEP 3: Filtering Modifications ===\n")
# Remove modified peptides (we want pure sequence processing)
df_clean <- df_dedup |>
  filter(is.na(modifications))

cat("Peptides discarded due to PTMs/modifications: ", nrow(df_dedup) - nrow(df_clean), "\n")
cat("Final cleaned positive epitopes remaining:    ", nrow(df_clean), "\n\n")

# Drop the modifications column now that it is empty
df_clean <- df_clean |> select(-modifications)


cat("=== STEP 4: Splitting by Disease State ===\n")
# Split by disease category
df_healthy <- df_clean |>
  filter(in_vivo_process_disease == "healthy")

cancer_patterns <- "cancer|tumor|leukemia|carcinoma|melanoma|sarcoma|lymphoma|myeloma|glioma|glioblastoma|neuroblastoma|mesothelioma"

cancer_list <- df_clean |>
  distinct(in_vivo_process_disease) |>
  filter(str_detect(in_vivo_process_disease, regex(cancer_patterns, ignore_case = TRUE)))

df_cancer <- df_clean |>
  filter(in_vivo_process_disease %in% cancer_list$in_vivo_process_disease)

cat("Healthy context epitopes: ", nrow(df_healthy), "\n")
cat("Cancer context epitopes:  ", nrow(df_cancer), "\n")
cat("Other/Unspecified:        ", nrow(df_clean) - (nrow(df_healthy) + nrow(df_cancer)), "\n\n")


cat("=== STEP 5: Exporting Data ===\n")
# Export
write_csv(df_clean,   "data/processed/pos_EL_all_epitopes_hla0201.csv")
write_csv(df_healthy, "data/processed/pos_EL_healthy_epitopes_hla0201.csv")
write_csv(df_cancer,  "data/processed/pos_EL_cancer_epitopes_hla0201.csv")

# Write stage counts for Sankey diagram
iedb_stage_counts <- tibble(
  stage = c("raw_assays", "duplicates", "unique_pairs", "ptm", "no_ptm"),
  count = c(nrow(df_raw), nrow(df_formatted) - nrow(df_dedup), nrow(df_dedup),
            nrow(df_dedup) - nrow(df_clean), nrow(df_clean))
)
write_csv(iedb_stage_counts, "data/processed/iedb_stage_counts.csv")
cat("Stage counts saved to data/processed/iedb_stage_counts.csv\n")

cat("✅ Saved cleaned files to data/processed/\n")
cat("Next step: run python src/fetch/fetch_uniprot_fastas.py\n")
