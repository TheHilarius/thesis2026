# ==============================================================================
# STEP 02 (a0101): Load + clean IEDB HLA-A*01:01 ligand-presentation export
#
# Edition of legacy src/pipeline/r/02_load_iedb_data.R for the external
# validation allele. Differences from legacy:
#   - Input:  data/raw/mhc_ligand_table_export_HLA-A0101.csv (comma-delimited;
#             55,183 rows, 100% HLA-A*01:01 / ligand presentation / linear
#             peptide — verified 2026-11-11)
#   - Output: data/processed/hla_a0101/pos_EL_all_epitopes_hla_a0101.csv
#             + stage counts in the same folder
# The legacy 02 script and its outputs are untouched.
# ==============================================================================
library(tidyverse)
source("src/pipeline_a0101/r/functions.R")
set_working_directory()
ensure_a0101_dirs()

cat("=== STEP 1: Loading Raw IEDB Data (HLA-A*01:01) ===\n")
df_raw <- read.csv("data/raw/mhc_ligand_table_export_HLA-A0101.csv",
                   sep = ",", stringsAsFactors = FALSE)
colnames(df_raw) <- clean_names(colnames(df_raw))

cat("Raw assays (rows) loaded from IEDB: ", nrow(df_raw), "\n")
cat("Total unique peptides in raw data:  ", n_distinct(df_raw$epitope_name), "\n\n")

cat("=== STEP 2: Formatting and De-duplication ===\n")
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
    source_uniprot_id = str_extract(source_molecule_iri, "(?<=uniprot/)[A-Z0-9]+(?:-[0-9]+)?"),
    parent_uniprot_id = str_extract(molecule_parent_iri, "(?<=uniprot/)[A-Z0-9]+(?:-[0-9]+)?"),
    uniprot_id = coalesce(source_uniprot_id, parent_uniprot_id)
  ) |>
  select(-source_molecule_iri, -molecule_parent_iri, -source_uniprot_id, -parent_uniprot_id)

df_dedup <- df_formatted |>
  distinct(peptide, uniprot_id, .keep_all = TRUE)

cat("Rows after grouping unique peptide + UniProt ID pairs: ", nrow(df_dedup), "\n")
cat("Redundant assays (duplicates) collapsed:               ", nrow(df_formatted) - nrow(df_dedup), "\n\n")

cat("=== STEP 3: Filtering Modifications ===\n")
df_clean <- df_dedup |>
  filter(is.na(modifications))

cat("Peptides discarded due to PTMs/modifications: ", nrow(df_dedup) - nrow(df_clean), "\n")
cat("Final cleaned positive epitopes remaining:    ", nrow(df_clean), "\n\n")

df_clean <- df_clean |> select(-modifications)

cat("=== STEP 4: Exporting Data ===\n")
write_csv(df_clean, pp("pos_EL_all_epitopes_hla_a0101.csv"))

iedb_stage_counts <- tibble(
  stage = c("raw_assays", "duplicates", "unique_pairs", "ptm", "no_ptm"),
  count = c(nrow(df_raw), nrow(df_formatted) - nrow(df_dedup), nrow(df_dedup),
            nrow(df_dedup) - nrow(df_clean), nrow(df_clean))
)
write_csv(iedb_stage_counts, pp("iedb_stage_counts.csv"))
cat("Stage counts saved to ", pp("iedb_stage_counts.csv"), "\n", sep = "")

cat("✅ Saved cleaned files to ", A0101_DIR, "\n", sep = "")
cat("Next step: python src/pipeline_a0101/fetch/fetch_a0101_fastas.py\n")
