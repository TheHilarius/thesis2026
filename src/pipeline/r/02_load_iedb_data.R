  library(tidyverse)
  source("src/pipeline/r/functions.R")
  set_working_directory()

  # ── Allele switch (required arg, no silent default) ──────────────────────
  # Usage: Rscript src/pipeline/r/02_load_iedb_data.R [hla0201|hla_a0101]
  # No arg = hla0201 (legacy call preserved byte-identical).
  # Cleaning logic below untouched — same exact code per allele.
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 0) {
    ALLELE_TAG <- "hla0201"
  } else {
    ALLELE_TAG <- match.arg(args[1], c("hla0201", "hla_a0101"))
  }

  IEDB_RAW <- if (ALLELE_TAG == "hla0201")
    "data/raw/iedb_200K_EL_epitopes.csv"
  else
    "data/raw/mhc_ligand_table_export_HLA-A0101.csv"

  OUT_POS <- if (ALLELE_TAG == "hla0201")
    "data/processed/pos_EL_all_epitopes_hla0201.csv"
  else
    "data/processed/hla_a0101/pos_EL_all_epitopes_hla_a0101.csv"

  OUT_STAGES <- if (ALLELE_TAG == "hla0201")
    "data/processed/iedb_stage_counts.csv"
  else
    "data/processed/hla_a0101/iedb_stage_counts.csv"

  if (ALLELE_TAG == "hla_a0101") dir.create("data/processed/hla_a0101", recursive = TRUE, showWarnings = FALSE)

  cat("=== Allele:", ALLELE_TAG, "===\n")
  cat("=== STEP 1: Loading Raw IEDB Data ===\n")
  # Load and clean IEDB 200K EL epitopes
  df_raw <- read.csv(IEDB_RAW,
                     sep = ",", stringsAsFactors = FALSE) # Oliver should use delimter = ","
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
  
  
  cat("=== STEP 4: Exporting Data ===\n")
  # Export
  write_csv(df_clean,   OUT_POS)
  
  # Write stage counts for Sankey diagram
  iedb_stage_counts <- tibble(
    stage = c("raw_assays", "duplicates", "unique_pairs", "ptm", "no_ptm"),
    count = c(nrow(df_raw), nrow(df_formatted) - nrow(df_dedup), nrow(df_dedup),
              nrow(df_dedup) - nrow(df_clean), nrow(df_clean))
  )
  write_csv(iedb_stage_counts, OUT_STAGES)
  cat("Stage counts saved to ", OUT_STAGES, "\n", sep = "")
  
  cat("✅ Saved cleaned files to ", dirname(OUT_POS), "/\n", sep = "")
  if (ALLELE_TAG == "hla0201") {
    cat("Next step: run python src/fetch/fetch_uniprot_fastas.py\n")
  } else {
    cat("Next step: run python src/pipeline_a0101/fetch/fetch_a0101_fastas.py\n")
  }
