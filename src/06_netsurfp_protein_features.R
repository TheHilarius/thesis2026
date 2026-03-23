library(tidyverse)
library(data.table)
source("src/functions.R")
set_working_directory()

df_raw <- read_csv("data/processed/epitopes_pos_and_neg_features.csv")

df_peptides <- df_raw |>
  select(peptide, n_flank, c_flank, full_context, uniprot_id, start, end) |>
  distinct() |> 
  rename(pep_start = start, pep_end = end) |>
  mutate(
    pep_start    = as.integer(pep_start),
    pep_end      = as.integer(pep_end),
    nflank_start = pep_start - nchar(n_flank),
    nflank_end   = pep_start - 1L,
    cflank_start = pep_end   + 1L,
    cflank_end   = pep_end   + nchar(c_flank),
    window_start = pep_start - nchar(n_flank),
    window_end   = pep_end   + nchar(c_flank)
  )

cat("Epitopes loaded:", nrow(df_peptides), "\n")

cat("Scanning NSP3 output files...\n")
path_df <- build_nsp3_path_lookup()
cat("NSP3 CSV files found:", nrow(path_df), "\n")

needed  <- unique(df_peptides$uniprot_id)
matched <- sum(needed %in% path_df$uniprot_id)
cat("Unique proteins needed:  ", length(needed), "\n")
cat("Matched to NSP3 output:  ", matched, "\n")
cat("Missing:                 ", length(needed) - matched, "\n")

path_df_needed <- path_df |>
  filter(uniprot_id %in% needed)

cat("Loading NSP3 data...\n")
nsp3_data <- path_df_needed |>
  mutate(data = map2(path, uniprot_id, read_nsp3_csv, .progress = TRUE)) |>
  select(data) |>
  unnest(data)

cat("Total residues loaded:", nrow(nsp3_data), "\n")


cat("Indexing by protein...\n")
nsp3_split <- split(nsp3_data, nsp3_data$uniprot)
cat("Proteins indexed:", length(nsp3_split), "\n")


cat("Extracting NSP3 features per peptide...\n")

netsurfp_features <- df_peptides |>
  mutate(
    feats = pmap(
      list(
        uniprot_id,
        nflank_start, nflank_end,
        pep_start,    pep_end,
        cflank_start, cflank_end,
        window_start, window_end
      ),
      extract_nsp3_windows,
      nsp3_split = nsp3_split,
      .progress  = TRUE
    )
  ) |>
  unnest(feats)

cat("✅ Done!\n")
cat("Rows:   ", nrow(netsurfp_features), "\n")
cat("Columns:", ncol(netsurfp_features), "\n")

netsurfp_features <- netsurfp_features |>
  select(
    # Identity
    peptide, n_flank, c_flank, full_context, uniprot_id, pep_start, pep_end,
    
    # Peptide window
    mean_rsa_peptide, mean_disorder_peptide,
    frac_helix_peptide, frac_sheet_peptide, frac_coil_peptide,
    mean_p_q3_H_peptide, mean_p_q3_E_peptide, mean_p_q3_C_peptide,
    frac_q8_G_peptide, frac_q8_H_peptide, frac_q8_I_peptide,
    frac_q8_B_peptide, frac_q8_E_peptide, frac_q8_S_peptide,
    frac_q8_T_peptide, frac_q8_C_peptide,
    mean_p_q8_G_peptide, mean_p_q8_H_peptide, mean_p_q8_I_peptide,
    mean_p_q8_B_peptide, mean_p_q8_E_peptide, mean_p_q8_S_peptide,
    mean_p_q8_T_peptide, mean_p_q8_C_peptide,
    
    # N-flank window
    mean_rsa_nflank, mean_disorder_nflank,
    frac_helix_nflank, frac_sheet_nflank, frac_coil_nflank,
    mean_p_q3_H_nflank, mean_p_q3_E_nflank, mean_p_q3_C_nflank,
    frac_q8_G_nflank, frac_q8_H_nflank, frac_q8_I_nflank,
    frac_q8_B_nflank, frac_q8_E_nflank, frac_q8_S_nflank,
    frac_q8_T_nflank, frac_q8_C_nflank,
    mean_p_q8_G_nflank, mean_p_q8_H_nflank, mean_p_q8_I_nflank,
    mean_p_q8_B_nflank, mean_p_q8_E_nflank, mean_p_q8_S_nflank,
    mean_p_q8_T_nflank, mean_p_q8_C_nflank,
    
    # C-flank window
    mean_rsa_cflank, mean_disorder_cflank,
    frac_helix_cflank, frac_sheet_cflank, frac_coil_cflank,
    mean_p_q3_H_cflank, mean_p_q3_E_cflank, mean_p_q3_C_cflank,
    frac_q8_G_cflank, frac_q8_H_cflank, frac_q8_I_cflank,
    frac_q8_B_cflank, frac_q8_E_cflank, frac_q8_S_cflank,
    frac_q8_T_cflank, frac_q8_C_cflank,
    mean_p_q8_G_cflank, mean_p_q8_H_cflank, mean_p_q8_I_cflank,
    mean_p_q8_B_cflank, mean_p_q8_E_cflank, mean_p_q8_S_cflank,
    mean_p_q8_T_cflank, mean_p_q8_C_cflank,
    
    # Full context window
    mean_rsa_full_context, mean_disorder_full_context,
    frac_helix_full_context, frac_sheet_full_context, frac_coil_full_context,
    mean_p_q3_H_full_context, mean_p_q3_E_full_context, mean_p_q3_C_full_context,
    frac_q8_G_full_context, frac_q8_H_full_context, frac_q8_I_full_context,
    frac_q8_B_full_context, frac_q8_E_full_context, frac_q8_S_full_context,
    frac_q8_T_full_context, frac_q8_C_full_context,
    mean_p_q8_G_full_context, mean_p_q8_H_full_context, mean_p_q8_I_full_context,
    mean_p_q8_B_full_context, mean_p_q8_E_full_context, mean_p_q8_S_full_context,
    mean_p_q8_T_full_context, mean_p_q8_C_full_context
  )

# Alphafold - plddt
library(tidyverse)
library(data.table)
source("src/functions.R")
set_working_directory()

df_raw <- read_csv("data/processed/epitopes_pos_and_neg_features.csv")

df_peptides <- df_raw |>
  select(peptide, n_flank, c_flank, full_context, uniprot_id, start, end) |>
  distinct() |> 
  rename(pep_start = start, pep_end = end) |>
  mutate(
    pep_start    = as.integer(pep_start),
    pep_end      = as.integer(pep_end),
    nflank_start = pep_start - nchar(n_flank),
    nflank_end   = pep_start - 1L,
    cflank_start = pep_end   + 1L,
    cflank_end   = pep_end   + nchar(c_flank),
    window_start = pep_start - nchar(n_flank),
    window_end   = pep_end   + nchar(c_flank)
  )

cat("Epitopes loaded:", nrow(df_peptides), "\n")

cat("Scanning NSP3 output files...\n")
path_df <- build_nsp3_path_lookup()
cat("NSP3 CSV files found:", nrow(path_df), "\n")

needed  <- unique(df_peptides$uniprot_id)
matched <- sum(needed %in% path_df$uniprot_id)
cat("Unique proteins needed:  ", length(needed), "\n")
cat("Matched to NSP3 output:  ", matched, "\n")
cat("Missing:                 ", length(needed) - matched, "\n")

path_df_needed <- path_df |>
  filter(uniprot_id %in% needed)

cat("Loading NSP3 data...\n")
nsp3_data <- path_df_needed |>
  mutate(data = map2(path, uniprot_id, read_nsp3_csv, .progress = TRUE)) |>
  select(data) |>
  unnest(data)

cat("Total residues loaded:", nrow(nsp3_data), "\n")


cat("Indexing by protein...\n")
nsp3_split <- split(nsp3_data, nsp3_data$uniprot)
cat("Proteins indexed:", length(nsp3_split), "\n")


cat("Extracting NSP3 features per peptide...\n")

netsurfp_features <- df_peptides |>
  mutate(
    feats = pmap(
      list(
        uniprot_id,
        nflank_start, nflank_end,
        pep_start,    pep_end,
        cflank_start, cflank_end,
        window_start, window_end
      ),
      extract_nsp3_windows,
      nsp3_split = nsp3_split,
      .progress  = TRUE
    )
  ) |>
  unnest(feats)

cat("✅ Done!\n")
cat("Rows:   ", nrow(netsurfp_features), "\n")
cat("Columns:", ncol(netsurfp_features), "\n")

netsurfp_features <- netsurfp_features |>
  select(
    # Identity
    peptide, n_flank, c_flank, full_context, uniprot_id, pep_start, pep_end,
    
    # Peptide window
    mean_rsa_peptide, mean_disorder_peptide,
    frac_helix_peptide, frac_sheet_peptide, frac_coil_peptide,
    mean_p_q3_H_peptide, mean_p_q3_E_peptide, mean_p_q3_C_peptide,
    frac_q8_G_peptide, frac_q8_H_peptide, frac_q8_I_peptide,
    frac_q8_B_peptide, frac_q8_E_peptide, frac_q8_S_peptide,
    frac_q8_T_peptide, frac_q8_C_peptide,
    mean_p_q8_G_peptide, mean_p_q8_H_peptide, mean_p_q8_I_peptide,
    mean_p_q8_B_peptide, mean_p_q8_E_peptide, mean_p_q8_S_peptide,
    mean_p_q8_T_peptide, mean_p_q8_C_peptide,
    
    # N-flank window
    mean_rsa_nflank, mean_disorder_nflank,
    frac_helix_nflank, frac_sheet_nflank, frac_coil_nflank,
    mean_p_q3_H_nflank, mean_p_q3_E_nflank, mean_p_q3_C_nflank,
    frac_q8_G_nflank, frac_q8_H_nflank, frac_q8_I_nflank,
    frac_q8_B_nflank, frac_q8_E_nflank, frac_q8_S_nflank,
    frac_q8_T_nflank, frac_q8_C_nflank,
    mean_p_q8_G_nflank, mean_p_q8_H_nflank, mean_p_q8_I_nflank,
    mean_p_q8_B_nflank, mean_p_q8_E_nflank, mean_p_q8_S_nflank,
    mean_p_q8_T_nflank, mean_p_q8_C_nflank,
    
    # C-flank window
    mean_rsa_cflank, mean_disorder_cflank,
    frac_helix_cflank, frac_sheet_cflank, frac_coil_cflank,
    mean_p_q3_H_cflank, mean_p_q3_E_cflank, mean_p_q3_C_cflank,
    frac_q8_G_cflank, frac_q8_H_cflank, frac_q8_I_cflank,
    frac_q8_B_cflank, frac_q8_E_cflank, frac_q8_S_cflank,
    frac_q8_T_cflank, frac_q8_C_cflank,
    mean_p_q8_G_cflank, mean_p_q8_H_cflank, mean_p_q8_I_cflank,
    mean_p_q8_B_cflank, mean_p_q8_E_cflank, mean_p_q8_S_cflank,
    mean_p_q8_T_cflank, mean_p_q8_C_cflank,
    
    # Full context window
    mean_rsa_full_context, mean_disorder_full_context,
    frac_helix_full_context, frac_sheet_full_context, frac_coil_full_context,
    mean_p_q3_H_full_context, mean_p_q3_E_full_context, mean_p_q3_C_full_context,
    frac_q8_G_full_context, frac_q8_H_full_context, frac_q8_I_full_context,
    frac_q8_B_full_context, frac_q8_E_full_context, frac_q8_S_full_context,
    frac_q8_T_full_context, frac_q8_C_full_context,
    mean_p_q8_G_full_context, mean_p_q8_H_full_context, mean_p_q8_I_full_context,
    mean_p_q8_B_full_context, mean_p_q8_E_full_context, mean_p_q8_S_full_context,
    mean_p_q8_T_full_context, mean_p_q8_C_full_context
  )
