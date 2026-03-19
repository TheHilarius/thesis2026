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
#
# Alphafold - pLDDT
#
library(tidyverse)
library(bio3d)

alphafold_dir <- "data/processed/structures/alphafold/"

af_files   <- list.files(alphafold_dir, pattern = "\\.pdb$", full.names = FALSE)
af_uniprot <- str_remove(af_files, "\\.pdb$")

# Only load proteins that are actually in our dataset
proteins_needed <- unique(df_peptides$uniprot_id)
af_uniprot_filt <- af_uniprot[af_uniprot %in% proteins_needed]

message("Loading pLDDT for ", length(af_uniprot_filt), " / ", 
        length(proteins_needed), " proteins found in AlphaFold...")

extract_plddt_lookup <- function(uniprot_id, dir = alphafold_dir) {
  pdb_path <- file.path(dir, paste0(uniprot_id, ".pdb"))
  tryCatch({
    pdb     <- read.pdb(pdb_path, verbose = FALSE)
    ca      <- pdb$atom[pdb$atom$elety == "CA", ]
    tibble(
      uniprot_id  = uniprot_id,
      residue_num = ca$resno,
      plddt       = ca$b
    )
  }, error = function(e) {
    warning(sprintf("Failed: %s — %s", uniprot_id, e$message))
    NULL
  })
}

# Full lookup table: one row per residue per protein
plddt_lookup <- map(af_uniprot_filt, extract_plddt_lookup) |>
  list_rbind()

message("Loaded pLDDT for ", n_distinct(plddt_lookup$uniprot_id), " proteins, ",
        nrow(plddt_lookup), " residues total.")


compute_region_plddt <- function(uid, start, end, lookup) {
  rows <- lookup[lookup$uniprot_id == uid &
                   lookup$residue_num >= start &
                   lookup$residue_num <= end, ]
  if (nrow(rows) == 0) return(NA_real_)
  mean(rows$plddt, na.rm = TRUE)
}

# Work only on rows that have an AlphaFold structure
df_af <- df_peptides |>
  filter(uniprot_id %in% af_uniprot_filt)

message("Computing per-region pLDDT for ", nrow(df_af), " peptide entries...")

# Use a split-apply pattern for speed (avoids slow rowwise())
plddt_lookup_split <- split(plddt_lookup, plddt_lookup$uniprot_id)

alphafold_features <- df_af |>
  mutate(
    mean_plddt_peptide = map2_dbl(
      uniprot_id, pep_start,
      ~ {
        pep_end_val <- df_af$pep_end[match(.x, df_af$uniprot_id)]
        lkp <- plddt_lookup_split[[.x]]
        if (is.null(lkp)) return(NA_real_)
        mean(lkp$plddt[lkp$residue_num >= .y & lkp$residue_num <= pep_end_val], na.rm = TRUE)
      }
    )
  )
