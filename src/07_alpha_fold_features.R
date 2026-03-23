library(tidyverse)
library(bio3d)
source("src/functions.R")
set_working_directory()

# ── 0. Config ─────────────────────────────────────────────────────────────────
# During development, point at the single test file folder
# Later just change this path to the full alphafold directory
ALPHAFOLD_DIR <- "data/raw/"          # ← swap to "data/processed/structures/alphafold/" later

# ── 1. Load peptide table ─────────────────────────────────────────────────────
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

cat("Peptides loaded:", nrow(df_peptides), "\n")

# ── 2. Build pLDDT lookup from all available PDB files ────────────────────────
proteins_needed <- unique(df_peptides$uniprot_id)

plddt_lookup_split <- build_plddt_lookup(
  proteins_needed = proteins_needed,
  dir             = ALPHAFOLD_DIR
)

# Which proteins actually got matched?
matched_proteins <- names(plddt_lookup_split)
cat("Proteins with pLDDT data:", length(matched_proteins), "\n")

# ── 3. Attach per-residue pLDDT vectors + regional means ──────────────────────
# Work only on peptides whose protein has an AlphaFold structure
df_af <- df_peptides |>
  filter(uniprot_id %in% matched_proteins)

cat("Peptides with AF coverage:", nrow(df_af), "/", nrow(df_peptides), "\n")

alphafold_features <- df_af |>
  mutate(
    # ── Per-residue pLDDT vectors (list column) ─────────────────────────────
    # Each cell is a numeric vector of length = region length
    plddt_vec_peptide      = pmap(
      list(uniprot_id, pep_start,    pep_end),
      extract_plddt_vector,
      lookup_split = plddt_lookup_split
    ),
    plddt_vec_nflank       = pmap(
      list(uniprot_id, nflank_start, nflank_end),
      extract_plddt_vector,
      lookup_split = plddt_lookup_split
    ),
    plddt_vec_cflank       = pmap(
      list(uniprot_id, cflank_start, cflank_end),
      extract_plddt_vector,
      lookup_split = plddt_lookup_split
    ),
    plddt_vec_full_context = pmap(
      list(uniprot_id, window_start, window_end),
      extract_plddt_vector,
      lookup_split = plddt_lookup_split
    ),
    
    # ── Scalar summaries (mean pLDDT per region) ────────────────────────────
    mean_plddt_peptide      = map_dbl(plddt_vec_peptide,      ~ mean(.x, na.rm = TRUE)),
    mean_plddt_nflank       = map_dbl(plddt_vec_nflank,       ~ mean(.x, na.rm = TRUE)),
    mean_plddt_cflank       = map_dbl(plddt_vec_cflank,       ~ mean(.x, na.rm = TRUE)),
    mean_plddt_full_context = map_dbl(plddt_vec_full_context, ~ mean(.x, na.rm = TRUE)),
    
    # Standard deviation of pLDDT in each region (measure of confidence variability)
    sd_plddt_peptide             = map_dbl(plddt_vec_peptide,      ~ sd(.x,   na.rm = TRUE)),
    sd_plddt_full_context        = map_dbl(plddt_vec_full_context, ~ sd(.x,   na.rm = TRUE)),
    
    frac_disordered_peptide      = map_dbl(plddt_vec_peptide,      ~ mean(.x < 70, na.rm = TRUE)),
    frac_disordered_full_context = map_dbl(plddt_vec_full_context, ~ mean(.x < 70, na.rm = TRUE)),
    # ── Min pLDDT in peptide (weakest-confidence residue) ───────────────────
    min_plddt_peptide       = map_dbl(plddt_vec_peptide,      ~ min(.x, na.rm = TRUE))
  )

cat("\n=== pLDDT feature summary ===\n")
print(alphafold_features |>
        select(mean_plddt_peptide, mean_plddt_nflank,
               mean_plddt_cflank,  mean_plddt_full_context,
               min_plddt_peptide) |>
        summary())

# ── 4. Quick sanity check on the test protein ─────────────────────────────────
# Show a few rows so you can verify the vectors look right
alphafold_features |>
  filter(uniprot_id == "X6REB3") |>
  select(peptide, pep_start, pep_end,
         plddt_vec_peptide, mean_plddt_peptide) |>
  head(5) |>
  print()

# ── 5. Save ───────────────────────────────────────────────────────────────────
# Save scalar-only version (list columns are not CSV-friendly)
alphafold_features |>
  select(peptide, uniprot_id, pep_start, pep_end,
         mean_plddt_peptide, mean_plddt_nflank,
         mean_plddt_cflank,  mean_plddt_full_context,
         min_plddt_peptide) |>
  write_csv("data/processed/alphafold_plddt_features.csv")

# Save full version with list columns as an RDS (preserves vectors)
saveRDS(alphafold_features,
        "data/processed/alphafold_plddt_features_with_vectors.rds")

cat("\n Saved:\n  data/processed/alphafold_plddt_features.csv (scalar means)\n  data/processed/alphafold_plddt_features_with_vectors.rds (full vectors)\n")