library(tidyverse)
library(bio3d)
source("src/functions.R")
set_working_directory()

# ── 0. Config ─────────────────────────────────────────────────────────────────
# During development, point at the single test file folder
# Later just change this path to the full alphafold directory
ALPHAFOLD_DIR <- "data/processed/structures/alphafold/"          # ← swap to "data/processed/structures/alphafold/" later

# ── 1. Load peptide table ─────────────────────────────────────────────────────
df_raw <- read_csv("data/processed/epitopes_pos_and_neg_features_with_nsp3.csv")

df_peptides <- df_raw |>
  select(peptide, n_flank, c_flank, full_context, uniprot_id, start, end, protein_length, label) |>
  distinct() |>
  rename(pep_start = start, pep_end = end) |>
  mutate(
    pep_start    = as.integer(pep_start),
    pep_end      = as.integer(pep_end),
    nflank_start = pep_start - nchar(n_flank),
    nflank_end   = pep_start - 1L,
    cflank_start = pep_end   + 1L,
    cflank_end   = pep_end   + nchar(c_flank)
  )

cat("Peptides loaded:", nrow(df_peptides), "\n")

# ── 2. Build pLDDT lookup from all available PDB files ────────────────────────
proteins_needed <- unique(df_peptides$uniprot_id)
cat("Unique proteins needed:", length(proteins_needed), "\n")

plddt_lookup_split <- build_plddt_lookup(
  proteins_needed = proteins_needed,
  dir             = ALPHAFOLD_DIR
)

# Which proteins actually got matched?
matched_proteins <- names(plddt_lookup_split)
cat("Proteins with pLDDT data:", length(matched_proteins), "\n")


# ── Validate AlphaFold model coverage against known protein lengths ────────────
protein_lengths <- df_peptides |>
  select(uniprot_id, protein_length) |>
  distinct()

modelled_ranges <- purrr::map(names(plddt_lookup_split), ~ {
  lkp <- plddt_lookup_split[[.x]]
  tibble(
    uniprot_id     = .x,
    min_modelled   = min(lkp$residue_num),
    max_modelled   = max(lkp$residue_num),
    total_modelled = nrow(lkp)
  )
}) |>
  list_rbind()

length_comparison <- modelled_ranges |>
  left_join(protein_lengths, by = "uniprot_id") |>
  mutate(
    length_diff  = protein_length - max_modelled,
    modelled_pct = round(max_modelled / protein_length * 100, 1)
  ) |>
  arrange(desc(length_diff))

cat("\n=== AlphaFold model coverage vs protein length (only mismatches) ===\n")
length_comparison |>
  filter(length_diff != 0) |>
  print(n = Inf, width = Inf)

# Find proteins where ANY peptide falls outside the modelled range
# These proteins are removed entirely — partial coverage is not reliable
proteins_with_outside_peptides <- df_af |>
  left_join(modelled_ranges |> select(uniprot_id, max_modelled), by = "uniprot_id") |>
  filter(pep_end > max_modelled) |>
  distinct(uniprot_id)

cat("\n=== Proteins with at least one peptide outside AlphaFold model range ===\n")
proteins_with_outside_peptides |>
  left_join(length_comparison |> select(uniprot_id, protein_length, max_modelled,
                                        length_diff, modelled_pct),
            by = "uniprot_id") |>
  print(width = Inf)

# Show the specific out-of-range peptides for documentation
peptides_outside_model <- df_af |>
  left_join(modelled_ranges |> select(uniprot_id, max_modelled), by = "uniprot_id") |>
  filter(
    uniprot_id %in% proteins_with_outside_peptides$uniprot_id,
    pep_end > max_modelled
  ) |>
  mutate(residues_outside = pep_end - max_modelled) |>
  select(peptide, uniprot_id, pep_start, pep_end, max_modelled, residues_outside) |>
  arrange(uniprot_id, pep_start)

cat("\nOut-of-range peptides (documentation only):\n")
print(peptides_outside_model, n = Inf, width = Inf)

# Remove all peptides from proteins that have ANY out-of-range peptide
proteins_to_remove <- proteins_with_outside_peptides$uniprot_id

df_af_clean <- df_af |>
  filter(!uniprot_id %in% proteins_to_remove)

cat("\n⚠️  Removing", length(proteins_to_remove), "proteins entirely,",
    "as they contain at least one peptide outside AlphaFold model coverage.\n")
cat("Peptides before removal:", nrow(df_af),                    "\n")
cat("Peptides after removal: ", nrow(df_af_clean),              "\n")
cat("Peptides removed:       ", nrow(df_af) - nrow(df_af_clean),"\n")


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


# Check how many peptides have no pLDDT coverage
n_missing <- alphafold_features |>
  filter(is.na(mean_plddt_peptide)) |>
  nrow()

cat("Peptides with no pLDDT coverage:", n_missing, "\n")

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
df_all <- df_raw |>
  left_join(alphafold_features, by =c("   ")) |>
  write_csv("data/processed/df_all.csv")



df_final_nsp3 <- df_raw |>
  left_join(netsurfp_features, by = c("peptide", "uniprot_id", 
                                      "n_flank", "c_flank", "full_context", 
                                      "start" = "pep_start", "end" = "pep_end"))


# Save full version with list columns as an RDS (preserves vectors)
saveRDS(alphafold_features,
        "data/processed/alphafold_plddt_features_with_vectors.rds")

cat("\n Saved:\n  data/processed/alphafold_plddt_features.csv (scalar means)\n  data/processed/alphafold_plddt_features_with_vectors.rds (full vectors)\n")



