library(tidyverse)
source("src/pipeline/r/functions.R")
set_working_directory()

# ── Allele + window-width switch (paths/params only; logic untouched) ─────────
# Usage: Rscript src/pipeline/r/07_alpha_fold_features.R [hla0201|hla_a0101]
ALLELE_TAG <- parse_allele_arg()
FLANK_SIZE <- as.integer(Sys.getenv("FLANK_SIZE", "5"))
if (!FLANK_SIZE %in% c(5L, 10L)) {
  stop("FLANK_SIZE must be 5 or 10, got: ", Sys.getenv("FLANK_SIZE"))
}
cat("=== Allele:", ALLELE_TAG, "| FLANK_SIZE:", FLANK_SIZE, "->",
    2 * FLANK_SIZE + 9, "-mer window ===\n")

# ── 0. Config ─────────────────────────────────────────────────────────────────
# During development, point at the single test file folder
# Later just change this path to the full alphafold directory
af_dirs <- "data/processed/structures/alphafold/"

# ── 1. Load peptide table ─────────────────────────────────────────────────────
df_raw <- read_csv(pp("epitopes_pos_and_neg_features_with_nsp3.csv", ALLELE_TAG))

df_peptides <- df_raw |>
  select(peptide, n_flank, c_flank, full_context, uniprot_id, start, end,
         protein_length, label, n_flank_len_real, c_flank_len_real) |>
  distinct() |>
  rename(pep_start = start, pep_end = end) |>
  mutate(
    pep_start    = as.integer(pep_start),
    pep_end      = as.integer(pep_end),
    nflank_start = pep_start - n_flank_len_real,
    nflank_end   = pep_start - 1L,
    cflank_start = pep_end   + 1L,
    cflank_end   = pep_end   + c_flank_len_real,
    window_start = pep_start - n_flank_len_real,
    window_end   = pep_end   + c_flank_len_real    
  )

cat("Peptides loaded:", nrow(df_peptides), "\n")

# ── 2. Build pLDDT lookup from all available PDB files ────────────────────────
proteins_needed <- unique(df_peptides$uniprot_id)
cat("Unique proteins needed:", length(proteins_needed), "\n")

plddt_lookup_split <- list()
for (d in af_dirs) {
  plddt_lookup_split <- c(plddt_lookup_split, build_plddt_lookup(
    proteins_needed = proteins_needed,
    dir = d
  ))
}

# Which proteins actually got matched?
matched_proteins <- names(plddt_lookup_split)
cat("Proteins with pLDDT data:", length(matched_proteins), "\n")


# ── 3. Filter to only peptides with AlphaFold coverage ───────────────────────
df_af <- df_peptides |>
  filter(uniprot_id %in% matched_proteins)

cat("Peptides with AlphaFold coverage:", nrow(df_af), "/", nrow(df_peptides), "\n")

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

cat("\n Removing", length(proteins_to_remove), "proteins entirely,",
    "as they contain at least one peptide outside AlphaFold model coverage.\n")
cat("Peptides before removal:", nrow(df_af),                    "\n")
cat("Peptides after removal: ", nrow(df_af_clean),              "\n")
cat("Peptides removed:       ", nrow(df_af) - nrow(df_af_clean),"\n")


alphafold_features <- df_af_clean |>
  mutate(
    # ── Boundary values for empty-flank imputation (P1 = first, P9 = last) ──
    plddt_p1 = pmap(
      list(uniprot_id, pep_start, pep_start),
      extract_plddt_vector, lookup_split = plddt_lookup_split
    ),
    plddt_p9 = pmap(
      list(uniprot_id, pep_end, pep_end),
      extract_plddt_vector, lookup_split = plddt_lookup_split
    ),

    # ── Per-residue pLDDT vectors ──────────────────────────────────────────
    plddt_vec_peptide = pmap(
      list(uniprot_id, pep_start, pep_end),
      extract_plddt_vector, lookup_split = plddt_lookup_split
    ),
    plddt_vec_nflank = pmap(
      list(uniprot_id, nflank_start, nflank_end, plddt_p1),
      \(uid, s, e, fb) extract_plddt_imputed(uid, s, e, plddt_lookup_split, FLANK_SIZE, "left", fb)
    ),
    plddt_vec_cflank = pmap(
      list(uniprot_id, cflank_start, cflank_end, plddt_p9),
      \(uid, s, e, fb) extract_plddt_imputed(uid, s, e, plddt_lookup_split, FLANK_SIZE, "right", fb)
    ),

    # ── Scalar summaries: mean per region ──────────────────────────
    mean_plddt_peptide = map_dbl(plddt_vec_peptide, ~ mean(.x, na.rm = TRUE)),
    mean_plddt_nflank  = map_dbl(plddt_vec_nflank,  ~ mean(.x, na.rm = TRUE)),
    mean_plddt_cflank  = map_dbl(plddt_vec_cflank,  ~ mean(.x, na.rm = TRUE))
  ) |>
  # Drop intermediate list columns and boundary vectors
  select(-plddt_p1, -plddt_p9, -plddt_vec_peptide, -plddt_vec_nflank, -plddt_vec_cflank)


# Check how many peptides have no pLDDT coverage
n_missing <- alphafold_features |>
  filter(is.na(mean_plddt_peptide)) |>
  nrow()

cat("Peptides with no pLDDT coverage:", n_missing, "\n")

cat("\n=== pLDDT feature summary ===\n")
print(alphafold_features |>
        select(mean_plddt_peptide,
               mean_plddt_nflank, mean_plddt_cflank) |>
        summary())

# ── 4. Quick sanity check on the test protein ─────────────────────────────────
# Show a few rows so you can verify the vectors look right
alphafold_features |>
  filter(uniprot_id == "X6REB3") |>
  select(peptide, pep_start, pep_end, mean_plddt_peptide) |>
  head(5) |>
  print()

# ── 5. Save ───────────────────────────────────────────────────────────────────

# Select only the new pLDDT scalar columns + join keys from alphafold_features
# Exclude list columns (not CSV-friendly) and columns already in df_raw
alphafold_scalar <- alphafold_features |>
  select(
    # Join keys
    peptide, uniprot_id, n_flank, c_flank, full_context,
    pep_start, pep_end,
    # pLDDT features: peptide
    mean_plddt_peptide,
    # pLDDT features: N-flank
    mean_plddt_nflank,
    # pLDDT features: C-flank
    mean_plddt_cflank
  )

df_all <- df_raw |>
  left_join(
    alphafold_scalar,
    by = c(
      "peptide", "uniprot_id",
      "n_flank", "c_flank", "full_context",
      "start" = "pep_start",
      "end"   = "pep_end"
    )
  )

cat("df_raw rows:  ", nrow(df_raw),  "\n")
cat("df_all rows:  ", nrow(df_all),  "\n")
cat("df_all cols:  ", ncol(df_all),  "\n")
cat("Rows with pLDDT data:", sum(!is.na(df_all$mean_plddt_peptide)), "\n")
cat("Rows without pLDDT (removed proteins or no AF file):",
    sum(is.na(df_all$mean_plddt_peptide)), "\n")

# ── Dedup identical context windows (one row per window) ──
# Window carries both labels -> keep the positive; else keep first row.
# Same peptide with DIFFERENT flanks is kept. Rule == make_variant_dataset.py
# --variant remove_dup_windows (row-identical to the Python reference).
n_before_dedup <- nrow(df_all)
ctx <- df_all$full_context
is_pos <- df_all$label == 1

first_any <- !duplicated(ctx)
first_pos <- rep(FALSE, n_before_dedup)
first_pos[which(is_pos)[!duplicated(ctx[is_pos])]] <- TRUE
has_pos <- ctx %in% unique(ctx[is_pos])
keep <- ifelse(has_pos, first_pos, first_any)

df_all <- df_all[keep, ]

n_dropped <- n_before_dedup - nrow(df_all)
cat("Context-window dedup: dropped", n_dropped, "rows (",
    n_before_dedup, "->", nrow(df_all), ")\n")
cat("  label balance:", sum(df_all$label == 1), "pos /",
    sum(df_all$label == 0), "neg\n")

# Hard guards (any allele):
stopifnot("duplicate context windows remain" =
            !any(duplicated(df_all$full_context)))
win_len <- 2 * FLANK_SIZE + 9
if (!all(nchar(df_all$full_context) == win_len)) {
  stop("full_context must be ", win_len, " chars after dedup — got widths: ",
       paste(sort(unique(nchar(df_all$full_context))), collapse = ", "))
}
# 0201 row pins (Python references: df_all_remove_dup_windows @29,
# flank5_rmdup @19)
if (ALLELE_TAG == "hla0201") {
  if (FLANK_SIZE == 10L) {
    stopifnot("0201/29mer rows" = n_before_dedup == 46617L &&
                nrow(df_all) == 45510L && n_dropped == 1107L)
  } else {
    stopifnot("0201/19mer rows" = n_before_dedup == 46617L &&
                nrow(df_all) == 45278L && n_dropped == 1339L)
  }
}

write_csv(df_all, pp("df_all.csv", ALLELE_TAG))
saveRDS(alphafold_features,
        pp("alphafold_plddt_features_with_vectors.rds", ALLELE_TAG))

cat("\nSaved:\n  ", pp("df_all.csv", ALLELE_TAG),
    " (all features, scalar only)\n  ",
    pp("alphafold_plddt_features_with_vectors.rds", ALLELE_TAG),
    " (full vectors)\n", sep = "")

# ============================================================================
# MISSINGNESS REPORT: Final dataset quality summary
# ============================================================================

cat("\n\n")
cat(strrep("=", 70), "\n")
cat("  MISSINGNESS REPORT: Final dataset quality summary\n")
cat(strrep("=", 70), "\n\n")

# AlphaFold missingness
af_missing <- df_all |>
  summarise(
    total = n(),
    missing_plddt = sum(is.na(mean_plddt_peptide)),
    missing_pct = round(missing_plddt / total * 100, 1)
  )

cat("AlphaFold pLDDT:\n")
cat("  Total entries:", af_missing$total, "\n")
cat("  Missing:", af_missing$missing_plddt, "(", af_missing$missing_pct, "%)\n\n")

# NetSurfP missingness (check for mean_rsa_peptide which is from step 06)
if ("mean_rsa_peptide" %in% names(df_all)) {
  nsp_missing <- df_all |>
    summarise(
      total = n(),
      missing_rsa = sum(is.na(mean_rsa_peptide)),
      missing_pct = round(missing_rsa / total * 100, 1)
    )
  
  cat("NetSurfP RSA:\n")
  cat("  Total entries:", nsp_missing$total, "\n")
  cat("  Missing:", nsp_missing$missing_rsa, "(", nsp_missing$missing_pct, "%)\n\n")
} else {
  cat("NetSurfP features not found in df_all (step 06 not yet run?)\n\n")
}

# Combined summary
cat(strrep("-", 50), "\n")
cat("Summary:\n")
cat("  AlphaFold pLDDT missing:", af_missing$missing_plddt, "\n")
if ("mean_rsa_peptide" %in% names(df_all)) {
  cat("  NetSurfP RSA missing:   ", nsp_missing$missing_rsa, "\n")
}

# Save report
df_missingness <- tibble(
  feature_group = c("AlphaFold pLDDT peptide", "AlphaFold pLDDT nflank", "AlphaFold pLDDT cflank",
                     if ("mean_rsa_peptide" %in% names(df_all)) c("NetSurfP RSA peptide", "NetSurfP RSA nflank", "NetSurfP RSA cflank")),
  missing_n = c(
    sum(is.na(df_all$mean_plddt_peptide)),
    sum(is.na(df_all$mean_plddt_nflank)),
    sum(is.na(df_all$mean_plddt_cflank)),
    if ("mean_rsa_peptide" %in% names(df_all)) c(
      sum(is.na(df_all$mean_rsa_peptide)),
      sum(is.na(df_all$mean_rsa_nflank)),
      sum(is.na(df_all$mean_rsa_cflank))
    )
  ),
  total = nrow(df_all),
  missing_pct = round(missing_n / total * 100, 1)
)

write_csv(df_missingness, pp("missingness_report.csv", ALLELE_TAG))
cat("\nSaved: ", pp("missingness_report.csv", ALLELE_TAG), "\n", sep = "")
print(df_missingness, n = Inf)

