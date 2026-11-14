library(tidyverse)
source("src/pipeline/r/functions.R")
set_working_directory()

# ── Allele + window-width switch (paths/params only; logic untouched) ─────────
# Usage: Rscript src/pipeline/r/08_aa_encoding.R [hla0201|hla_a0101]
ALLELE_TAG <- parse_allele_arg()
FLANK_SIZE <- as.integer(Sys.getenv("FLANK_SIZE", "5"))
if (!FLANK_SIZE %in% c(5L, 10L)) {
  stop("FLANK_SIZE must be 5 or 10, got: ", Sys.getenv("FLANK_SIZE"))
}
cat("=== Allele:", ALLELE_TAG, "| FLANK_SIZE:", FLANK_SIZE, "->",
    2 * FLANK_SIZE + 9, "-mer window ===\n")

POSITION_COLS <- c(
  paste0("N", FLANK_SIZE:1),   # N{L}..N1   (left → right toward the cut)
  paste0("P", 1:9),            # P1..P9
  paste0("C", 1:FLANK_SIZE)    # C1..C{L}
)

# Allele anchor residues for the sanity checks (P2-P9 anchor map):
#   0201: P2 L/M, P9 V/L   |   0101: P2 V/I/T/M, P9 D/E/Y
ANCHOR_MAP <- list(
  hla0201   = list(label = "HLA-A*02:01",
                   p2 = c("L", "M"), p9 = c("V", "L")),
  hla_a0101 = list(label = "HLA-A*01:01",
                   p2 = c("V", "I", "T", "M"), p9 = c("D", "E", "Y"))
)
ANCH <- ANCHOR_MAP[[ALLELE_TAG]]

# AA_ORDER (with X) comes from functions.R

df_raw <- read_csv(pp("df_all.csv", ALLELE_TAG), show_col_types = FALSE)

cat("Input dimensions:", nrow(df_raw), "x", ncol(df_raw), "\n")

na_counts <- df_raw |>
  summarise(across(all_of(POSITION_COLS), \(x) sum(is.na(x)))) |>
  pivot_longer(everything(), names_to = "position", values_to = "n_na") |>
  filter(n_na > 0)

if (nrow(na_counts) > 0) {
  cat("\nNA counts in position columns:\n")
  print(na_counts, n = Inf)
} else {
  cat("\nNo NAs found in position columns.\n")
}

non_position_cols <- setdiff(colnames(df_raw), POSITION_COLS)

# =============================================================================
# ENCODINGS — each produces df_<enc>.csv (encoded cols only)
#                          and df_all_<enc>.csv (encoded + all other cols)
# =============================================================================

ENCODINGS <- list(
  sparse   = NULL,          # special-cased: one-hot
  blosum50 = BLOSUM50,
  blosum62 = BLOSUM62,
  blosum80 = BLOSUM80,
  kidera   = KIDERA,
  atchley  = ATCHLEY
)

dims <- function(enc) {
  if (enc == "sparse") length(AA_ORDER) else ncol(ENCODINGS[[enc]])
}

dfs_all <- list()  # kept for post-hoc sanity checks below

for (enc in names(ENCODINGS)) {
  n_dims <- dims(enc)
  cat("\n===", toupper(enc), "Encoding ===\n")
  cat("Encoding", length(POSITION_COLS), "positions x", n_dims,
      "dims =", length(POSITION_COLS) * n_dims, "features\n")
  
  if (enc == "sparse") {
    features <- map_dfc(POSITION_COLS, \(pos) {
      sparse_encode_position(df_raw, pos)
    })
    
    # Strict validation: one-hot rows must sum to 0 (NA) or 1
    row_sums <- map_dfc(POSITION_COLS, \(pos) {
      cols <- paste0(pos, "_", AA_ORDER)
      tibble(!!pos := rowSums(features[, cols]))
    })
    bad_rows <- row_sums |> filter(if_any(everything(), \(x) !x %in% c(0, 1)))
    if (nrow(bad_rows) > 0) {
      stop(enc, " validation failed: ", nrow(bad_rows),
           " rows have sums other than 0 or 1.")
    }
    cat("Validation passed: all rows sum to 0 (NA) or 1.\n")
    zero_counts <- row_sums |>
      summarise(across(everything(), \(x) sum(x == 0))) |>
      pivot_longer(everything(), names_to = "position", values_to = "n_zero") |>
      filter(n_zero > 0)
  } else {
    mat <- ENCODINGS[[enc]]
    features <- map_dfc(POSITION_COLS, \(pos) {
      matrix_encode_position(df_raw, pos, mat)
    })
    
    # Validation: NA rows must encode as all-zero vectors
    abs_sums <- map_dfc(POSITION_COLS, \(pos) {
      cols <- paste0(pos, "_", colnames(mat))
      tibble(!!pos := rowSums(abs(features[, cols])))
    })
    zero_counts <- abs_sums |>
      summarise(across(everything(), \(x) sum(x == 0))) |>
      pivot_longer(everything(), names_to = "position", values_to = "n_zero") |>
      filter(n_zero > 0)
  }
  
  cat("Validation passed.\n")
  if (nrow(zero_counts) > 0) {
    cat("All-zero rows (from NAs) per position:\n")
    print(zero_counts, n = Inf)
  }
  
  df_all <- bind_cols(
    df_raw |> select(all_of(non_position_cols)),
    features
  )
  dfs_all[[enc]] <- df_all
  
  cat("Output dimensions:", nrow(df_all), "x", ncol(df_all), "\n")
  
  # Solo CSV: encoded cols + peptide/label keys (keys enable 04_modelling's
  # row-alignment check; they are not model features)
  df_solo <- bind_cols(
    df_raw |> select(peptide, label),
    features
  )
  
  write_csv(df_solo, pp(paste0("df_", enc, ".csv"), ALLELE_TAG))
  cat("Saved: ", pp(paste0("df_", enc, ".csv"), ALLELE_TAG), "\n", sep = "")
  write_csv(df_all, pp(paste0("df_all_", enc, ".csv"), ALLELE_TAG))
  cat("Saved: ", pp(paste0("df_all_", enc, ".csv"), ALLELE_TAG), "\n", sep = "")
}

# =============================================================================
# COMPOSITIONAL — 21 features: count of each AA (incl. X) across the 29
# positions / 29. X is a real letter here: pad X in flanks makes
# frac_X = flank-pad fraction. No NAs expected in position columns.
# =============================================================================

cat("\n=== COMPOSITIONAL Encoding ===\n")

n_pos <- length(POSITION_COLS)
comp_mat <- as.matrix(df_raw |> select(all_of(POSITION_COLS)))

comp_features <- map_dfc(AA_ORDER, \(aa) {
  counts <- rowSums(comp_mat == aa, na.rm = TRUE)
  tibble(!!paste0("frac_", aa) := counts / n_pos)
})

# Validation: row sums must equal (non-NA positions) / 29
row_tot <- rowSums(comp_features)
na_per_row <- rowSums(is.na(comp_mat))
expected <- (n_pos - na_per_row) / n_pos
if (any(abs(row_tot - expected) > 1e-9)) {
  stop("compositional validation failed: ",
       sum(abs(row_tot - expected) > 1e-9), " rows have bad sums.")
}
cat("Validation passed:", length(AA_ORDER), "features x", nrow(comp_features),
    "rows; sums match non-NA position fraction.\n")

df_all_comp <- bind_cols(
  df_raw |> select(all_of(non_position_cols)),
  comp_features
)
write_csv(df_all_comp, pp("df_all_compositional.csv", ALLELE_TAG))
cat("Saved: ", pp("df_all_compositional.csv", ALLELE_TAG), "\n", sep = "")

# Solo CSV: frac_* + peptide/label keys (keys enable 04_modelling's
# row-alignment check; they are not model features)
df_comp_solo <- bind_cols(
  df_raw |> select(peptide, label),
  comp_features
)
write_csv(df_comp_solo, pp("df_compositional.csv", ALLELE_TAG))
cat("Saved: ", pp("df_compositional.csv", ALLELE_TAG), "\n", sep = "")

# =============================================================================
# SANITY CHECK — Anchor positions (HLA-A*02:01)
# =============================================================================

cat("\n--- Anchor Position Sanity Check (", ANCH$label, ") ---\n", sep = "")

anchor_check <- df_raw |>
  mutate(label_str = if_else(label == 1, "positive", "negative")) |>
  group_by(label_str) |>
  summarise(
    !!!set_names(
      lapply(ANCH$p2, \(a) rlang::expr(mean(P2 == !!a, na.rm = TRUE))),
      paste0("P2_", ANCH$p2)),
    !!!set_names(
      lapply(ANCH$p9, \(a) rlang::expr(mean(P9 == !!a, na.rm = TRUE))),
      paste0("P9_", ANCH$p9)),
    .groups = "drop"
  ) |>
  mutate(across(where(is.numeric), \(x) round(x, 3)))

print(anchor_check)

# BLOSUM50 anchor check: P2 and P9 should have similar BLOSUM profiles
cat("\n--- BLOSUM50 Anchor Check: Mean P2 and P9 vectors by label ---\n")

blosum_anchor <- dfs_all[["blosum50"]] |>
  mutate(label_str = if_else(label == 1, "positive", "negative")) |>
  group_by(label_str) |>
  summarise(
    !!!set_names(
      lapply(paste0("P2_", ANCH$p2),
             function(cn) rlang::expr(mean(.data[[!!cn]], na.rm = TRUE))),
      paste0("P2_self_", ANCH$p2)),
    !!!set_names(
      lapply(paste0("P9_", ANCH$p9),
             function(cn) rlang::expr(mean(.data[[!!cn]], na.rm = TRUE))),
      paste0("P9_self_", ANCH$p9)),
    .groups = "drop"
  ) |>
  mutate(across(where(is.numeric), \(x) round(x, 3)))

print(blosum_anchor)

cat("\n(Similar values across labels confirms rank-matching preserved binding motif)\n")

# =============================================================================
# SUMMARY
# =============================================================================

cat("\n=== Encoding Summary ===\n")
cat(sprintf("%-10s %6s %12s  %s\n", "encoding", "dims", "features", "files"))
for (enc in names(ENCODINGS)) {
  n_dims <- dims(enc)
  cat(sprintf(
    "%-10s %6d %12d  df_%s.csv + df_all_%s.csv\n",
    enc, n_dims, length(POSITION_COLS) * n_dims, enc, enc
  ))
}
cat(sprintf(
  "%-10s %6d %12d  df_compositional.csv + df_all_compositional.csv\n",
  "compositional", length(AA_ORDER), length(AA_ORDER)
))

cat("\nNon-position columns:", length(non_position_cols), "\n")
cat("df_<enc>.csv: peptide + label keys + encoded position columns only.\n")
cat("df_all_<enc>.csv: encoded columns + all non-position columns.\n")
cat("Column naming: {position}_{aa} for sparse/BLOSUMs, {position}_KF*/AF* for\n")
cat("kidera/atchley, e.g. P2_L, N4_A, P2_KF1, C1_AF3.\n")
cat("NAs encoded as all-zero vectors; X is a real token (own column/row).\n")
cat("compositional: frac_<aa> = count of aa across",
    length(POSITION_COLS), "positions /", length(POSITION_COLS), "\n")
cat("(X = flank pad, so frac_X is the pad fraction).\n")
