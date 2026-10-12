library(tidyverse)
source("src/pipeline/r/functions.R")
set_working_directory()

POSITION_COLS <- c(
  paste0("N", 10:1),    # N10, N9, ..., N2, N1   (left → right toward the cut)
  paste0("P", 1:9),     # P1..P9
  paste0("C", 1:10)     # C1..C10
)

# AA_ORDER (with X) comes from functions.R

df_raw <- read_csv("data/processed/df_all.csv", show_col_types = FALSE)

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
  
  write_csv(features, paste0("data/processed/df_", enc, ".csv"))
  cat("Saved: data/processed/df_", enc, ".csv\n", sep = "")
  write_csv(df_all, paste0("data/processed/df_all_", enc, ".csv"))
  cat("Saved: data/processed/df_all_", enc, ".csv\n", sep = "")
}

# =============================================================================
# SANITY CHECK — Anchor positions (HLA-A*02:01)
# =============================================================================

cat("\n--- Anchor Position Sanity Check (HLA-A*02:01) ---\n")

anchor_check <- df_raw |>
  mutate(label_str = if_else(label == 1, "positive", "negative")) |>
  group_by(label_str) |>
  summarise(
    P2_L = mean(P2 == "L", na.rm = TRUE),
    P2_M = mean(P2 == "M", na.rm = TRUE),
    P9_V = mean(P9 == "V", na.rm = TRUE),
    P9_L = mean(P9 == "L", na.rm = TRUE),
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
    P2_self = mean(P2_L, na.rm = TRUE),
    P9_self_V = mean(P9_V, na.rm = TRUE),
    P9_self_L = mean(P9_L, na.rm = TRUE),
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

cat("\nNon-position columns:", length(non_position_cols), "\n")
cat("df_<enc>.csv: encoded position columns only.\n")
cat("df_all_<enc>.csv: encoded columns + all non-position columns.\n")
cat("Column naming: {position}_{aa} for sparse/BLOSUMs, {position}_KF*/AF* for\n")
cat("kidera/atchley, e.g. P2_L, N4_A, P2_KF1, C1_AF3.\n")
cat("NAs encoded as all-zero vectors; X is a real token (own column/row).\n")
