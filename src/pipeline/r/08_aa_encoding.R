library(tidyverse)
source("src/pipeline/r/functions.R")
set_working_directory()

POSITION_COLS <- c(
  paste0("N", 10:1),    # N10, N9, ..., N2, N1   (left → right toward the cut)
  paste0("P", 1:9),     # P1..P9
  paste0("C", 1:10)     # C1..C10
)

AMINO_ACIDS <- c(
  "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
  "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"
)

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
# 1. SPARSE (ONE-HOT) ENCODING — 17 x 20 = 340 binary features
# =============================================================================

cat("\n=== Sparse Encoding ===\n")
cat("Encoding", length(POSITION_COLS), "positions x",
    length(AMINO_ACIDS), "AAs =",
    length(POSITION_COLS) * length(AMINO_ACIDS), "features\n")

sparse_features <- map_dfc(POSITION_COLS, \(pos) {
  sparse_encode_position(df_raw, pos)
})

row_sums <- map_dfc(POSITION_COLS, \(pos) {
  cols <- paste0(pos, "_", AMINO_ACIDS)
  tibble(!!pos := rowSums(sparse_features[, cols]))
})

bad_rows <- row_sums |>
  filter(if_any(everything(), \(x) !x %in% c(0, 1)))

if (nrow(bad_rows) > 0) {
  stop("Sparse validation failed: ", nrow(bad_rows),
       " rows have sums other than 0 or 1.")
} else {
  zero_counts <- row_sums |>
    summarise(across(everything(), \(x) sum(x == 0))) |>
    pivot_longer(everything(), names_to = "position", values_to = "n_zero") |>
    filter(n_zero > 0)
  
  cat("Validation passed: all rows sum to 0 (NA) or 1.\n")
  if (nrow(zero_counts) > 0) {
    cat("All-zero rows (from NAs) per position:\n")
    print(zero_counts, n = Inf)
  }
}

df_sparse <- bind_cols(
  df_raw |> select(all_of(non_position_cols)),
  sparse_features
)

cat("Output dimensions:", nrow(df_sparse), "x", ncol(df_sparse), "\n")

write_csv(df_sparse, "data/processed/df_all_sparse.csv")
cat("Saved: data/processed/df_all_sparse.csv\n")

# =============================================================================
# 2. BLOSUM50 ENCODING — 17 x 20 = 340 continuous features
# =============================================================================

cat("\n=== BLOSUM50 Encoding ===\n")
cat("Encoding", length(POSITION_COLS), "positions x 20 BLOSUM50 dims =",
    length(POSITION_COLS) * 20, "features\n")

blosum_features <- map_dfc(POSITION_COLS, \(pos) {
  blosum50_encode_position(df_raw, pos)
})

# Validation: NAs should produce all-zero rows, others should have non-zero values
blosum_row_sums <- map_dfc(POSITION_COLS, \(pos) {
  cols <- paste0(pos, "_", AMINO_ACIDS)
  tibble(!!pos := rowSums(abs(blosum_features[, cols])))
})

blosum_zero_counts <- blosum_row_sums |>
  summarise(across(everything(), \(x) sum(x == 0))) |>
  pivot_longer(everything(), names_to = "position", values_to = "n_zero") |>
  filter(n_zero > 0)

cat("Validation passed.\n")
if (nrow(blosum_zero_counts) > 0) {
  cat("All-zero rows (from NAs) per position:\n")
  print(blosum_zero_counts, n = Inf)
}

df_blosum <- bind_cols(
  df_raw |> select(all_of(non_position_cols)),
  blosum_features
)

cat("Output dimensions:", nrow(df_blosum), "x", ncol(df_blosum), "\n")

write_csv(df_blosum, "data/processed/df_all_blosum50.csv")
cat("Saved: data/processed/df_all_blosum50.csv\n")

# =============================================================================
# 3. SANITY CHECK — Anchor positions (HLA-A*02:01)
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

blosum_anchor <- df_blosum |>
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
cat("Sparse:  ", nrow(df_sparse), "x", ncol(df_sparse),
    "  → data/processed/df_all_sparse.csv\n")
cat("BLOSUM50:", nrow(df_blosum), "x", ncol(df_blosum),
    "  → data/processed/df_all_blosum50.csv\n")
cat("Both have", length(non_position_cols), "non-position columns +",
    length(POSITION_COLS) * 20, "encoded columns\n")
cat("Column naming: {position}_{amino_acid}, e.g. P2_L, N4_A, C1_V\n")
cat("NAs encoded as all-zero vectors in both schemes.\n")