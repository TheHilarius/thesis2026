library(tidyverse)
source("src/functions.R")
set_working_directory()

POSITION_COLS <- c(
  paste0("N", 4:1),
  paste0("P", 1:9),
  paste0("C", 1:4)
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

sparse_encode_position <- function(df, pos_col, alphabet = AMINO_ACIDS) {
  values <- df[[pos_col]]
  
  unexpected <- setdiff(na.omit(unique(values)), alphabet)
  if (length(unexpected) > 0) {
    warning(
      "Position '", pos_col, "' contains unexpected values: ",
      paste(unexpected, collapse = ", "),
      " — these will be treated as NA (all zeros)."
    )
    values[values %in% unexpected] <- NA
  }
  
  map_dfc(alphabet, \(aa) {
    tibble(!!paste0(pos_col, "_", aa) := as.integer(!is.na(values) & values == aa))
  })
}

cat("\nEncoding", length(POSITION_COLS), "position columns x",
    length(AMINO_ACIDS), "levels =",
    length(POSITION_COLS) * length(AMINO_ACIDS), "new features\n")

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
  stop("Encoding validation failed: ", nrow(bad_rows),
       " rows have sums other than 0 or 1.")
} else {
  zero_counts <- row_sums |>
    summarise(across(everything(), \(x) sum(x == 0))) |>
    pivot_longer(everything(), names_to = "position", values_to = "n_zero") |>
    filter(n_zero > 0)
  
  cat("Encoding validation passed: all rows sum to 0 (NA) or 1.\n")
  if (nrow(zero_counts) > 0) {
    cat("All-zero rows (from NAs) per position:\n")
    print(zero_counts, n = Inf)
  }
}

non_position_cols <- setdiff(colnames(df_raw), POSITION_COLS)

df_sparse <- bind_cols(
  df_raw |> select(all_of(non_position_cols)),
  sparse_features
)

cat("\nOutput dimensions:", nrow(df_sparse), "x", ncol(df_sparse), "\n")
cat("  Original non-position columns:", length(non_position_cols), "\n")
cat("  New sparse-encoded columns:   ", ncol(sparse_features), "\n")

write_csv(df_sparse, "data/processed/df_all_sparse.csv")
cat("\nSaved: data/processed/df_all_sparse.csv\n")

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