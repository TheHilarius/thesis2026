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
    cflank_end   = pep_end   + nchar(c_flank)
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

# Deduplicate proteins with duplicate residue entries
dup_counts <- sapply(nsp3_split, function(df) sum(duplicated(df$n)))
n_dup_proteins <- sum(dup_counts > 0)

if (n_dup_proteins > 0) {
  cat("WARNING:", n_dup_proteins, "protein(s) have duplicate residue numbers.\n")
  cat("  Affected:", paste(names(dup_counts[dup_counts > 0]), collapse = ", "), "\n")
  
  nsp3_split <- lapply(nsp3_split, function(df) {
    df[!duplicated(df$n), ]
  })
  
  cat("  Deduplicated. Keeping first occurrence per residue.\n")
} else {
  cat("No duplicate residues found.\n")
}

nsp3_split <- lapply(nsp3_split, function(df) {
  df[!duplicated(df$n), ]
})

## Engineering the probability of q8 states (POINT SYSTEM)
# --- Step 1: Extract RSA + disorder (mean pooling is fine for these) ---
rsa_disorder <- df_peptides |>
  mutate(
    feats = pmap(
      list(
        uniprot_id,
        nflank_start, nflank_end,
        pep_start,    pep_end,
        cflank_start, cflank_end
      ),
      extract_nsp3_windows,
      nsp3_split = nsp3_split,
      .progress  = TRUE
    )
  ) |>
  unnest(feats)
cat("Extracting RSA and disorder features...\n")

cat("RSA/disorder done. Rows:", nrow(rsa_disorder), "\n")

# --- Step 2: Extract Q8 structural points (full-context method) ---
cat("Step 2: Computing Q8 structural points (full-context with transitions)...\n")

q8_features <- df_peptides |>
  mutate(
    q8 = pmap(
      list(
        uniprot_id,
        nflank_start, nflank_end,
        pep_start,    pep_end,
        cflank_start, cflank_end
      ),
      extract_nsp3_q8_features,
      nsp3_split = nsp3_split,
      .progress  = TRUE
    )
  ) |>
  unnest(q8) |>
  select(
    peptide, n_flank, c_flank, full_context, uniprot_id, pep_start, pep_end,
    starts_with("q8point_"), starts_with("q8trans_")
  )

cat("  Q8 points done. Rows:", nrow(q8_features), "\n")

# --- Step 3: Combine RSA/disorder + Q8 points ---
cat("Step 3: Combining features...\n")

netsurfp_features <- rsa_disorder |>
  left_join(q8_features,
            by = c("peptide", "n_flank", "c_flank", "full_context",
                   "uniprot_id", "pep_start", "pep_end"))

cat("  Combined NSP3 features:", nrow(netsurfp_features), "rows,",
    ncol(netsurfp_features), "columns\n")

# --- Step 4: Join back to main dataset and clean up old Q8 columns ---
cat("Step 4: Joining to main dataset...\n")

df_final_nsp3 <- df_raw |>
  # Remove old Q8 features if present from previous runs
  select(-any_of(c(
    paste0("frac_q8_", c("G","H","I","B","E","S","T","C"), "_peptide"),
    paste0("frac_q8_", c("G","H","I","B","E","S","T","C"), "_nflank"),
    paste0("frac_q8_", c("G","H","I","B","E","S","T","C"), "_cflank"),
    paste0("mean_p_q8_", c("G","H","I","B","E","S","T","C"), "_peptide"),
    paste0("mean_p_q8_", c("G","H","I","B","E","S","T","C"), "_nflank"),
    paste0("mean_p_q8_", c("G","H","I","B","E","S","T","C"), "_cflank"),
    # Also remove old RSA/disorder if they existed in df_raw
    "mean_rsa_peptide", "mean_disorder_peptide",
    "mean_rsa_nflank", "mean_disorder_nflank",
    "mean_rsa_cflank", "mean_disorder_cflank"
  ))) |>
  left_join(netsurfp_features,
            by = c("peptide", "uniprot_id",
                   "n_flank", "c_flank", "full_context",
                   "start" = "pep_start", "end" = "pep_end"))

cat("  Final dataset:", nrow(df_final_nsp3), "rows,", ncol(df_final_nsp3), "columns\n")

# ============================================================================
# Q8 FEATURE ANALYSIS
# ============================================================================
cat("\n")
cat("=" |> strrep(70), "\n")
cat("Q8 FEATURE ANALYSIS\n")
cat("=" |> strrep(70), "\n\n")

q8_cols <- names(df_final_nsp3)[grepl("^q8point_|^q8trans_", names(df_final_nsp3))]

# --- Overall prevalence ---
cat("Overall feature prevalence:\n")
overall_prev <- df_final_nsp3 |>
  select(all_of(q8_cols)) |>
  summarise(across(everything(), ~mean(.x, na.rm = TRUE))) |>
  pivot_longer(everything(), names_to = "feature", values_to = "prevalence") |>
  arrange(desc(prevalence))

overall_prev |>
  mutate(prevalence = round(prevalence * 100, 1)) |>
  print(n = 40)

# --- Per-label comparison ---
cat("\nPer-label comparison:\n")
q8_by_label <- df_final_nsp3 |>
  select(label, all_of(q8_cols)) |>
  group_by(label) |>
  summarise(across(everything(), ~mean(.x, na.rm = TRUE)), .groups = "drop") |>
  pivot_longer(-label, names_to = "feature", values_to = "fraction") |>
  pivot_wider(names_from = label, values_from = fraction, names_prefix = "label_") |>
  mutate(
    diff = label_1 - label_0,
    feature_type = if_else(grepl("^q8point_", feature), "point", "transition"),
    state = str_extract(feature, "(?<=_)[A-Z](?=_)"),
    window = str_extract(feature, "(peptide|nflank|cflank|n_pep|pep_c)$")
  )

# --- Chi-squared tests ---
cat("Running chi-squared tests...\n")
q8_stats <- map_dfr(q8_cols, function(col) {
  tbl <- table(
    label   = df_final_nsp3$label,
    feature = df_final_nsp3[[col]]
  )
  if (ncol(tbl) < 2 || nrow(tbl) < 2) {
    return(tibble(feature = col, chi_sq = NA_real_, p_value = NA_real_, cramers_v = NA_real_))
  }
  test <- chisq.test(tbl, correct = FALSE)
  n <- sum(tbl)
  tibble(
    feature   = col,
    chi_sq    = as.numeric(test$statistic),
    p_value   = test$p.value,
    cramers_v = as.numeric(sqrt(test$statistic / n))
  )
})

# --- Combine and apply BH correction ---
q8_full <- q8_by_label |>
  left_join(q8_stats, by = "feature") |>
  mutate(
    p_adjusted = p.adjust(p_value, method = "BH"),
    sig_adj = case_when(
      is.na(p_adjusted) ~ "",
      p_adjusted < 0.001 ~ "***",
      p_adjusted < 0.01  ~ "**",
      p_adjusted < 0.05  ~ "*",
      TRUE ~ "ns"
    )
  ) |>
  arrange(p_value)

cat("\n=== Multiple Testing Correction (Benjamini-Hochberg) ===\n")
cat("Features significant before correction (p < 0.05):",
    sum(q8_full$p_value < 0.05, na.rm = TRUE), "\n")
cat("Features significant after correction (q < 0.05): ",
    sum(q8_full$p_adjusted < 0.05, na.rm = TRUE), "\n\n")

q8_full |>
  mutate(
    across(c(label_0, label_1, diff), ~round(.x, 4)),
    p_raw = formatC(p_value, format = "e", digits = 2),
    p_adj = formatC(p_adjusted, format = "e", digits = 2),
    cramers_v = round(cramers_v, 4)
  ) |>
  select(feature, label_0, label_1, diff, cramers_v, p_raw, p_adj, sig_adj, state, window) |>
  print(n = 40)

# --- Summaries ---
cat("\n=== Summary: Mean absolute difference by Q8 state ===\n")
q8_full |>
  group_by(state) |>
  summarise(
    mean_abs_diff = mean(abs(diff)),
    direction = if_else(mean(diff) > 0, "more in presented", "more in non-presented"),
    n_sig_adj = sum(p_adjusted < 0.05, na.rm = TRUE),
    n_total = n(),
    .groups = "drop"
  ) |>
  arrange(desc(mean_abs_diff)) |>
  print()

cat("\n=== Summary: Mean absolute difference by feature type ===\n")
q8_full |>
  group_by(feature_type) |>
  summarise(mean_abs_diff = mean(abs(diff)), .groups = "drop") |>
  print()

cat("\n=== Summary: Mean absolute difference by window ===\n")
q8_full |>
  group_by(window) |>
  summarise(mean_abs_diff = mean(abs(diff)), .groups = "drop") |>
  arrange(desc(mean_abs_diff)) |>
  print()

# ============================================================================
# VISUALIZATION
# ============================================================================

# --- Plot 1: Bar chart of effect sizes ---
plot_data <- q8_full |>
  filter(!is.na(p_adjusted), abs(diff) > 0.001) |>
  mutate(
    sig = p_adjusted < 0.05,
    feature = fct_reorder(feature, abs(diff)),
    direction = if_else(diff > 0, "More in presented", "More in non-presented")
  )

p1 <- ggplot(plot_data, aes(x = diff, y = feature, fill = direction)) +
  geom_col(aes(alpha = sig)) +
  scale_fill_manual(values = c("More in presented" = "#2166AC",
                               "More in non-presented" = "#B2182B")) +
  scale_alpha_manual(values = c("TRUE" = 1, "FALSE" = 0.3),
                     labels = c("TRUE" = "q < 0.05 (BH)", "FALSE" = "ns"),
                     name = "Significance") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  labs(
    title = "Q8 Structural Feature Differences: Presented vs Non-presented",
    subtitle = "Binary point system with biological minimum-length runs (BH-corrected)",
    x = "Difference in prevalence (presented − non-presented)",
    y = NULL, fill = "Direction"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

print(p1)
ggsave("results/q8_feature_differences.png", p1, width = 10, height = 10, dpi = 150)

# --- Plot 2: Heatmap — points by state × window ---
heatmap_data <- q8_full |>
  filter(feature_type == "point") |>
  mutate(
    state_label = case_when(
      state == "H" ~ "\u03b1-helix (H)", state == "G" ~ "3\u2081\u2080-helix (G)",
      state == "I" ~ "\u03c0-helix (I)", state == "E" ~ "\u03b2-strand (E)",
      state == "B" ~ "\u03b2-bridge (B)", state == "T" ~ "Turn (T)",
      state == "S" ~ "Bend (S)", state == "C" ~ "Coil (C)"
    ),
    window = factor(window, levels = c("nflank", "peptide", "cflank")),
    sig_label = if_else(sig_adj %in% c("***", "**", "*"), sig_adj, "")
  )

p2 <- ggplot(heatmap_data, aes(x = window, y = state_label, fill = diff)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = paste0(round(diff * 100, 1), "%\n", sig_label)), size = 3.5) +
  scale_fill_gradient2(
    low = "#B2182B", mid = "white", high = "#2166AC", midpoint = 0,
    name = "\u0394 prevalence\n(pos \u2212 neg)", labels = scales::percent
  ) +
  labs(
    title = "Q8 Structural Points: Presented vs Non-presented by Window",
    subtitle = "Positive (blue) = more common in presented peptides (BH-corrected)",
    x = "Window", y = "Secondary Structure"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid = element_blank())

print(p2)
ggsave("results/q8_point_heatmap.png", p2, width = 7, height = 6, dpi = 150)

# --- Plot 3: Heatmap — transitions by state × boundary ---
trans_data <- q8_full |>
  filter(feature_type == "transition") |>
  mutate(
    state_label = case_when(
      state == "H" ~ "\u03b1-helix (H)", state == "G" ~ "3\u2081\u2080-helix (G)",
      state == "I" ~ "\u03c0-helix (I)", state == "E" ~ "\u03b2-strand (E)",
      state == "B" ~ "\u03b2-bridge (B)", state == "T" ~ "Turn (T)",
      state == "S" ~ "Bend (S)", state == "C" ~ "Coil (C)"
    ),
    boundary = case_when(
      window == "n_pep" ~ "N-flank \u2192 Peptide",
      window == "pep_c" ~ "Peptide \u2192 C-flank"
    ),
    sig_label = if_else(sig_adj %in% c("***", "**", "*"), sig_adj, "")
  )

p3 <- ggplot(trans_data, aes(x = boundary, y = state_label, fill = diff)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = paste0(round(diff * 100, 1), "%\n", sig_label)), size = 3.5) +
  scale_fill_gradient2(
    low = "#B2182B", mid = "white", high = "#2166AC", midpoint = 0,
    name = "\u0394 prevalence\n(pos \u2212 neg)", labels = scales::percent
  ) +
  labs(
    title = "Q8 Structural Transitions at Cleavage Boundaries",
    subtitle = "Does a continuous structure span the proteasomal cleavage site? (BH-corrected)",
    x = "Boundary", y = "Secondary Structure"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid = element_blank())

print(p3)
ggsave("results/q8_transition_heatmap.png", p3, width = 7, height = 6, dpi = 150)

# ============================================================================
# WRITE LOG FILE
# ============================================================================

log_file <- paste0("results/q8_feature_analysis_log_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt")
sink(log_file)

cat("=" |> strrep(70), "\n")
cat("Q8 STRUCTURAL POINT SYSTEM — FEATURE ANALYSIS LOG\n")
cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("=" |> strrep(70), "\n\n")

cat("INPUT\n")
cat("  Dataset:", nrow(df_final_nsp3), "peptides\n")
cat("  Presented (label=1):", sum(df_final_nsp3$label == 1, na.rm = TRUE), "\n")
cat("  Non-presented (label=0):", sum(df_final_nsp3$label == 0, na.rm = TRUE), "\n")
cat("  Peptides with Q8 data:", sum(!is.na(df_final_nsp3$q8point_H_peptide)), "\n")
cat("  Peptides missing Q8:", sum(is.na(df_final_nsp3$q8point_H_peptide)), "\n\n")

cat("METHOD\n")
cat("  Feature type: Binary Q8 structural points\n")
cat("  Approach: Run-length encoding on argmax of Q8 probabilities\n")
cat("  Context: Full (nflank + peptide + cflank computed as single block)\n")
cat("  Rescue threshold:", RESCUE_THRESHOLD, "\n")
cat("  Multiple testing: Benjamini-Hochberg\n")
cat("  Minimum consecutive residues per state:\n")
for (s in Q8_STATES) cat(sprintf("    %s: %d\n", s, Q8_MIN_RUNS[s]))

cat("\nOVERALL FEATURE PREVALENCE\n")
cat(sprintf("  %-25s %s\n", "Feature", "Prevalence"))
cat("  ", strrep("-", 40), "\n")
for (i in seq_len(nrow(overall_prev))) {
  cat(sprintf("  %-25s %.1f%%\n", overall_prev$feature[i], overall_prev$prevalence[i] * 100))
}
cat("\n\nPER-LABEL COMPARISON (BH-adjusted)\n")
cat(sprintf("  %-25s %8s %8s %8s %10s %10s %s\n",
            "Feature", "Neg", "Pos", "Diff", "p_raw", "p_adj", "Sig"))
cat("  ", strrep("-", 85), "\n")
for (i in seq_len(nrow(q8_full))) {
  r <- q8_full[i, ]
  cat(sprintf("  %-25s %7.4f  %7.4f  %+7.4f  %10s  %10s  %s\n",
              r$feature, r$label_0, r$label_1, r$diff,
              formatC(r$p_value, format = "e", digits = 2),
              formatC(r$p_adjusted, format = "e", digits = 2),
              r$sig_adj))
}

cat("\n\nSUMMARY BY Q8 STATE\n")
state_summary <- q8_full |>
  group_by(state) |>
  summarise(
    mean_abs_diff = mean(abs(diff)),
    direction = if_else(mean(diff) > 0, "more in presented", "more in non-presented"),
    n_sig = sum(p_adjusted < 0.05, na.rm = TRUE),
    n_total = n(),
    .groups = "drop"
  ) |>
  arrange(desc(mean_abs_diff))
for (i in seq_len(nrow(state_summary))) {
  r <- state_summary[i, ]
  cat(sprintf("  %s: mean |diff| = %.4f, %s, %d/%d significant (BH q<0.05)\n",
              r$state, r$mean_abs_diff, r$direction, r$n_sig, r$n_total))
}

cat("\n\nBIOLOGICAL INTERPRETATION\n")
cat("  1. HELIX ENRICHMENT: α-helix is the only structure enriched in presented\n")
cat("     peptides. Strongest signal: helix at N-flank→peptide boundary (+5.0%).\n")
cat("  2. COIL DEPLETION: Coil depleted in presented peptides, especially in\n")
cat("     peptide window (-4.5%) and N-boundary (-3.0%).\n")
cat("  3. HELIX-COIL ANTI-CORRELATION: Signal is specifically helix↔coil.\n")
cat("     Strand, turn, bend do not fill the gap — presented peptides come\n")
cat("     from well-folded helical regions, not just 'any structure'.\n")
cat("  4. N-TERMINAL ASYMMETRY: N-flank and N-boundary show larger effects\n")
cat("     than C-flank. C-flank features mostly non-significant.\n")
cat("  5. FEATURE HIERARCHY: peptide > nflank > N-boundary > C-boundary > cflank.\n")

cat("\n\nFEATURES TO CONSIDER DROPPING (near-zero variance)\n")
low_var <- overall_prev |> filter(prevalence < 0.01 | prevalence > 0.99)
if (nrow(low_var) > 0) {
  for (i in seq_len(nrow(low_var))) {
    cat(sprintf("  %s (%.2f%%)\n", low_var$feature[i], low_var$prevalence[i] * 100))
  }
} else {
  cat("  None below 1%% or above 99%%\n")
}

cat("\n", "=" |> strrep(70), "\n")
cat("END OF LOG\n")
sink()
cat("Log written to:", log_file, "\n")

# ============================================================================
# SAVE FINAL DATASET
# ============================================================================

write_csv(df_final_nsp3, "data/processed/epitopes_pos_and_neg_features_with_nsp3.csv")
cat("\nSaved: data/processed/epitopes_pos_and_neg_features_with_nsp3.csv\n")
cat("Final dimensions:", nrow(df_final_nsp3), "x", ncol(df_final_nsp3), "\n")

