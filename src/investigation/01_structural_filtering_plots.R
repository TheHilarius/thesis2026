#!/usr/bin/env Rscript
# ==============================================================================
# 01_structural_filtering_plots.R
# Investigation plots for structural data filtering pipeline.
#
# Generates:
#   1. Euler/Venn diagram — protein-level filter overlap
#   2. Stacked bar chart — peptide removal by filter stage
#   3. Missing data comparison — old vs new dataset
#
# Output: results/figures/filtering/
# ==============================================================================

library(eulerr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT <- getwd()
DATA_DIR <- file.path(ROOT, "data", "processed")
OUT_DIR <- file.path(ROOT, "results", "figures", "filtering")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("Output directory:", OUT_DIR, "\n")

# ==============================================================================
# 1. PROTEIN-LEVEL EULER / VENN DIAGRAM
# ==============================================================================
#
# Sets from exclusion_ledger.csv:
#   A = NetSurfP Length Filter (too_short OR too_long)
#   B = AlphaFold No PDB (no_pdb)
#   C = AF Range Exceeded (out_of_range)
#
# Note: A ⊂ B (all length-filtered proteins also lack a PDB)
# ==============================================================================

cat("\n=== Plot 1: Protein-Level Euler Diagram ===\n")

ledger <- read_csv(file.path(DATA_DIR, "exclusion_ledger.csv"), show_col_types = FALSE)

# Count each set (among excluded proteins only)
excluded <- ledger %>% filter(excluded == TRUE)

n_netsurfp <- excluded %>%
  filter(too_short == TRUE | too_long == TRUE) %>%
  nrow()

n_no_pdb <- excluded %>%
  filter(no_pdb == TRUE) %>%
  nrow()

n_oor <- excluded %>%
  filter(out_of_range == TRUE) %>%
  nrow()

# Overlaps (all among excluded)
n_netsurfp_no_pdb <- excluded %>%
  filter((too_short == TRUE | too_long == TRUE) & no_pdb == TRUE) %>%
  nrow()

n_netsurfp_oor <- excluded %>%
  filter((too_short == TRUE | too_long == TRUE) & out_of_range == TRUE) %>%
  nrow()

n_no_pdb_oor <- excluded %>%
  filter(no_pdb == TRUE & out_of_range == TRUE) %>%
  nrow()

n_all_three <- excluded %>%
  filter((too_short == TRUE | too_long == TRUE) & no_pdb == TRUE & out_of_range == TRUE) %>%
  nrow()

cat(sprintf("  NetSurfP Length: %d proteins\n", n_netsurfp))
cat(sprintf("  AlphaFold No PDB: %d proteins\n", n_no_pdb))
cat(sprintf("  AF Range Exceeded: %d proteins\n", n_oor))
cat(sprintf("  Overlap (Length ∩ NoPDB): %d\n", n_netsurfp_no_pdb))
cat(sprintf("  Total excluded: %d\n", nrow(excluded)))

# Euler diagram input — named vector with set intersections
# Format: "A&B" for overlap, "A" for A-only, etc.
# Since A ⊂ B: A∩B = all of A, A∩C = 0, B∩C = 0, A∩B∩C = 0
euler_input <- c(
  "NetSurfP Length"          = n_netsurfp - n_netsurfp_no_pdb,  # A-only (should be 0 since A⊂B)
  "AlphaFold No PDB"         = n_no_pdb - n_netsurfp_no_pdb,    # B-only
  "AF Range Exceeded"        = n_oor,                            # C-only
  "NetSurfP Length&AlphaFold No PDB" = n_netsurfp_no_pdb,        # A∩B
  "NetSurfP Length&AF Range Exceeded" = n_netsurfp_oor,           # A∩C
  "AlphaFold No PDB&AF Range Exceeded" = n_no_pdb_oor,           # B∩C
  "NetSurfP Length&AlphaFold No PDB&AF Range Exceeded" = n_all_three  # A∩B∩C
)

# Remove zero entries for cleaner plot
euler_input <- euler_input[euler_input > 0]

fit <- euler(euler_input)

png(file.path(OUT_DIR, "filtering_venn_protein.png"),
    width = 2400, height = 1800, res = 300)

plot(fit,
     quantities = list(cex = 1.4, font = 2),
     labels = list(cex = 1.1, font = 3),
     fills = list(fill = c("#e74c3c", "#3498db", "#f39c12"), alpha = 0.65),
     edges = list(col = "black", lwd = 2),
     main = "Protein-Level Filter Overlap")

dev.off()
cat("Saved: filtering_venn_protein.png\n")

# ==============================================================================
# 2. PEPTIDE-LEVEL STACKED BAR CHART
# ==============================================================================
#
# Shows peptides removed at each filtering stage.
# Data from sankey_counts.csv (cumulative cascade).
# ==============================================================================

cat("\n=== Plot 2: Peptide-Level Stacked Bar Chart ===\n")

sankey <- read_csv(file.path(DATA_DIR, "sankey_counts.csv"), show_col_types = FALSE)

# Define the filtering stages and their removal counts
# These are the structural/preprocessing filters that remove peptides
filter_stages <- tibble(
  stage = factor(c(
    "PTM Filter",
    "Non-9mer",
    "Missing UniProt",
    "NetSurfP Length",
    "AlphaFold No PDB"
  ), levels = rev(c(
    "PTM Filter",
    "Non-9mer",
    "Missing UniProt",
    "NetSurfP Length",
    "AlphaFold No PDB"
  ))),
  removed = c(
    sankey$count[sankey$stage == "ptm"],
    sankey$count[sankey$stage == "non_9mer"],
    sankey$count[sankey$stage == "na_uniprot"],
    sankey$count[sankey$stage == "nsp3_length"],
    sankey$count[sankey$stage == "alphafold"]
  ),
  category = factor(c(
    "Preprocessing",
    "Preprocessing",
    "Preprocessing",
    "Structural",
    "Structural"
  ), levels = c("Preprocessing", "Structural"))
)

cat("  Filter stage removals:\n")
for (i in seq_len(nrow(filter_stages))) {
  cat(sprintf("    %s: %s peptides\n",
              as.character(filter_stages$stage[i]),
              format(filter_stages$removed[i], big.mark = ",")))
}

p_bar <- ggplot(filter_stages, aes(x = removed, y = stage, fill = category)) +
  geom_col(width = 0.7, alpha = 0.85) +
  geom_text(aes(label = format(removed, big.mark = ",")),
            hjust = -0.1, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("Preprocessing" = "#95a5a6",
                                "Structural" = "#e74c3c"),
                    name = "Filter Type") +
  scale_x_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "Peptides Removed by Filtering Stage",
    subtitle = "From raw assays to verified 9-mers",
    x = "Peptides Removed",
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey40"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top"
  )

ggsave(file.path(OUT_DIR, "filtering_stacked_bar.png"),
       p_bar, width = 10, height = 5, dpi = 300)
cat("Saved: filtering_stacked_bar.png\n")

# ==============================================================================
# 3. MISSING DATA COMPARISON (OLD vs NEW)
# ==============================================================================
#
# Grouped bar chart: structural features × missing count (old vs new).
# ==============================================================================

cat("\n=== Plot 3: Missing Data Comparison ===\n")

df_old <- read_csv(file.path(DATA_DIR, "df_all_old.csv"), show_col_types = FALSE)
df_new <- read_csv(file.path(DATA_DIR, "df_all.csv"), show_col_types = FALSE)

structural_cols <- c("mean_plddt_peptide", "mean_plddt_nflank", "mean_plddt_cflank",
                     "mean_rsa_peptide", "mean_rsa_nflank", "mean_rsa_cflank")

# Friendly labels
feature_labels <- c(
  "mean_plddt_peptide" = "pLDDT Peptide",
  "mean_plddt_nflank"  = "pLDDT N-flank",
  "mean_plddt_cflank"  = "pLDDT C-flank",
  "mean_rsa_peptide"   = "RSA Peptide",
  "mean_rsa_nflank"    = "RSA N-flank",
  "mean_rsa_cflank"    = "RSA C-flank"
)

feature_levels <- rev(c(
  "pLDDT Peptide", "pLDDT N-flank", "pLDDT C-flank",
  "RSA Peptide", "RSA N-flank", "RSA C-flank"
))

missing_df <- tibble(
  feature = factor(rep(feature_labels[structural_cols], 2),
                   levels = feature_levels),
  dataset = rep(c("Before (df_all_old)", "After (df_all)"), each = 6),
  missing = c(
    sapply(structural_cols, function(c) sum(is.na(df_old[[c]]))),
    sapply(structural_cols, function(c) sum(is.na(df_new[[c]])))
  ),
  pct = c(
    sapply(structural_cols, function(c) mean(is.na(df_old[[c]])) * 100),
    sapply(structural_cols, function(c) mean(is.na(df_new[[c]])) * 100)
  )
)

cat("  Missing values comparison:\n")
for (col in structural_cols) {
  n_old <- sum(is.na(df_old[[col]]))
  n_new <- sum(is.na(df_new[[col]]))
  cat(sprintf("    %s: %d → %d (%.1f%% → %.3f%%)\n",
              col, n_old, n_new,
              n_old / nrow(df_old) * 100,
              n_new / nrow(df_new) * 100))
}

p_missing <- ggplot(missing_df, aes(x = missing, y = feature, fill = dataset)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, alpha = 0.85) +
  geom_text(aes(label = ifelse(missing > 0,
                               format(missing, big.mark = ","), "")),
            position = position_dodge(width = 0.8),
            hjust = -0.1, size = 3, fontface = "bold") +
  scale_fill_manual(values = c("Before (df_all_old)" = "#e74c3c",
                                "After (df_all)" = "#2ecc71"),
                    name = "Dataset") +
  scale_x_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.2))) +
  labs(
    title = "Missing Structural Features: Before vs After Filtering",
    subtitle = sprintf("df_all_old: %s rows  |  df_all: %s rows  |  Removed: %s rows",
                       format(nrow(df_old), big.mark = ","),
                       format(nrow(df_new), big.mark = ","),
                       format(nrow(df_old) - nrow(df_new), big.mark = ",")),
    x = "Missing Values",
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey40"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top"
  )

ggsave(file.path(OUT_DIR, "filtering_missing_comparison.png"),
       p_missing, width = 10, height = 5, dpi = 300)
cat("Saved: filtering_missing_comparison.png\n")

# ==============================================================================
# Summary
# ==============================================================================

cat("\n=== Done ===\n")
cat("All plots saved to:", OUT_DIR, "\n")
cat("  filtering_venn_protein.png       — Euler diagram of protein filter overlap\n")
cat("  filtering_stacked_bar.png        — Peptide removal by filter stage\n")
cat("  filtering_missing_comparison.png — Missing data before vs after\n")
