#!/usr/bin/env Rscript
# ==============================================================================
# 01_structural_filtering_plots.R
# Investigation plots for the structural filtering pipeline.
#
# Generates:
#   1. Protein-level overlap plot
#   2. Peptide removal by filter stage
#   3. Missing data before vs after filtering
#   4. Positive peptide structural loss
#   5. Positive peptide removal breakdown
#
# Output: results/figures/filtering/
# ==============================================================================

library(eulerr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)

# File paths.
ROOT <- getwd()
DATA_DIR <- file.path(ROOT, "data", "processed")
OUT_DIR <- file.path(ROOT, "results", "figures", "filtering")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("Output directory:", OUT_DIR, "\n")

# ==============================================================================
# Protein-level overlap plot.
#
# Sets from exclusion_ledger.csv:
#   A = length cutoff (>5000 aa)
#   B = AlphaFold no PDB
#   C = AF range exceeded
#
# The length cutoff is a subset of the no-PDB group.

cat("\n=== Plot 1: Protein-Level Euler Diagram ===\n")

ledger <- read_csv(file.path(DATA_DIR, "exclusion_ledger.csv"), show_col_types = FALSE)

# Count each set among excluded proteins only.
excluded <- ledger |> filter(excluded == TRUE)

n_netsurfp <- excluded |>
  filter(too_long == TRUE) |>
  nrow()

n_no_pdb <- excluded |>
  filter(no_pdb == TRUE) |>
  nrow()

n_oor <- excluded |>
  filter(out_of_range == TRUE) |>
  nrow()

# Overlaps (all among excluded)
n_netsurfp_no_pdb <- excluded |>
  filter(too_long == TRUE & no_pdb == TRUE) |>
  nrow()

n_netsurfp_oor <- excluded |>
  filter(too_long == TRUE & out_of_range == TRUE) |>
  nrow()

n_no_pdb_oor <- excluded |>
  filter(no_pdb == TRUE & out_of_range == TRUE) |>
  nrow()

n_all_three <- excluded |>
  filter(too_long == TRUE & no_pdb == TRUE & out_of_range == TRUE) |>
  nrow()

cat(sprintf("  Length >5000 aa: %d proteins\n", n_netsurfp))
cat(sprintf("  AlphaFold No PDB: %d proteins\n", n_no_pdb))
cat(sprintf("  AF Range Exceeded: %d proteins\n", n_oor))
cat(sprintf("  Overlap (Length ∩ NoPDB): %d\n", n_netsurfp_no_pdb))
cat(sprintf("  Total excluded: %d\n", nrow(excluded)))

# Euler input as set intersections.
# A is a subset of B, so the overlap is the full length-cutoff set.
euler_input <- c(
  "Length >5000 aa" = n_netsurfp - n_netsurfp_no_pdb,
  "AlphaFold No PDB" = n_no_pdb - n_netsurfp_no_pdb,
  "AF Range Exceeded" = n_oor,
  "Length >5000 aa&AlphaFold No PDB" = n_netsurfp_no_pdb,
  "Length >5000 aa&AF Range Exceeded" = n_netsurfp_oor,
  "AlphaFold No PDB&AF Range Exceeded" = n_no_pdb_oor,
  "Length >5000 aa&AlphaFold No PDB&AF Range Exceeded" = n_all_three
)

# Drop zero entries for a cleaner plot.
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

# Protein length comparison by AlphaFold coverage.

cat("\n=== Plot 1b: Protein Length Comparison ===\n")

len_df <- ledger |>
  filter(!is.na(seq_length), seq_length > 0) |>
  mutate(alpha_fold_status = if_else(has_pdb, "Has AlphaFold structure", "No AlphaFold structure"))

len_summary <- len_df |>
  group_by(alpha_fold_status) |>
  summarise(
    n = n(),
    median_len = median(seq_length),
    mean_len = mean(seq_length),
    .groups = "drop"
  )

print(len_summary)

p_len <- ggplot(len_df, aes(x = seq_length, fill = alpha_fold_status, colour = alpha_fold_status)) +
  geom_density(alpha = 0.25, linewidth = 1) +
  scale_x_log10(labels = scales::comma) +
  scale_fill_manual(values = c("Has AlphaFold structure" = "#2ecc71", "No AlphaFold structure" = "#e74c3c")) +
  scale_colour_manual(values = c("Has AlphaFold structure" = "#2ecc71", "No AlphaFold structure" = "#e74c3c")) +
  labs(
    title = "Protein lengths look different when AlphaFold structures are missing",
    subtitle = "Proteins with a structure versus proteins without one, on a log length scale",
    x = "Protein length (log scale)",
    y = "Density",
    fill = NULL,
    colour = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey40"),
    legend.position = "top"
  )

ggsave(file.path(OUT_DIR, "protein_length_alphaFold_comparison.png"),
       p_len, width = 10, height = 5, dpi = 300)
cat("Saved: protein_length_alphaFold_comparison.png\n")

# Peptide removal by filter stage.

cat("\n=== Plot 2: Peptide-Level Stacked Bar Chart ===\n")

sankey <- read_csv(file.path(DATA_DIR, "sankey_counts.csv"), show_col_types = FALSE)

# Define the filter stages and their removal counts.
filter_stages <- tibble(
  stage = factor(c(
    "PTM Filter",
    "Non-9mer",
    "Missing UniProt",
    "Length >5000 aa",
    "AlphaFold No PDB"
  ), levels = rev(c(
    "PTM Filter",
    "Non-9mer",
    "Missing UniProt",
    "Length >5000 aa",
    "AlphaFold No PDB"
  ))),
  removed = c(
    sankey$count[sankey$stage == "ptm"],
    sankey$count[sankey$stage == "non_9mer"],
    sankey$count[sankey$stage == "na_uniprot"],
    sankey$count[sankey$stage == "too_long"],
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

# Missing data before vs after filtering.

cat("\n=== Plot 3: Missing Data Comparison ===\n")

df_old <- read_csv(file.path(DATA_DIR, "df_all_old.csv"), show_col_types = FALSE)
df_new <- read_csv(file.path(DATA_DIR, "df_all.csv"), show_col_types = FALSE)

structural_cols <- c("mean_plddt_peptide", "mean_plddt_nflank", "mean_plddt_cflank",
                     "mean_rsa_peptide", "mean_rsa_nflank", "mean_rsa_cflank")

# Friendly labels.
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
    subtitle = sprintf("Reduced from %s to %s rows (%s removed)",
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
# Positive peptides removed for structural reasons.
# Only positive peptides, since they are hard to replace.

cat("\n=== Plot 4: Positive Peptide Euler Diagram ===\n")

# Identify removed peptides.
old_peptides <- df_old |> pull(peptide) |> unique()
new_peptides <- df_new |> pull(peptide) |> unique()
removed_peptides <- setdiff(old_peptides, new_peptides)

# Keep only removed positive peptides.
df_removed_pos <- df_old |>
  filter(label == 1, peptide %in% removed_peptides)

# Count missing structural features among removed positives.
pos_no_plddt <- df_removed_pos |> filter(is.na(mean_plddt_peptide)) |> pull(peptide) |> unique()
pos_no_rsa <- df_removed_pos |> filter(is.na(mean_rsa_peptide)) |> pull(peptide) |> unique()

n_pos_removed <- df_removed_pos |> pull(peptide) |> unique() |> length()
n_pos_no_plddt <- length(pos_no_plddt)
n_pos_no_rsa <- length(pos_no_rsa)
n_pos_no_both <- length(intersect(pos_no_plddt, pos_no_rsa))
n_pos_plddt_only <- n_pos_no_plddt - n_pos_no_both
n_pos_rsa_only <- n_pos_no_rsa - n_pos_no_both
n_pos_has_structure <- n_pos_removed - length(union(pos_no_plddt, pos_no_rsa))

cat(sprintf("  Total positive peptides removed: %d\n", n_pos_removed))
cat(sprintf("  Missing pLDDT: %d (%.1f%%)\n", n_pos_no_plddt, n_pos_no_plddt / n_pos_removed * 100))
cat(sprintf("  Missing RSA: %d (%.1f%%)\n", n_pos_no_rsa, n_pos_no_rsa / n_pos_removed * 100))
cat(sprintf("  Missing both: %d (%.1f%%)\n", n_pos_no_both, n_pos_no_both / n_pos_removed * 100))
cat(sprintf("  Other removed positives: %d (%.1f%%)\n",
            n_pos_has_structure, n_pos_has_structure / n_pos_removed * 100))

# Euler input.
euler_pos_venn <- c(
  "Missing pLDDT" = n_pos_plddt_only,
  "Missing RSA" = n_pos_rsa_only,
  "Missing pLDDT&Missing RSA" = n_pos_no_both
)
euler_pos_venn <- euler_pos_venn[euler_pos_venn > 0]

fit_pos_venn <- euler(euler_pos_venn)

# Plot with simple styling.
png(file.path(OUT_DIR, "filtering_venn_peptide_positives.png"),
    width = 3000, height = 2400, res = 300)

par(mar = c(1, 1, 5, 1))
plot(fit_pos_venn,
     quantities = list(cex = 1.6, font = 2),
     labels = list(cex = 1.2, font = 3),
     fills = list(fill = c("#e74c3c", "#3498db"), alpha = 0.6),
     edges = list(col = c("#c0392b", "#2980b9"), lwd = 3),
     main = paste0(
        "Positive peptides removed by structural filtering\n",
       "1,782 positives removed: ",
       format(n_pos_no_plddt, big.mark = ","), " lack pLDDT, ",
       format(n_pos_no_rsa, big.mark = ","), " lack RSA, ",
       format(n_pos_no_both, big.mark = ","), " lack both"
     ))

dev.off()
cat("Saved: filtering_venn_peptide_positives.png\n")

# Positive peptide removal breakdown.

cat("\n=== Plot 5: Positive Peptide Removal Breakdown ===\n")

# Build breakdown data.
pos_breakdown <- tibble(
  category = factor(
    c("Missing pLDDT only", "Missing RSA only", "Missing both", "Other removed positives"),
    levels = c("Missing pLDDT only", "Missing RSA only", "Missing both", "Other removed positives")
  ),
  count = c(n_pos_plddt_only, n_pos_rsa_only, n_pos_no_both, n_pos_has_structure),
  pct = count / n_pos_removed * 100
)

cat("  Positive peptides removed by category:\n")
for (i in seq_len(nrow(pos_breakdown))) {
  cat(sprintf("    %s: %d (%.1f%%)\n",
              as.character(pos_breakdown$category[i]),
              pos_breakdown$count[i],
              pos_breakdown$pct[i]))
}

# Stacked horizontal bar.
p_pos_breakdown <- ggplot(pos_breakdown, aes(x = count, y = "Removed\nPositives",
                                              fill = category)) +
  geom_col(width = 0.6, alpha = 0.85) +
  geom_text(aes(label = sprintf("%s\n(%.1f%%)", format(count, big.mark = ","), pct)),
            position = position_stack(vjust = 0.5),
            size = 3.5, fontface = "bold", color = "white") +
  scale_fill_manual(values = c(
    "Missing pLDDT only" = "#e74c3c",
    "Missing RSA only" = "#3498db",
    "Missing both" = "#8e44ad",
    "Other removed positives" = "#95a5a6"
  ), name = "Reason") +
  scale_x_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.1))) +
  labs(
    title = "Why Were Positive Peptides Removed?",
    subtitle = sprintf("Of %s positives removed, %s lost to structural filters (%.1f%%)",
                       format(n_pos_removed, big.mark = ","),
                       format(n_pos_no_plddt + n_pos_no_rsa - n_pos_no_both, big.mark = ","),
                       (n_pos_no_plddt + n_pos_no_rsa - n_pos_no_both) / n_pos_removed * 100),
    x = "Positive Peptides Removed",
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey40"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank()
  )

ggsave(file.path(OUT_DIR, "filtering_positives_breakdown.png"),
       p_pos_breakdown, width = 10, height = 4, dpi = 300)
cat("Saved: filtering_positives_breakdown.png\n")

# Summary.

cat("\n=== Done ===\n")
cat("All plots saved to:", OUT_DIR, "\n")
cat("  filtering_venn_protein.png          — Euler diagram of protein filter overlap\n")
cat("  filtering_stacked_bar.png           — Peptide removal by filter stage\n")
cat("  filtering_missing_comparison.png    — Missing data before vs after\n")
cat("  filtering_venn_peptide_positives.png — Positive peptide Venn (structural)\n")
cat("  filtering_positives_breakdown.png   — Positive removal by reason\n")
