# =============================================================================
# Categorical Residue / Cleavage-Context Analysis
# Person 2: Chi-square, Cramér's V, enrichment, plots
# =============================================================================

library(tidyverse)
library(broom)
library(ggplot2)
library(scales)
library(bio3d)
source("src/functions.R")
set_working_directory()

# -----------------------------------------------------------------------------
# 0. Setup output directories
# -----------------------------------------------------------------------------
dir.create("results",                         showWarnings = FALSE, recursive = TRUE)
dir.create("results/figures/categorical",     showWarnings = FALSE, recursive = TRUE)

# -----------------------------------------------------------------------------
# 1. Load and clean data
# -----------------------------------------------------------------------------
df_all <- read_csv("data/processed/df_all.csv", show_col_types = TRUE)

cat("Dataset dimensions:", nrow(df_all), "x", ncol(df_all), "\n")
cat("Label distribution:\n")
print(table(df_all$label))

# -----------------------------------------------------------------------------
# Define ALL cleavage-context features (P4–P4' for N- and C-terminal sites)
# -----------------------------------------------------------------------------
n_cleavage_features <- c(
  "n_cleavage_P4",
  "n_cleavage_P3",
  "n_cleavage_P2",
  "n_cleavage_P1",
  "n_cleavage_P1_prime",
  "n_cleavage_P2_prime",
  "n_cleavage_P3_prime",
  "n_cleavage_P4_prime"
)

c_cleavage_features <- c(
  "c_cleavage_P4",
  "c_cleavage_P3",
  "c_cleavage_P2",
  "c_cleavage_P1",
  "c_cleavage_P1_prime",
  "c_cleavage_P2_prime",
  "c_cleavage_P3_prime",
  "c_cleavage_P4_prime"
)

cleavage_features <- c(n_cleavage_features, c_cleavage_features)

# Human-readable labels used in plots (named vector for easy recode)
feature_labels <- c(
  "n_cleavage_P4"       = "N-term P4",
  "n_cleavage_P3"       = "N-term P3",
  "n_cleavage_P2"       = "N-term P2",
  "n_cleavage_P1"       = "N-term P1",
  "n_cleavage_P1_prime" = "N-term P1'",
  "n_cleavage_P2_prime" = "N-term P2'",
  "n_cleavage_P3_prime" = "N-term P3'",
  "n_cleavage_P4_prime" = "N-term P4'",
  "c_cleavage_P4"       = "C-term P4",
  "c_cleavage_P3"       = "C-term P3",
  "c_cleavage_P2"       = "C-term P2",
  "c_cleavage_P1"       = "C-term P1",
  "c_cleavage_P1_prime" = "C-term P1'",
  "c_cleavage_P2_prime" = "C-term P2'",
  "c_cleavage_P3_prime" = "C-term P3'",
  "c_cleavage_P4_prime" = "C-term P4'"
)

# -----------------------------------------------------------------------------
# Clean: keep only rows with non-NA label; convert features to character
# -----------------------------------------------------------------------------
df_clean <- df_all %>%
  filter(!is.na(label)) %>%
  mutate(label = factor(label, levels = c(0, 1),
                        labels = c("Not Presented", "Presented"))) %>%
  mutate(across(all_of(cleavage_features), as.character))

cat("\nRows after cleaning:", nrow(df_clean), "\n")
cat("Missing values per cleavage feature:\n")
df_clean %>%
  select(all_of(cleavage_features)) %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  print()

# =============================================================================
# 2. Helper functions
# =============================================================================

# -----------------------------------------------------------------------------
# 2a. Cramér's V from a contingency table
# -----------------------------------------------------------------------------
cramers_v <- function(ct) {
  chi2 <- suppressWarnings(chisq.test(ct)$statistic)
  n    <- sum(ct)
  k    <- min(nrow(ct), ncol(ct))
  v    <- sqrt(chi2 / (n * (k - 1)))
  as.numeric(v)
}

# -----------------------------------------------------------------------------
# 2b. Per-residue enrichment table
#   Returns: feature | residue | n_0 | n_1 | prop_0 | prop_1 |
#            log2_enrichment | odds_ratio
# -----------------------------------------------------------------------------
compute_enrichment <- function(df, feature) {
  
  ct <- table(df[[feature]], df$label)
  
  # Normalised frequencies (proportions within each label)
  prop_mat <- prop.table(ct, margin = 2)    # column proportions
  
  enrich <- as_tibble(prop_mat, .name_repair = "minimal") %>%
    rename(residue = 1, label = 2, proportion = n) %>%
    pivot_wider(names_from = label, values_from = proportion,
                names_prefix = "prop_") %>%
    rename(
      prop_0 = `prop_Not Presented`,
      prop_1 = `prop_Presented`
    ) %>%
    mutate(
      feature         = feature,
      log2_enrichment = log2((prop_1 + 1e-6) / (prop_0 + 1e-6))
    )
  
  # Odds ratios from raw counts (Haldane-Anscombe correction +0.5)
  counts <- as_tibble(ct, .name_repair = "minimal") %>%
    rename(residue = 1, label = 2, count = n) %>%
    pivot_wider(names_from = label, values_from = count,
                names_prefix = "n_") %>%
    rename(
      n_0 = `n_Not Presented`,
      n_1 = `n_Presented`
    ) %>%
    mutate(
      odds_ratio = ((n_1 + 0.5) / (sum(n_1) - n_1 + 0.5)) /
        ((n_0 + 0.5) / (sum(n_0) - n_0 + 0.5))
    )
  
  enrich <- left_join(enrich, counts, by = "residue") %>%
    relocate(feature, residue, n_0, n_1, prop_0, prop_1,
             log2_enrichment, odds_ratio)
  
  return(enrich)
}

# -----------------------------------------------------------------------------
# 2c. Chi-square + Cramér's V summary for one feature
# -----------------------------------------------------------------------------
test_feature <- function(df, feature) {
  
  sub <- df %>%
    filter(!is.na(.data[[feature]])) %>%
    mutate(residue = .data[[feature]])
  
  ct       <- table(sub$residue, sub$label)
  chi_test <- suppressWarnings(chisq.test(ct))
  v        <- cramers_v(ct)
  n_levels <- nrow(ct)
  
  tibble(
    feature    = feature,
    n_obs      = sum(ct),
    n_levels   = n_levels,
    chi2       = chi_test$statistic,
    df_chi2    = chi_test$parameter,
    p_value    = chi_test$p.value,
    cramers_v  = v
  )
}

# -----------------------------------------------------------------------------
# 2d. Grouped bar plot (within-label proportions) for ONE feature
# -----------------------------------------------------------------------------
plot_grouped_bar <- function(enrich_df, feature_name) {
  
  plot_data <- enrich_df %>%
    filter(feature == feature_name) %>%
    pivot_longer(cols = c(prop_0, prop_1),
                 names_to  = "label",
                 values_to = "proportion") %>%
    mutate(
      label   = recode(label,
                       "prop_0" = "Not Presented",
                       "prop_1" = "Presented"),
      residue = fct_reorder(residue, proportion, .fun = max, .desc = TRUE)
    )
  
  ggplot(plot_data, aes(x = residue, y = proportion, fill = label)) +
    geom_col(position = position_dodge(width = 0.7),
             width = 0.6, colour = "white", linewidth = 0.2) +
    scale_fill_manual(
      values = c("Not Presented" = "#E84646", "Presented" = "#2ecc71"),
      name   = "Label"
    ) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    labs(
      title    = paste0("Residue proportions by label — ",
                        feature_labels[feature_name]),
      subtitle = "Bars show within-label proportions",
      x        = "Residue (amino acid)",
      y        = "Proportion within label"
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title       = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      legend.position  = "top"
    )
}

# -----------------------------------------------------------------------------
# 2e. Stacked proportion plot  (fixed fct_reorder)
# -----------------------------------------------------------------------------
plot_stacked_proportion <- function(df, feature_name) {
  
  sub <- df %>%
    filter(!is.na(.data[[feature_name]])) %>%
    rename(residue = all_of(feature_name))
  
  counts <- sub %>%
    count(residue, label) %>%
    group_by(residue) %>%
    mutate(prop = n / sum(n)) %>%
    ungroup()
  
  # Compute the "Presented" proportion per residue for ordering.
  # Residues with no "Presented" rows get 0 (not NA).
  residue_order <- counts %>%
    filter(label == "Presented") %>%
    select(residue, prop_presented = prop) %>%
    # right-join so residues with zero presented rows are kept
    right_join(distinct(counts, residue), by = "residue") %>%
    mutate(prop_presented = replace_na(prop_presented, 0)) %>%
    arrange(desc(prop_presented)) %>%
    pull(residue)
  
  counts <- counts %>%
    mutate(residue = factor(residue, levels = residue_order))
  
  ggplot(counts, aes(x = residue, y = prop, fill = label)) +
    geom_col(width = 0.7, colour = "white", linewidth = 0.2) +
    scale_fill_manual(
      values = c("Not Presented" = "#4878CF", "Presented" = "#E84646"),
      name   = "Label"
    ) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    geom_hline(yintercept = mean(df$label == "Presented"),
               linetype = "dashed", colour = "grey40") +
    labs(
      title    = paste0("Stacked label proportions — ",
                        feature_labels[feature_name]),
      subtitle = "Dashed line = overall presentation rate",
      x        = "Residue (amino acid)",
      y        = "Proportion"
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title       = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      legend.position  = "top"
    )
}

# =============================================================================
# 3. Run chi-square + Cramér's V for all 16 features
# =============================================================================
cat("\n--- Running chi-square tests ---\n")

test_results <- map_dfr(cleavage_features, ~test_feature(df_clean, .x))

# FDR correction across all tests
test_results <- test_results %>%
  mutate(
    p_adj_BH        = p.adjust(p_value, method = "BH"),
    feature_label   = feature_labels[feature],
    v_interpretation = case_when(
      cramers_v >= 0.25 ~ "strong",
      cramers_v >= 0.15 ~ "moderate",
      cramers_v >= 0.05 ~ "weak",
      TRUE              ~ "negligible"
    ),
    sig = case_when(
      p_adj_BH < 0.001 ~ "***",
      p_adj_BH < 0.01  ~ "**",
      p_adj_BH < 0.05  ~ "*",
      TRUE             ~ "ns"
    )
  ) %>%
  arrange(p_adj_BH)

cat("\nStatistical test summary:\n")
test_results %>%
  mutate(chi2 = round(chi2, 1), cramers_v = round(cramers_v, 4)) %>%
  select(feature_label, n_obs, n_levels, chi2, df_chi2,
         cramers_v, v_interpretation, p_adj_BH, sig) %>%
  print(n = 20)

# Quick inspection of unique values per feature
cat("\nUnique value counts per feature:\n")
for (feat in cleavage_features) {
  cat(sprintf("  %-30s : %d unique values\n", feat,
              n_distinct(df_clean[[feat]], na.rm = TRUE)))
}

# write_csv(test_results, "results/categorical_feature_tests.csv")
# cat("Saved: results/categorical_feature_tests.csv\n")

# =============================================================================
# 4. Compute per-residue enrichment for all 16 features
# =============================================================================
cat("\n--- Computing per-residue enrichment ---\n")

enrichment_all <- map_dfr(cleavage_features, ~compute_enrichment(df_clean, .x))

cat("\nEnrichment table (first 20 rows):\n")
print(head(enrichment_all, 20))

# write_csv(enrichment_all, "results/residue_enrichment_tables.csv")
# cat("Saved: results/residue_enrichment_tables.csv\n")

# =============================================================================
# 5. Plots
# =============================================================================
cat("\n--- Generating plots ---\n")

# 5a. Grouped bar plots (one PNG per feature)
for (feat in cleavage_features) {
  p     <- plot_grouped_bar(enrichment_all, feat)
  fname <- paste0("results/figures/categorical/grouped_bar_", feat, ".png")
  ggsave(fname, plot = p, width = 10, height = 5, dpi = 150)
  cat("Saved:", fname, "\n")
}

# 5b. Stacked proportion plots (one PNG per feature)
for (feat in cleavage_features) {
  p     <- plot_stacked_proportion(df_clean, feat)
  fname <- paste0("results/figures/categorical/stacked_prop_", feat, ".png")
  ggsave(fname, plot = p, width = 10, height = 5, dpi = 150)
  cat("Saved:", fname, "\n")
}

# -----------------------------------------------------------------------------
# 5c. Log2-enrichment heatmap — split into N-terminal and C-terminal panels
#     so the figure stays readable with 8 positions each
# -----------------------------------------------------------------------------

make_heatmap <- function(enrich_df, features, title_suffix) {
  
  hdata <- enrich_df %>%
    filter(feature %in% features) %>%
    mutate(
      log2_enrich_capped = pmax(pmin(log2_enrichment, 3), -3),
      feature_label      = factor(feature_labels[feature],
                                  levels = feature_labels[features])
    )
  
  # Order residues by mean enrichment across the selected features
  residue_order <- hdata %>%
    group_by(residue) %>%
    summarise(mean_enrich = mean(log2_enrichment, na.rm = TRUE),
              .groups = "drop") %>%
    arrange(mean_enrich) %>%
    pull(residue)
  
  hdata <- hdata %>%
    mutate(residue = factor(residue, levels = residue_order))
  
  ggplot(hdata, aes(x = feature_label, y = residue,
                    fill = log2_enrich_capped)) +
    geom_tile(colour = "white", linewidth = 0.4) +
    geom_text(aes(label = round(log2_enrichment, 2)),
              size = 2.6, colour = "grey10") +
    scale_fill_gradient2(
      low      = "#3A5FCD",
      mid      = "white",
      high     = "#CC2929",
      midpoint = 0,
      limits   = c(-3, 3),
      oob      = squish,
      name     = "log2\nenrichment\n(capped ±3)"
    ) +
    labs(
      title    = paste0("Log2 enrichment — ", title_suffix),
      subtitle = "Presented (label=1) vs Not Presented (label=0)",
      x        = "Cleavage position",
      y        = "Amino acid"
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title  = element_text(face = "bold"),
      axis.text.x = element_text(angle = 35, hjust = 1),
      panel.grid  = element_blank()
    )
}

p_heat_n <- make_heatmap(enrichment_all, n_cleavage_features,
                         "N-terminal cleavage site (P4–P4')")
p_heat_c <- make_heatmap(enrichment_all, c_cleavage_features,
                         "C-terminal cleavage site (P4–P4')")

ggsave("results/figures/categorical/log2_enrichment_heatmap_Nterm.png",
       plot = p_heat_n, width = 10, height = 10, dpi = 150)
cat("Saved: results/figures/categorical/log2_enrichment_heatmap_Nterm.png\n")

ggsave("results/figures/categorical/log2_enrichment_heatmap_Cterm.png",
       plot = p_heat_c, width = 10, height = 10, dpi = 150)
cat("Saved: results/figures/categorical/log2_enrichment_heatmap_Cterm.png\n")

# -----------------------------------------------------------------------------
# 5d. Odds-ratio dot plot — faceted by position, split N / C terminal
# -----------------------------------------------------------------------------

make_or_plot <- function(enrich_df, features, title_suffix) {
  
  plot_data <- enrich_df %>%
    filter(feature %in% features) %>%
    mutate(
      feature_label = factor(feature_labels[feature],
                             levels = feature_labels[features]),
      enriched      = log2_enrichment > 0
    )
  
  ggplot(plot_data,
         aes(x = log2(odds_ratio), y = residue, colour = enriched)) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
    geom_point(size = 2.5, alpha = 0.85) +
    scale_colour_manual(
      values = c("TRUE" = "#CC2929", "FALSE" = "#3A5FCD"),
      labels = c("TRUE" = "Enriched", "FALSE" = "Depleted"),
      name   = NULL
    ) +
    facet_wrap(~feature_label, ncol = 4) +
    labs(
      title    = paste0("Per-residue odds ratios — ", title_suffix),
      subtitle = "log2(OR) > 0 → enriched in presented peptides",
      x        = "log2(Odds Ratio)",
      y        = "Amino acid"
    ) +
    theme_bw(base_size = 11) +
    theme(
      plot.title       = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      legend.position  = "bottom"
    )
}

p_or_n <- make_or_plot(enrichment_all, n_cleavage_features,
                       "N-terminal cleavage site")
p_or_c <- make_or_plot(enrichment_all, c_cleavage_features,
                       "C-terminal cleavage site")

ggsave("results/figures/categorical/odds_ratio_dotplot_Nterm.png",
       plot = p_or_n, width = 16, height = 7, dpi = 150)
cat("Saved: results/figures/categorical/odds_ratio_dotplot_Nterm.png\n")

ggsave("results/figures/categorical/odds_ratio_dotplot_Cterm.png",
       plot = p_or_c, width = 16, height = 7, dpi = 150)
cat("Saved: results/figures/categorical/odds_ratio_dotplot_Cterm.png\n")

# =============================================================================
# 6. Console summary: top enriched / depleted residues per feature
# =============================================================================
cat("\n=== TOP 5 ENRICHED RESIDUES PER FEATURE (label=1) ===\n")
enrichment_all %>%
  group_by(feature) %>%
  slice_max(log2_enrichment, n = 5) %>%
  mutate(feature_label = feature_labels[feature]) %>%
  select(feature_label, residue, n_0, n_1,
         prop_0, prop_1, log2_enrichment, odds_ratio) %>%
  print(n = 80)

cat("\n=== TOP 5 DEPLETED RESIDUES PER FEATURE (label=1) ===\n")
enrichment_all %>%
  group_by(feature) %>%
  slice_min(log2_enrichment, n = 5) %>%
  mutate(feature_label = feature_labels[feature]) %>%
  select(feature_label, residue, n_0, n_1,
         prop_0, prop_1, log2_enrichment, odds_ratio) %>%
  print(n = 80)

cat("\n--- Analysis complete ---\n")