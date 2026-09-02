#!/usr/bin/env Rscript

# =============================================================================
# Categorical Residue / Cleavage-Context Analysis
# Chi-square, Cramér's V, enrichment, and Differential Logo Plots
# =============================================================================

library(tidyverse)
library(broom)
library(ggplot2)
library(scales)
library(ggseqlogo)
source("src/pipeline/r/functions.R")
set_working_directory()

# -----------------------------------------------------------------------------
# 0. Setup output directories
# -----------------------------------------------------------------------------
dir.create("results",                         showWarnings = FALSE, recursive = TRUE)
dir.create("results/figures/categorical",     showWarnings = FALSE, recursive = TRUE)

# -----------------------------------------------------------------------------
# 1. Load and clean data
# -----------------------------------------------------------------------------
df_all <- read_csv("data/processed/df_all.csv", show_col_types = FALSE)

cat("Dataset dimensions:", nrow(df_all), "x", ncol(df_all), "\n")
cat("Label distribution:\n")
print(table(df_all$label))

# -----------------------------------------------------------------------------
# Define ALL 17 positional features (N-flank, Peptide, C-flank)
# -----------------------------------------------------------------------------
n_flank_features <- paste0("N", 10:1)               # was N4:N1
peptide_features <- paste0("P", 1:9)
c_flank_features <- paste0("C", 1:10)               # was C1:C4

all_pos_features <- c(n_flank_features, peptide_features, c_flank_features)
# Total: 29 positions

# Human-readable labels used in plots
feature_labels <- setNames(
  c(
    paste0("N-term N", 10:1),     # "N-term N10" ... "N-term N1"
    paste0("Peptide P", 1:9),     # "Peptide P1" ... "Peptide P9"
    paste0("C-term C", 1:10)      # "C-term C1" ... "C-term C10"
  ),
  all_pos_features
)

aa_alphabet <- c("A","C","D","E","F","G","H","I","K","L",
                 "M","N","P","Q","R","S","T","V","W","Y")

# -----------------------------------------------------------------------------
# Clean: keep only rows with non-NA label; convert features to character
# -----------------------------------------------------------------------------
df_clean <- df_all %>%
  filter(!is.na(label)) %>%
  mutate(label = factor(label, levels = c(0, 1),
                        labels = c("Not Presented", "Presented"))) %>%
  mutate(across(all_of(all_pos_features), as.character))

cat("\nRows after cleaning:", nrow(df_clean), "\n")
cat("Missing values per position (NA implies sequence hit a protein terminus):\n")
df_clean %>%
  select(all_of(all_pos_features)) %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  print()

# =============================================================================
# 2. Helper functions
# =============================================================================

cramers_v <- function(ct) {
  chi2 <- suppressWarnings(chisq.test(ct)$statistic)
  n    <- sum(ct)
  k    <- min(nrow(ct), ncol(ct))
  v    <- sqrt(chi2 / (n * (k - 1)))
  as.numeric(v)
}

compute_enrichment <- function(df, feature) {
  ct <- table(df[[feature]], df$label)
  
  # Normalised frequencies
  prop_mat <- prop.table(ct, margin = 2)
  
  enrich <- as_tibble(prop_mat, .name_repair = "minimal") %>%
    rename(residue = 1, label = 2, proportion = n) %>%
    pivot_wider(names_from = label, values_from = proportion,
                names_prefix = "prop_") %>%
    rename(prop_0 = `prop_Not Presented`, prop_1 = `prop_Presented`) %>%
    mutate(
      feature         = feature,
      log2_enrichment = log2((prop_1 + 1e-6) / (prop_0 + 1e-6))
    )
  
  # Odds ratios (Haldane-Anscombe correction)
  counts <- as_tibble(ct, .name_repair = "minimal") %>%
    rename(residue = 1, label = 2, count = n) %>%
    pivot_wider(names_from = label, values_from = count, names_prefix = "n_") %>%
    rename(n_0 = `n_Not Presented`, n_1 = `n_Presented`) %>%
    mutate(
      odds_ratio = ((n_1 + 0.5) / (sum(n_1) - n_1 + 0.5)) /
        ((n_0 + 0.5) / (sum(n_0) - n_0 + 0.5))
    )
  
  enrich <- left_join(enrich, counts, by = "residue") %>%
    relocate(feature, residue, n_0, n_1, prop_0, prop_1, log2_enrichment, odds_ratio)
  
  return(enrich)
}

test_feature <- function(df, feature) {
  sub <- df %>% filter(!is.na(.data[[feature]])) %>% mutate(residue = .data[[feature]])
  ct       <- table(sub$residue, sub$label)
  chi_test <- suppressWarnings(chisq.test(ct))
  v        <- cramers_v(ct)
  
  tibble(
    feature    = feature,
    n_obs      = sum(ct),
    n_levels   = nrow(ct),
    chi2       = chi_test$statistic,
    df_chi2    = chi_test$parameter,
    p_value    = chi_test$p.value,
    cramers_v  = v
  )
}

plot_grouped_bar <- function(df, feature_name) {
  sub <- df %>% filter(!is.na(.data[[feature_name]])) %>% mutate(residue = .data[[feature_name]])
  total_n <- nrow(sub)
  
  counts <- sub %>%
    count(residue, label, name = "n") %>%
    group_by(label) %>%
    mutate(total = sum(n), proportion = n / total) %>%
    ungroup() %>%
    mutate(residue = fct_reorder(residue, proportion, .fun = max, .desc = TRUE))
  
  ggplot(counts, aes(x = residue, y = proportion, fill = label)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6, colour = "white", linewidth = 0.2) +
    scale_fill_manual(values = c("Not Presented" = "#E84646", "Presented" = "#2ecc71"), name = "Label") +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    labs(
      title    = paste0("Residue proportions by label — ", feature_labels[feature_name]),
      subtitle = paste0("n=", format(total_n, big.mark = ","), "; bars show within-label proportions"),
      x = "Amino acid", y = "Proportion within label"
    ) +
    theme_bw(base_size = 12) + theme(plot.title = element_text(face = "bold"), legend.position = "top")
}

plot_stacked_proportion <- function(df, feature_name) {
  sub <- df %>% filter(!is.na(.data[[feature_name]])) %>% rename(residue = all_of(feature_name))
  total_n <- nrow(sub)
  
  counts <- sub %>%
    count(residue, label) %>%
    group_by(residue) %>%
    mutate(prop = n / sum(n), total = sum(n)) %>%
    ungroup()
  
  residue_order <- counts %>% filter(label == "Presented") %>% select(residue, prop_presented = prop) %>%
    right_join(distinct(counts, residue), by = "residue") %>%
    mutate(prop_presented = replace_na(prop_presented, 0)) %>%
    arrange(desc(prop_presented)) %>% pull(residue)
  
  counts <- counts %>% mutate(residue = factor(residue, levels = residue_order))
  annotations <- counts %>% select(residue, total) %>% distinct() %>% mutate(y_pos = 1.03)
  
  ggplot(counts, aes(x = residue, y = prop, fill = label)) +
    geom_col(width = 0.7, colour = "white", linewidth = 0.2) +
    scale_fill_manual(values = c("Not Presented" = "#E84646", "Presented" = "#2ecc71"), name = "Label") +
    scale_y_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0, 0.08))) +
    geom_hline(yintercept = mean(df$label == "Presented"), linetype = "dashed", colour = "grey40") +
    geom_text(data = annotations, aes(x = residue, y = y_pos, label = total), inherit.aes = FALSE, size = 3, fontface = "bold", colour = "grey20", vjust = 0) +
    labs(
      title    = paste0("Stacked label proportions — ", feature_labels[feature_name]),
      subtitle = paste0("Dashed line = overall presentation rate; n=", format(total_n, big.mark = ",")),
      x = "Amino acid", y = "Proportion"
    ) +
    theme_bw(base_size = 12) + theme(plot.title = element_text(face = "bold"), legend.position = "top")
}

# =============================================================================
# 3. Run chi-square + Cramér's V for all 17 features
# =============================================================================
cat("\n--- Running chi-square tests ---\n")

test_results <- map_dfr(all_pos_features, ~test_feature(df_clean, .x)) %>%
  mutate(
    p_adj_BH = p.adjust(p_value, method = "BH"),
    feature_label = feature_labels[feature],
    v_interpretation = case_when(cramers_v >= 0.25 ~ "strong", cramers_v >= 0.15 ~ "moderate", cramers_v >= 0.05 ~ "weak", TRUE ~ "negligible"),
    sig = case_when(p_adj_BH < 0.001 ~ "***", p_adj_BH < 0.01 ~ "**", p_adj_BH < 0.05 ~ "*", TRUE ~ "ns")
  ) %>% arrange(desc(cramers_v))

cat("\nStatistical test summary (sorted by Effect Size):\n")
test_results %>%
  mutate(chi2 = round(chi2, 1), cramers_v = round(cramers_v, 4)) %>%
  select(feature_label, n_obs, chi2, cramers_v, v_interpretation, sig) %>%
  print(n = 20)

write_csv(test_results, "results/categorical_feature_tests.csv")

# =============================================================================
# 4. Compute per-residue enrichment
# =============================================================================
cat("\n--- Computing per-residue enrichment ---\n")
enrichment_all <- map_dfr(all_pos_features, ~compute_enrichment(df_clean, .x))
write_csv(enrichment_all, "results/residue_enrichment_tables.csv")

# =============================================================================
# 5. Plots (Heatmaps and Bar Charts)
# =============================================================================
cat("\n--- Generating plots ---\n")

for (feat in all_pos_features) {
  ggsave(paste0("results/figures/categorical/residue_bars/grouped_bar_", feat, ".png"), plot = plot_grouped_bar(df_clean, feat), width = 10, height = 5, dpi = 150)
  ggsave(paste0("results/figures/categorical/residue_stacked/stacked_prop_", feat, ".png"), plot = plot_stacked_proportion(df_clean, feat), width = 10, height = 5, dpi = 150)
}

# -----------------------------------------------------------------------------
# 5a. Unified Log2-enrichment heatmap (All 17 positions)
# -----------------------------------------------------------------------------
hdata <- enrichment_all %>%
  mutate(
    log2_enrich_capped = pmax(pmin(log2_enrichment, 3), -3),
    feature_label = factor(feature_labels[feature], levels = feature_labels[all_pos_features])
  ) %>%
  mutate(residue = factor(residue, levels = rev(aa_alphabet)))

p_heat_all <- ggplot(hdata, aes(x = feature_label, y = residue, fill = log2_enrich_capped)) +
  geom_tile(colour = "white", linewidth = 0.4) +
  geom_text(aes(label = round(log2_enrichment, 2)), size = 2.6, colour = "grey10") +
  geom_vline(xintercept = 10.5, linetype = "dashed", color = "black") +
  geom_vline(xintercept = 19.5, linetype = "dashed", color = "black") +
  scale_fill_gradient2(low = "#2ecc71", mid = "white", high = "#CC2929", midpoint = 0, limits = c(-3, 3), oob = squish, name = "log2\nenrichment\n(capped ±3)") +
  labs(title = "Log2 enrichment — All 29 Positions", subtitle = "Presented (label=1) vs Not Presented (label=0)", x = "Sequence Position", y = "Amino acid") +
  theme_bw(base_size = 12) + theme(plot.title = element_text(face = "bold"), axis.text.x = element_text(angle = 45, hjust = 1), panel.grid = element_blank())

ggsave("results/figures/categorical/log2_enrichment_heatmap_all.png", plot = p_heat_all, width = 20, height = 8, dpi = 300)

# -----------------------------------------------------------------------------
# 5b. Unified Odds-ratio dot plot (Faceted by all 17 positions)
# -----------------------------------------------------------------------------
plot_data <- enrichment_all %>%
  mutate(
    feature_label = factor(feature_labels[feature], levels = feature_labels[all_pos_features]),
    enriched = log2_enrichment > 0
  )

p_or_all <- ggplot(plot_data, aes(x = log2(odds_ratio), y = residue, colour = enriched)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
  geom_point(size = 2.5, alpha = 0.85) +
  scale_colour_manual(values = c("TRUE" = "#2ecc71", "FALSE" = "#CC2929"), labels = c("TRUE" = "Enriched", "FALSE" = "Depleted"), name = NULL) +
  facet_wrap(~feature_label, ncol = 6) +
  labs(title = "Per-residue odds ratios — All 29 Positions", subtitle = "log2(OR) > 0 → enriched in presented peptides", x = "log2(Odds Ratio)", y = "Amino acid") +
  theme_bw(base_size = 11) + theme(plot.title = element_text(face = "bold"), panel.grid.minor = element_blank(), legend.position = "bottom")

ggsave("results/figures/categorical/odds_ratio_dotplot_all.png", plot = p_or_all, width = 16, height = 10, dpi = 300)

# =============================================================================
# 6. Differential Sequence Logo Plot (Using ggseqlogo)
# =============================================================================
cat("\n--- Generating Differential Logo Plot ---\n")

get_ppm_for_label <- function(df, label_val, cols, alphabet) {
  sub_df <- df |> filter(label == label_val)
  ppm <- sapply(cols, function(col) {
    # Drop NAs natively per column so terminus gaps don't skew probabilities
    vals <- na.omit(sub_df[[col]])
    counts <- table(factor(vals, levels = alphabet))
    counts / sum(counts)
  })
  rownames(ppm) <- alphabet
  return(ppm)
}

ppm_1 <- get_ppm_for_label(df_clean, "Presented", all_pos_features, aa_alphabet)
ppm_0 <- get_ppm_for_label(df_clean, "Not Presented", all_pos_features, aa_alphabet)

# Difference Matrix: >0 = enriched in Presented, <0 = depleted in Presented
diff_ppm <- ppm_1 - ppm_0

p_diff_logo <- ggseqlogo(diff_ppm, method = "custom") +
  labs(
    title = "Differential Sequence Logo: Presented vs. Not Presented",
    subtitle = "Top (Positive) = Enriched in Presented | Bottom (Negative) = Depleted in Presented",
    x = "Sequence Position (N-Flank → Peptide → C-Flank)",
    y = "Difference in Probability"
  ) +
  scale_x_continuous(breaks = 1:29, labels = all_pos_features) +
  geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 0.5) +
  geom_vline(xintercept = 10.5, linetype = "dashed", color = "grey50") +  
  geom_vline(xintercept = 19.5, linetype = "dashed", color = "grey50") + 
  theme_logo() +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "grey30"),
    axis.text.x = element_text(size = 11, face = "bold", angle = 0),
    panel.grid.major.y = element_line(color = "grey90")
  )

ggsave("results/figures/categorical/differential_logo_29mer.png", plot = p_diff_logo, width = 20, height = 5, dpi = 300)
cat("Saved: results/figures/categorical/differential_logo_29mer.png\n")

cat("\n--- Analysis complete ---\n")