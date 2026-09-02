#!/usr/bin/env Rscript
library(tidyverse)
library(broom)
library(ggplot2)
library(ggrepel)
source("src/pipeline/r/functions.R")
set_working_directory()

# =============================================================================
# CONFIGURATION
# =============================================================================
# Change this to "base", "sparse", or "blosum50"
dataset_type <- "sparse" 

if (dataset_type == "base") {
  input_file <- "data/processed/df_all.csv"
  run_suffix <- ""
} else {
  input_file <- paste0("data/processed/df_all_", dataset_type, ".csv")
  run_suffix <- paste0("_", dataset_type)
}

results_dir <- "results"
figures_dir <- file.path("results", "figures", paste0("numeric_9mer", run_suffix))
per_feature_dir <- file.path(figures_dir, "per_feature")
heatmap_dir <- file.path(figures_dir, "heatmap")

dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(per_feature_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(heatmap_dir, showWarnings = FALSE, recursive = TRUE)

# Load data
message("Loading dataset: ", input_file)
df_raw <- read_csv(input_file, show_col_types = FALSE)

df_raw %>%
  mutate(length = nchar(peptide)) %>%
  count(length)

df_raw <- df_raw |>
  filter(!is.na(label)) |>
  mutate(label = as.integer(label)) |>
  filter(label %in% c(0, 1))

message("Label counts:")
print(table(df_raw$label))

# =============================================================================
# FEATURE CLASSIFICATION
# =============================================================================

continuous_features <- list(
  peptide = intersect(
    c("mean_rsa_peptide", "mean_disorder_peptide",
      "mean_plddt_peptide", "sd_plddt_peptide", "min_plddt_peptide",
      "frac_disordered_peptide",
      "rel_distance_from_n_terminus", "rel_distance_from_c_terminus",
      "protein_length"),
    names(df_raw)
  ),
  n_flank = intersect(
    c("mean_rsa_nflank", "mean_disorder_nflank",
      "mean_plddt_nflank", "sd_plddt_nflank", "frac_disordered_nflank"),
    names(df_raw)
  ),
  c_flank = intersect(
    c("mean_rsa_cflank", "mean_disorder_cflank",
      "mean_plddt_cflank", "sd_plddt_cflank", "frac_disordered_cflank"),
    names(df_raw)
  )
)

binary_features <- list(
  peptide     = names(df_raw)[str_detect(names(df_raw), "^q8point_.*_peptide$")],
  n_flank     = names(df_raw)[str_detect(names(df_raw), "^q8point_.*_nflank$")],
  c_flank     = names(df_raw)[str_detect(names(df_raw), "^q8point_.*_cflank$")],
  trans_n_pep = names(df_raw)[str_detect(names(df_raw), "^q8trans_.*_n_pep$")],
  trans_pep_c = names(df_raw)[str_detect(names(df_raw), "^q8trans_.*_pep_c$")]
)

# Identify the encoded sequence columns dynamically (N1-10, P1-9, C1-10 followed by _AminoAcid)
sequence_features <- names(df_raw)[
  str_detect(names(df_raw), "^(N([1-9]|10)|P[1-9]|C([1-9]|10))_[A-Z]$")
]

cat("\n=== Feature Inventory ===\n")
cat("Continuous features:\n")
iwalk(continuous_features, ~ cat("  ", .y, ":", length(.x), "\n"))
cat("Binary features:\n")
iwalk(binary_features, ~ cat("  ", .y, ":", length(.x), "\n"))
cat(sprintf("Sequence encoded features (%s): %d\n", dataset_type, length(sequence_features)))

# Pretty labels
pretty_feature_label <- function(x) {
  x |>
    str_replace("_(peptide|nflank|cflank)$", "") |>
    str_replace("_(n_pep|pep_c)$", "") |>
    str_replace("^mean_rsa$", "Mean RSA") |>
    str_replace("^mean_disorder$", "Mean disorder") |>
    str_replace("^mean_plddt$", "Mean pLDDT") |>
    str_replace("^sd_plddt$", "SD pLDDT") |>
    str_replace("^min_plddt$", "Min pLDDT") |>
    str_replace("^frac_disordered$", "Frac. disordered") |>
    str_replace("^rel_distance_from_n_terminus$", "Rel. dist. N-terminus") |>
    str_replace("^rel_distance_from_c_terminus$", "Rel. dist. C-terminus") |>
    str_replace("^protein_length$", "Protein length") |>
    str_replace("^q8point_H$", "Q8 α-helix present") |>
    str_replace("^q8point_G$", "Q8 3₁₀-helix present") |>
    str_replace("^q8point_I$", "Q8 π-helix present") |>
    str_replace("^q8point_E$", "Q8 β-strand present") |>
    str_replace("^q8point_B$", "Q8 β-bridge present") |>
    str_replace("^q8point_T$", "Q8 turn present") |>
    str_replace("^q8point_S$", "Q8 bend present") |>
    str_replace("^q8point_C$", "Q8 coil present") |>
    str_replace("^q8trans_H$", "Q8 α-helix transition") |>
    str_replace("^q8trans_G$", "Q8 3₁₀-helix transition") |>
    str_replace("^q8trans_I$", "Q8 π-helix transition") |>
    str_replace("^q8trans_E$", "Q8 β-strand transition") |>
    str_replace("^q8trans_B$", "Q8 β-bridge transition") |>
    str_replace("^q8trans_T$", "Q8 turn transition") |>
    str_replace("^q8trans_S$", "Q8 bend transition") |>
    str_replace("^q8trans_C$", "Q8 coil transition")
}

# =============================================================================
# HELPERS
# =============================================================================

sanitize_filename <- function(x) {
  x |> str_replace_all("[^A-Za-z0-9_\\-]", "_")
}

safe_wilcox <- function(data, feature) {
  sub <- data |>
    select(all_of(feature), label) |>
    filter(!is.na(.data[[feature]]), !is.na(label))
  
  x0 <- sub |> filter(label == 0) |> pull(all_of(feature))
  x1 <- sub |> filter(label == 1) |> pull(all_of(feature))
  
  if (length(x0) < 2 || length(x1) < 2) {
    return(tibble(feature = feature, n_0 = length(x0), n_1 = length(x1),
                  mean_0 = NA_real_, mean_1 = NA_real_, median_0 = NA_real_,
                  median_1 = NA_real_, sd_0 = NA_real_, sd_1 = NA_real_,
                  pooled_sd = NA_real_, smd = NA_real_, smd_se = NA_real_,
                  smd_ci_low = NA_real_, smd_ci_high = NA_real_,
                  wilcox_statistic = NA_real_, p_value = NA_real_,
                  median_diff = NA_real_, mean_diff = NA_real_))
  }
  
  wt <- tryCatch(
    suppressWarnings(wilcox.test(x1, x0, alternative = "two.sided", exact = FALSE)),
    error = function(e) NULL
  )
  
  mean_0 <- mean(x0); mean_1 <- mean(x1)
  sd_0 <- sd(x0); sd_1 <- sd(x1)
  n0 <- length(x0); n1 <- length(x1)
  pooled_sd <- sqrt(((n0 - 1) * sd_0^2 + (n1 - 1) * sd_1^2) / (n0 + n1 - 2))
  smd <- ifelse(is.finite(pooled_sd) && pooled_sd > 0, (mean_1 - mean_0) / pooled_sd, NA_real_)
  smd_se <- ifelse(is.finite(smd),
                   sqrt((n0 + n1) / (n0 * n1) + (smd^2) / (2 * (n0 + n1 - 2))),
                   NA_real_)
  
  tibble(
    feature = feature, n_0 = n0, n_1 = n1,
    mean_0 = mean_0, mean_1 = mean_1, median_0 = median(x0), median_1 = median(x1),
    sd_0 = sd_0, sd_1 = sd_1, pooled_sd = pooled_sd, smd = smd,
    smd_se = smd_se, smd_ci_low = smd - 1.96 * smd_se, smd_ci_high = smd + 1.96 * smd_se,
    wilcox_statistic = ifelse(is.null(wt), NA_real_, as.numeric(wt$statistic)),
    p_value = ifelse(is.null(wt), NA_real_, wt$p.value),
    median_diff = median(x1) - median(x0), mean_diff = mean_1 - mean_0
  )
}

safe_logit <- function(data, feature, scale_feature = FALSE) {
  sub <- data |>
    select(all_of(feature), label) |>
    filter(!is.na(.data[[feature]]), !is.na(label))
  
  if (nrow(sub) < 10 || n_distinct(sub[[feature]]) < 2) {
    return(tibble(feature = feature, n = nrow(sub), scaled = scale_feature,
                  estimate = NA_real_, std_error = NA_real_, statistic = NA_real_,
                  p_value = NA_real_, odds_ratio = NA_real_,
                  conf_low = NA_real_, conf_high = NA_real_))
  }
  
  if (scale_feature) {
    sub <- sub |> mutate(x = as.numeric(scale(.data[[feature]])))
    form <- label ~ x; term_name <- "x"
  } else {
    form <- as.formula(paste("label ~", feature)); term_name <- feature
  }
  
  fit <- tryCatch(glm(form, data = sub, family = binomial()), error = function(e) NULL)
  if (is.null(fit)) return(tibble(feature = feature, n = nrow(sub), scaled = scale_feature,
                                  estimate = NA_real_, std_error = NA_real_, statistic = NA_real_,
                                  p_value = NA_real_, odds_ratio = NA_real_,
                                  conf_low = NA_real_, conf_high = NA_real_))
  
  tid <- tidy(fit) |> filter(term == term_name)
  if (nrow(tid) == 0) return(tibble(feature = feature, n = nrow(sub), scaled = scale_feature,
                                    estimate = NA_real_, std_error = NA_real_, statistic = NA_real_,
                                    p_value = NA_real_, odds_ratio = NA_real_,
                                    conf_low = NA_real_, conf_high = NA_real_))
  
  est <- tid$estimate[1]; se <- tid$std.error[1]
  tibble(feature = feature, n = nrow(sub), scaled = scale_feature,
         estimate = est, std_error = se, statistic = tid$statistic[1],
         p_value = tid$p.value[1], odds_ratio = exp(est),
         conf_low = exp(est - 1.96 * se), conf_high = exp(est + 1.96 * se))
}

plot_numeric_feature <- function(data, feature, outdir) {
  sub <- data |>
    select(all_of(feature), label) |>
    filter(!is.na(.data[[feature]]), !is.na(label)) |>
    mutate(label = factor(label, levels = c(0, 1),
                          labels = c("Not presented", "Presented")))
  if (nrow(sub) == 0) return(NULL)
  fname <- sanitize_filename(feature)
  
  p_box <- ggplot(sub, aes(x = label, y = .data[[feature]], fill = label)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.85, width = 0.5) +
    scale_fill_manual(values = c("#e74c3c", "#2ecc71")) +
    labs(title = paste("Distribution of", feature), x = "Label", y = feature) +
    theme_bw() + theme(legend.position = "none")
  ggsave(file.path(outdir, paste0(fname, "_boxplot.png")),
         plot = p_box, width = 6, height = 4, dpi = 300)
  
  p_ecdf <- ggplot(sub, aes(x = .data[[feature]], color = label)) +
    stat_ecdf(linewidth = 1) +
    scale_color_manual(values = c("#e74c3c", "#2ecc71")) +
    labs(title = paste("ECDF of", feature), x = feature, y = "ECDF", color = "Label") +
    theme_bw()
  ggsave(file.path(outdir, paste0(fname, "_ecdf.png")),
         plot = p_ecdf, width = 6, height = 4, dpi = 300)
}

# =============================================================================
# 1. CORRELATION HEATMAPS (structural features only, safely excludes sequences)
# =============================================================================
message("Generating correlation heatmaps...")

heatmap_feature_groups <- list(
  peptide = c(continuous_features$peptide, binary_features$peptide),
  n_flank = c(continuous_features$n_flank, binary_features$n_flank),
  c_flank = c(continuous_features$c_flank, binary_features$c_flank),
  transitions = c(binary_features$trans_n_pep, binary_features$trans_pep_c)
)

heatmap_titles <- c(
  peptide     = "Correlation Heatmap — Peptide",
  n_flank     = "Correlation Heatmap — N-flank",
  c_flank     = "Correlation Heatmap — C-flank",
  transitions = "Correlation Heatmap — Q8 Transitions"
)

for (group_name in names(heatmap_feature_groups)) {
  cols <- heatmap_feature_groups[[group_name]]
  cols <- intersect(cols, names(df_raw))
  if (length(cols) < 2) next
  
  col_vars <- apply(df_raw[cols], 2, var, na.rm = TRUE)
  cols <- cols[col_vars > 0]
  if (length(cols) < 2) next
  
  cor_mat <- cor(df_raw[cols], use = "pairwise.complete.obs")
  pretty_labels <- pretty_feature_label(cols)
  rownames(cor_mat) <- pretty_labels
  colnames(cor_mat) <- pretty_labels
  
  cor_long <- as.data.frame(as.table(cor_mat)) |>
    rename(Feature_1 = Var1, Feature_2 = Var2, Correlation = Freq) |>
    filter(!is.na(Correlation))
  
  n_feats <- length(cols)
  plot_size <- max(7, n_feats * 0.6 + 2)
  
  p_heat <- ggplot(cor_long, aes(x = Feature_1, y = Feature_2, fill = Correlation)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.2f", Correlation)),
              size = max(2, 4 - n_feats * 0.1), color = "black", alpha = 0.7) +
    scale_fill_gradient2(low = "#3b82f6", mid = "white", high = "#f59e0b",
                         midpoint = 0, limits = c(-1, 1)) +
    labs(title = heatmap_titles[group_name],
         subtitle = "Pearson / point-biserial / phi correlation",
         x = NULL, y = NULL, fill = "Correlation") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 8),
          axis.text.y = element_text(size = 8),
          panel.grid = element_blank(), plot.title.position = "plot")
  
  ggsave(file.path(heatmap_dir, paste0("correlation_heatmap_", group_name, ".png")),
         plot = p_heat, width = plot_size, height = plot_size, dpi = 300)
  message("  Saved: correlation_heatmap_", group_name)
}

# =============================================================================
# 2. BINARY FEATURE PROPORTIONS
# =============================================================================
message("Analyzing binary feature proportions...")

all_binary <- unlist(binary_features, use.names = FALSE)

binary_prop_results <- map_dfr(all_binary, function(feat) {
  sub <- df_raw |> filter(!is.na(.data[[feat]]), !is.na(label))
  props <- sub |>
    group_by(label) |>
    summarise(n = n(), n_present = sum(.data[[feat]] == 1), .groups = "drop") |>
    mutate(prop = n_present / n)
  
  tab <- table(sub[[feat]], sub$label)
  ft <- tryCatch(fisher.test(tab), error = function(e) NULL)
  
  tibble(
    feature = feat,
    prop_0 = props$prop[props$label == 0],
    prop_1 = props$prop[props$label == 1],
    prop_diff = props$prop[props$label == 1] - props$prop[props$label == 0],
    odds_ratio = if (!is.null(ft)) ft$estimate else NA_real_,
    p_value = if (!is.null(ft)) ft$p.value else NA_real_
  )
}) |>
  mutate(
    p_adj = p.adjust(p_value, method = "BH"),
    feature_label = pretty_feature_label(feature),
    region = case_when(
      str_detect(feature, "_peptide$") ~ "peptide",
      str_detect(feature, "_nflank$")  ~ "n_flank",
      str_detect(feature, "_cflank$")  ~ "c_flank",
      str_detect(feature, "_n_pep$")   ~ "transition (N→pep)",
      str_detect(feature, "_pep_c$")   ~ "transition (pep→C)"
    )
  ) |>
  arrange(p_adj)

write_csv(binary_prop_results, file.path(results_dir, paste0("binary_feature_proportions", run_suffix, ".csv")))

# Plot binary proportions
binary_plot_data <- binary_prop_results |>
  select(feature, feature_label, region, prop_0, prop_1) |>
  pivot_longer(cols = c(prop_0, prop_1), names_to = "label", values_to = "proportion") |>
  mutate(label = factor(ifelse(label == "prop_0", "Not presented", "Presented"),
                        levels = c("Not presented", "Presented")))

for (reg in unique(binary_plot_data$region)) {
  plot_sub <- binary_plot_data |> filter(region == reg)
  if (nrow(plot_sub) == 0) next
  
  p_binary <- ggplot(plot_sub, aes(x = feature_label, y = proportion, fill = label)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6, alpha = 0.85) +
    scale_fill_manual(values = c("Not presented" = "#e74c3c", "Presented" = "#2ecc71")) +
    labs(title = paste("Q8 feature prevalence —", reg),
         subtitle = "Proportion of peptides where structural element is present",
         x = NULL, y = "Proportion", fill = "Label") +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "bottom", plot.title.position = "plot")
  
  ggsave(file.path(figures_dir, paste0("binary_proportions_", sanitize_filename(reg), ".png")),
         plot = p_binary, width = 10, height = 6, dpi = 300)
}

# =============================================================================
# 3. PCA (Continuous + Sequence Encoded Features)
# =============================================================================
message("Running PCA on continuous + sequence features...")

all_continuous <- unlist(continuous_features, use.names = FALSE)
# Include the sequence features dynamically if they exist!
df_pca_input <- df_raw |> select(all_of(c(all_continuous, sequence_features)))

for (col in names(df_pca_input)) {
  df_pca_input[[col]][is.na(df_pca_input[[col]])] <- median(df_pca_input[[col]], na.rm = TRUE)
}

valid_pca_cols <- names(df_pca_input)[apply(df_pca_input, 2, var, na.rm = TRUE) > 0]

if (length(valid_pca_cols) >= 2) {
  pca_res <- prcomp(df_pca_input[, valid_pca_cols], center = TRUE, scale. = TRUE)
  n_pcs <- min(5, ncol(pca_res$x))
  pca_df <- as_tibble(pca_res$x[, 1:n_pcs])
  names(pca_df) <- paste0("PC", 1:n_pcs)
  df_raw <- bind_cols(df_raw, pca_df)
  
  write_csv(as_tibble(pca_res$rotation[, 1:n_pcs], rownames = "feature"),
            file.path(results_dir, paste0("pca_loadings", run_suffix, ".csv")))
  
  pca_var <- tibble(PC = seq_along(pca_res$sdev),
                    Variance = (pca_res$sdev^2) / sum(pca_res$sdev^2))
  
  subtitle_text <- if(length(sequence_features) > 0) {
    paste0("PCA of Structural Features + ", length(sequence_features), " ", dataset_type, " encoded sequence features")
  } else {
    "PCA of continuous NetSurfP + AlphaFold features only"
  }
  
  p_scree <- ggplot(pca_var |> slice_head(n = min(20, nrow(pca_var))),
                    aes(x = PC, y = Variance)) +
    geom_col(fill = "grey70", color = "black") +
    geom_line(color = "#e74c3c", linewidth = 1) +
    geom_point(size = 2, color = "#e74c3c") +
    scale_y_continuous(labels = scales::percent) +
    labs(title = "PCA Scree Plot", subtitle = subtitle_text,
         x = "Principal Component", y = "Variance Explained") +
    theme_bw()
  ggsave(file.path(figures_dir, paste0("pca_scree_plot", run_suffix, ".png")),
         plot = p_scree, width = 8, height = 5, dpi = 300)
  
  if (n_pcs >= 2) {
    pc1_var <- pca_var$Variance[1] * 100
    pc2_var <- pca_var$Variance[2] * 100
    
    p_pca <- ggplot(df_raw, aes(
      x = PC1, y = PC2,
      fill = factor(label, labels = c("Not presented", "Presented")),
      color = factor(label, labels = c("Not presented", "Presented"))
    )) +
      geom_point(shape = 21, alpha = 0.8, size = 1.5, stroke = 0.5) +
      scale_fill_manual(values = c("#e74c3c", "#2ecc71")) +
      scale_color_manual(values = c("#e74c3c", "#2ecc71")) +
      labs(title = "Peptide Landscape PCA",
           subtitle = subtitle_text,
           x = sprintf("PC1 (%.1f%%)", pc1_var),
           y = sprintf("PC2 (%.1f%%)", pc2_var),
           fill = "Label", color = "Label") +
      theme_bw() + theme(legend.position = "bottom")
    ggsave(file.path(figures_dir, paste0("pca_scatter", run_suffix, ".png")),
           plot = p_pca, width = 8, height = 7, dpi = 300)
  }
}

# =============================================================================
# 3b. UMAP (Continuous + Sequence Encoded Features)
# =============================================================================
message("Running UMAP on continuous + sequence features...")

# We use the uwot package for speed on 50k+ rows.
if (requireNamespace("uwot", quietly = TRUE)) {
  
  # Best practice: run UMAP on the top PCs to denoise and speed up computation.
  # We use up to the top 50 PCs (or max available if fewer).
  n_pcs_umap <- min(50, ncol(pca_res$x))
  umap_input <- pca_res$x[, 1:n_pcs_umap]
  
  set.seed(42) # Crucial for reproducible UMAP layouts
  umap_res <- uwot::umap(
    umap_input, 
    n_neighbors = 15,    # Controls local vs global structure (15 is standard)
    min_dist = 0.1,      # Controls how tightly points pack together
    metric = "euclidean",
    fast_sgd = TRUE      # Speeds up processing for large datasets
  )
  
  umap_df <- as_tibble(umap_res)
  names(umap_df) <- c("UMAP1", "UMAP2")
  
  # Bind UMAP coordinates to the main dataframe
  df_raw <- bind_cols(df_raw, umap_df)
  
  # Because 53k points will overplot heavily, we use a smaller size and alpha
  p_umap <- ggplot(df_raw, aes(
    x = UMAP1, y = UMAP2,
    fill = factor(label, labels = c("Not presented", "Presented")),
    color = factor(label, labels = c("Not presented", "Presented"))
  )) +
    geom_point(shape = 21, alpha = 0.6, size = 0.8, stroke = 0.2) +
    scale_fill_manual(values = c("#e74c3c", "#2ecc71")) +
    scale_color_manual(values = c("#e74c3c", "#2ecc71")) +
    labs(title = "Peptide Landscape UMAP",
         subtitle = paste0("UMAP based on top ", n_pcs_umap, " PCs (", subtitle_text, ")"),
         x = "UMAP 1",
         y = "UMAP 2",
         fill = "Label", color = "Label") +
    theme_bw() + 
    theme(legend.position = "bottom")
  
  ggsave(file.path(figures_dir, paste0("umap_scatter", run_suffix, ".png")),
         plot = p_umap, width = 8, height = 7, dpi = 300)
  
  message("  Saved: umap_scatter", run_suffix, ".png")
  
} else {
  message("  Skipping UMAP: 'uwot' package not installed. Run install.packages('uwot').")
}

# =============================================================================
# 4. WILCOXON TESTS (structural features only)
# =============================================================================
message("Running Wilcoxon tests on continuous features...")

wilcox_results_by_group <- imap(continuous_features, function(features, group_name) {
  if (length(features) == 0) return(NULL)
  
  res <- map_dfr(features, ~ safe_wilcox(df_raw, .x)) |>
    mutate(group = group_name,
           p_adj = p.adjust(p_value, method = "BH"),
           direction = case_when(
             smd > 0 ~ "Higher in presented",
             smd < 0 ~ "Lower in presented",
             TRUE ~ "No difference"),
           abs_smd = abs(smd)) |>
    arrange(p_adj, desc(abs_smd))
  
  write_csv(res, file.path(results_dir, paste0("wilcoxon_", group_name, run_suffix, ".csv")))
  res
})

numeric_test_results <- bind_rows(wilcox_results_by_group) |>
  mutate(p_adj_global = p.adjust(p_value, method = "BH"))
write_csv(numeric_test_results, file.path(results_dir, paste0("numeric_feature_tests", run_suffix, ".csv")))

for (group_name in names(continuous_features)) {
  plot_df <- numeric_test_results |>
    filter(group == group_name, !is.na(smd), is.finite(smd)) |>
    mutate(feature_label = pretty_feature_label(feature),
           feature_label = fct_reorder(feature_label, smd))
  if (nrow(plot_df) == 0) next
  
  p_wilcox <- ggplot(plot_df, aes(x = smd, y = feature_label, color = direction)) +
    geom_errorbarh(aes(xmin = smd_ci_low, xmax = smd_ci_high), height = 0.2, linewidth = 0.7) +
    geom_point(size = 3) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
    scale_color_manual(values = c("Higher in presented" = "#2ecc71",
                                  "Lower in presented" = "#e74c3c",
                                  "No difference" = "grey50")) +
    coord_cartesian(xlim = c(-0.3, 0.3)) +
    labs(title = paste("Wilcoxon — continuous features —", group_name),
         x = "Standardized mean difference (95% CI)", y = NULL, color = "Direction") +
    theme_bw() + theme(legend.position = "bottom")
  
  ggsave(file.path(figures_dir, paste0("wilcoxon_", group_name, ".png")),
         plot = p_wilcox, width = 8, height = 6, dpi = 300)
}

# =============================================================================
# 5. LOGISTIC REGRESSION (structural only, excludes sequences)
# =============================================================================
message("Running univariate logistic regressions...")

# Using PCs dynamically generated from previous step
all_numeric <- c(all_continuous, all_binary,
                 intersect(paste0("PC", 1:5), names(df_raw)))

logit_results <- map_dfr(all_numeric, ~ safe_logit(df_raw, .x, scale_feature = FALSE)) |>
  mutate(p_adj = p.adjust(p_value, method = "BH")) |> arrange(p_adj)
write_csv(logit_results, file.path(results_dir, paste0("logistic_regression", run_suffix, ".csv")))

logit_scaled <- map_dfr(all_numeric, ~ safe_logit(df_raw, .x, scale_feature = TRUE)) |>
  mutate(p_adj = p.adjust(p_value, method = "BH")) |> arrange(p_adj)
write_csv(logit_scaled, file.path(results_dir, paste0("logistic_regression_scaled", run_suffix, ".csv")))

# All features OR plot
all_or <- logit_scaled |>
  filter(!is.na(odds_ratio), !is.na(conf_low), !is.na(conf_high)) |>
  mutate(feature = fct_reorder(feature, odds_ratio))

max_log_dist_all <- max(abs(log10(c(all_or$conf_low, all_or$conf_high))), na.rm = TRUE) * 1.1
sym_limits_all <- c(10^(-max_log_dist_all), 10^(max_log_dist_all))

p_or_all <- ggplot(all_or, aes(x = odds_ratio, y = feature)) +
  geom_point(size = 2.5, color = "#2ecc71") +
  geom_errorbarh(aes(xmin = conf_low, xmax = conf_high), height = 0.2, color = "grey30") +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey40") +
  scale_x_log10(limits = sym_limits_all) +
  labs(title = "All structural features — univariate logistic regression (scaled)",
       x = "Odds ratio (log scale, symmetric around 1.0)", y = "Feature") +
  theme_bw()

ggsave(file.path(figures_dir, paste0("odds_ratios_all_scaled", run_suffix, ".png")),
       plot = p_or_all, width = 9, height = max(6, nrow(all_or) * 0.3 + 2), dpi = 300)

# Top 30 features OR plot
top_or <- all_or |>
  mutate(abs_log_or = abs(log(odds_ratio))) |> 
  slice_max(abs_log_or, n = 30) |>
  select(-abs_log_or) 

max_log_dist_top <- max(abs(log10(c(top_or$conf_low, top_or$conf_high))), na.rm = TRUE) * 1.1
sym_limits_top <- c(10^(-max_log_dist_top), 10^(max_log_dist_top))

p_or_top <- ggplot(top_or, aes(x = odds_ratio, y = feature)) +
  geom_point(size = 3, color = "#2ecc71") +
  geom_errorbarh(aes(xmin = conf_low, xmax = conf_high), height = 0.2, color = "grey30") +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey40") +
  scale_x_log10(limits = sym_limits_top) +
  labs(title = "Top 30 structural features — univariate logistic regression (scaled)",
       x = "Odds ratio (log scale, symmetric around 1.0)", y = "Feature") +
  theme_bw()

ggsave(file.path(figures_dir, paste0("odds_ratios_top30_scaled", run_suffix, ".png")),
       plot = p_or_top, width = 8, height = 6, dpi = 300)

# =============================================================================
# PROTEIN LENGTH BIAS DIAGNOSTIC
# =============================================================================
message("Investigating protein_length signal...")

if ("protein_length" %in% names(df_raw)) {
  
  pos_len <- df_raw |> filter(label == 1) |> pull(protein_length)
  neg_len <- df_raw |> filter(label == 0) |> pull(protein_length)
  
  ks_len <- ks.test(pos_len, neg_len)
  cliff_len <- cliff_delta_sampled(pos_len, neg_len)
  
  cat("\n=== Protein Length Bias: Effect Sizes ===\n")
  cat("KS D:          ", round(ks_len$statistic, 4), "\n")
  cat("Cliff's Delta: ", round(cliff_len$estimate, 4),
      " (", cliff_len$magnitude, ")\n")
  
  p_prot_len <- ggplot(df_raw, aes(x = protein_length,
                                   fill = factor(label, labels = c("Not presented", "Presented")))) +
    geom_density(alpha = 0.5, color = "black", linewidth = 0.3) +
    scale_fill_manual(values = c("#e74c3c", "#2ecc71")) +
    scale_x_log10(labels = scales::comma) +
    annotate("text", x = 40000, y = Inf,
             label = paste0("KS D = ", round(ks_len$statistic, 4),
                            "\nCliff's \u0394 = ", round(cliff_len$estimate, 4),
                            " (", cliff_len$magnitude, ")"),
             vjust = 1.5, hjust = 1, size = 4, fontface = "bold", colour = "#e74c3c") +
    labs(title = "Protein Length Bias: Distribution by Label",
         subtitle = paste0("Positives skew shorter | KS D = ", round(ks_len$statistic, 4),
                           " | Cliff's \u0394 = ", round(cliff_len$estimate, 4),
                           " (", cliff_len$magnitude, ")"),
         x = "Protein length (log scale)", y = "Density", fill = "Label") +
    theme_bw(base_size = 13) +
    theme(legend.position = "top", panel.grid.minor = element_blank(),
          plot.title.position = "plot")
  
  ggsave(file.path(figures_dir, "protein_length_bias_density.png"),
         plot = p_prot_len, width = 8, height = 5, dpi = 300)
  
  p_prot_len_ecdf <- ggplot(df_raw, aes(x = protein_length,
                                        colour = factor(label, labels = c("Not presented", "Presented")))) +
    stat_ecdf(linewidth = 0.8) +
    scale_colour_manual(values = c("#e74c3c", "#2ecc71")) +
    scale_x_log10(labels = scales::comma) +
    annotate("text", x = 50000, y = 0.08,
             label = paste0("KS D = ", round(ks_len$statistic, 4)),
             size = 4.5, fontface = "bold", hjust = 1, colour = "#e74c3c") +
    labs(title = "Protein Length Bias: Cumulative Distribution",
         subtitle = "Curve separation = protein length imbalance between classes",
         x = "Protein length (log scale)", y = "Cumulative Proportion", colour = "Label") +
    theme_bw(base_size = 13) +
    theme(legend.position = "top", panel.grid.minor = element_blank(),
          plot.title.position = "plot")
  
  ggsave(file.path(figures_dir, "protein_length_bias_ecdf.png"),
         plot = p_prot_len_ecdf, width = 8, height = 5, dpi = 300)
}

# =============================================================================
# RELATIVE POSITION BIAS DIAGNOSTIC
# =============================================================================
message("Investigating relative terminus distance signal...")

if (all(c("rel_distance_from_n_terminus", "rel_distance_from_c_terminus") %in% names(df_raw))) {
  
  pos_n_dist <- df_raw |> filter(label == 1) |> pull(rel_distance_from_n_terminus)
  neg_n_dist <- df_raw |> filter(label == 0) |> pull(rel_distance_from_n_terminus)
  ks_n_dist <- ks.test(pos_n_dist, neg_n_dist)
  cliff_n_dist <- cliff_delta_sampled(pos_n_dist, neg_n_dist)
  
  pos_c_dist <- df_raw |> filter(label == 1) |> pull(rel_distance_from_c_terminus)
  neg_c_dist <- df_raw |> filter(label == 0) |> pull(rel_distance_from_c_terminus)
  ks_c_dist <- ks.test(pos_c_dist, neg_c_dist)
  cliff_c_dist <- cliff_delta_sampled(pos_c_dist, neg_c_dist)
  
  p_n_dist <- ggplot(df_raw, aes(x = rel_distance_from_n_terminus,
                                 fill = factor(label, labels = c("Not presented", "Presented")))) +
    geom_density(alpha = 0.5, color = "black", linewidth = 0.3) +
    scale_fill_manual(values = c("#e74c3c", "#2ecc71")) +
    annotate("text", x = 0.95, y = Inf,
             label = paste0("KS D = ", round(ks_n_dist$statistic, 4),
                            "\nCliff's \u0394 = ", round(cliff_n_dist$estimate, 4),
                            " (", cliff_n_dist$magnitude, ")"),
             vjust = 1.5, hjust = 1, size = 4, fontface = "bold", colour = "#e74c3c") +
    labs(title = "Relative Distance from N-terminus: Distribution by Label",
         subtitle = paste0("KS D = ", round(ks_n_dist$statistic, 4),
                           " | Cliff's \u0394 = ", round(cliff_n_dist$estimate, 4),
                           " (", cliff_n_dist$magnitude, ")"),
         x = "Relative distance from N-terminus (0 = start, 1 = end)",
         y = "Density", fill = "Label") +
    theme_bw(base_size = 13) +
    theme(legend.position = "top", panel.grid.minor = element_blank(),
          plot.title.position = "plot")
  
  ggsave(file.path(figures_dir, "rel_distance_n_terminus_density.png"),
         plot = p_n_dist, width = 8, height = 5, dpi = 300)
  
  p_c_dist <- ggplot(df_raw, aes(x = rel_distance_from_c_terminus,
                                 fill = factor(label, labels = c("Not presented", "Presented")))) +
    geom_density(alpha = 0.5, color = "black", linewidth = 0.3) +
    scale_fill_manual(values = c("#e74c3c", "#2ecc71")) +
    annotate("text", x = 0.95, y = Inf,
             label = paste0("KS D = ", round(ks_c_dist$statistic, 4),
                            "\nCliff's \u0394 = ", round(cliff_c_dist$estimate, 4),
                            " (", cliff_c_dist$magnitude, ")"),
             vjust = 1.5, hjust = 1, size = 4, fontface = "bold", colour = "#e74c3c") +
    labs(title = "Relative Distance from C-terminus: Distribution by Label",
         subtitle = paste0("KS D = ", round(ks_c_dist$statistic, 4),
                           " | Cliff's \u0394 = ", round(cliff_c_dist$estimate, 4),
                           " (", cliff_c_dist$magnitude, ")"),
         x = "Relative distance from C-terminus (0 = end, 1 = start)",
         y = "Density", fill = "Label") +
    theme_bw(base_size = 13) +
    theme(legend.position = "top", panel.grid.minor = element_blank(),
          plot.title.position = "plot")
  
  ggsave(file.path(figures_dir, "rel_distance_c_terminus_density.png"),
         plot = p_c_dist, width = 8, height = 5, dpi = 300)
}

# =============================================================================
# 6. PLDDT QUADRATIC MODEL
# =============================================================================
message("Running quadratic model for mean_plddt_peptide...")

if ("mean_plddt_peptide" %in% names(df_raw)) {
  plddt_quad_df <- df_raw |> filter(!is.na(label), !is.na(mean_plddt_peptide))
  plddt_quad_model <- glm(label ~ mean_plddt_peptide + I(mean_plddt_peptide^2),
                          data = plddt_quad_df, family = binomial())
  plddt_quad_results <- tidy(plddt_quad_model) |>
    mutate(odds_ratio = exp(estimate),
           conf_low = exp(estimate - 1.96 * std.error),
           conf_high = exp(estimate + 1.96 * std.error))
  write_csv(plddt_quad_results, file.path(results_dir, paste0("mean_plddt_peptide_quadratic_model", run_suffix, ".csv")))
}

# =============================================================================
# 7. VOLCANO PLOT (combined continuous + binary)
# =============================================================================
message("Making volcano plot...")

volcano_continuous <- numeric_test_results |>
  filter(!is.na(smd), !is.na(p_adj_global), p_adj_global > 0) |>
  transmute(feature, effect_size = smd, log_p = -log10(p_adj_global), type = "continuous")

volcano_binary <- binary_prop_results |>
  filter(!is.na(prop_diff), !is.na(p_adj), p_adj > 0) |>
  transmute(feature, effect_size = prop_diff, log_p = -log10(p_adj), type = "binary")

volcano_data <- bind_rows(volcano_continuous, volcano_binary)

p_volcano <- ggplot(volcano_data, aes(x = effect_size, y = log_p, color = type)) +
  geom_point(alpha = 0.8, size = 2) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  scale_color_manual(values = c("continuous" = "#e74c3c", "binary" = "#3498db")) +
  geom_text_repel(
    data = volcano_data |> slice_max(log_p, n = 10),
    aes(label = feature), size = 3, color = "black",
    box.padding = 0.5, max.overlaps = Inf
  ) +
  labs(title = "Feature association summary",
       subtitle = "Continuous: SMD from Wilcoxon | Binary: proportion diff from Fisher",
       x = "Effect size", y = "-log10(FDR-adjusted p-value)", color = "Type") +
  theme_bw()
ggsave(file.path(figures_dir, paste0("features_volcano", run_suffix, ".png")),
       plot = p_volcano, width = 8, height = 6, dpi = 300)

# =============================================================================
# 8. PER-FEATURE PLOTS
# =============================================================================

top_plot_features <- intersect(
  c("mean_rsa_peptide", "mean_disorder_peptide",
    "mean_plddt_peptide", "sd_plddt_peptide", "min_plddt_peptide",
    "frac_disordered_peptide",
    "mean_rsa_nflank", "mean_disorder_nflank",
    "mean_plddt_nflank", "frac_disordered_nflank",
    "mean_rsa_cflank", "mean_disorder_cflank",
    "mean_plddt_cflank", "frac_disordered_cflank",
    "PC1", "PC2"),
  names(df_raw)
)

walk(top_plot_features, ~ plot_numeric_feature(df_raw, .x, per_feature_dir))

# =============================================================================
# UNIFIED FOREST PLOTS: per region (continuous + binary together)
# =============================================================================
message("Making per-region forest plots...")

forest_regions <- list(
  peptide = c(continuous_features$peptide, binary_features$peptide),
  n_flank = c(continuous_features$n_flank, binary_features$n_flank),
  c_flank = c(continuous_features$c_flank, binary_features$c_flank)
)

forest_titles <- c(
  peptide = "Univariate logistic regression — Peptide",
  n_flank = "Univariate logistic regression — N-flank",
  c_flank = "Univariate logistic regression — C-flank"
)

for (region_name in names(forest_regions)) {
  feats <- forest_regions[[region_name]]
  
  plot_df <- logit_scaled |>
    filter(feature %in% feats, !is.na(odds_ratio), !is.na(conf_low), !is.na(conf_high)) |>
    mutate(
      feature_label = pretty_feature_label(feature),
      type = ifelse(str_detect(feature, "^q8(point|trans)_"), "binary", "continuous"),
      direction = case_when(
        odds_ratio > 1 ~ "Higher in presented",
        odds_ratio < 1 ~ "Lower in presented",
        TRUE ~ "No difference"
      ),
      feature_label = fct_reorder(feature_label, odds_ratio)
    )
  
  if (nrow(plot_df) == 0) next
  
  max_log_dist <- max(abs(log10(c(plot_df$conf_low, plot_df$conf_high))), na.rm = TRUE) * 1.1
  sym_limits <- c(10^(-max_log_dist), 10^(max_log_dist))
  
  p_forest <- ggplot(plot_df, aes(x = odds_ratio, y = feature_label,
                                  color = direction, shape = type)) +
    geom_point(size = 3) +
    geom_errorbarh(aes(xmin = conf_low, xmax = conf_high), height = 0.2, linewidth = 0.5) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "grey40") +
    scale_x_log10(limits = sym_limits) +
    scale_color_manual(values = c("Higher in presented" = "#2ecc71",
                                  "Lower in presented" = "#e74c3c",
                                  "No difference" = "grey50")) +
    scale_shape_manual(values = c("continuous" = 16, "binary" = 17)) +
    labs(title = forest_titles[region_name],
         subtitle = "Scaled odds ratios | ● continuous | ▲ binary (q8point)",
         x = "Odds ratio (log scale, symmetric around 1.0)",
         y = NULL, color = "Direction", shape = "Type") +
    theme_bw() +
    theme(legend.position = "bottom", legend.box = "vertical")
  
  ggsave(file.path(figures_dir, paste0("forest_", region_name, run_suffix, ".png")),
         plot = p_forest, width = 9, height = max(4, nrow(plot_df) * 0.35 + 2), dpi = 300)
  message("  Saved: forest_", region_name)
}
message("\n✅ Done.")