#!/usr/bin/env Rscript
library(tidyverse)
library(broom)
library(ggplot2)
library(ggrepel)
source("src/functions.R")
set_working_directory()

input_file <- "data/processed/df_all.csv"
results_dir <- "results"
figures_dir <- file.path("results", "figures", "numeric_9mer")

dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

# Load data
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

# pretty feature labels for Wilcoxon
pretty_feature_label <- function(x) {
  x |>
    str_replace("_nflank$", "") |>
    str_replace("_peptide$", "") |>
    str_replace("_cflank$", "") |>
    str_replace("_full_context$", "") |>
    
    # Q8 fractions
    str_replace("^frac_q8_G$", "Frac. 3-10 helix") |>
    str_replace("^frac_q8_H$", "Frac. alpha helix") |>
    str_replace("^frac_q8_I$", "Frac. pi helix") |>
    str_replace("^frac_q8_B$", "Frac. beta bridge") |>
    str_replace("^frac_q8_E$", "Frac. beta strand") |>
    str_replace("^frac_q8_S$", "Frac. bend") |>
    str_replace("^frac_q8_T$", "Frac. turn") |>
    str_replace("^frac_q8_C$", "Frac. coil") |>
    
    # Q8 mean probabilities
    str_replace("^mean_p_q8_G$", "Mean Q8 3-10 helix prob.") |>
    str_replace("^mean_p_q8_H$", "Mean Q8 alpha helix prob.") |>
    str_replace("^mean_p_q8_I$", "Mean Q8 pi helix prob.") |>
    str_replace("^mean_p_q8_B$", "Mean Q8 beta bridge prob.") |>
    str_replace("^mean_p_q8_E$", "Mean Q8 beta strand prob.") |>
    str_replace("^mean_p_q8_S$", "Mean Q8 bend prob.") |>
    str_replace("^mean_p_q8_T$", "Mean Q8 turn prob.") |>
    str_replace("^mean_p_q8_C$", "Mean Q8 coil prob.") |>
    
    # Q3 mean probabilities
    str_replace("^mean_p_q3_H$", "Mean Q3 helix prob.") |>
    str_replace("^mean_p_q3_E$", "Mean Q3 sheet prob.") |>
    str_replace("^mean_p_q3_C$", "Mean Q3 coil prob.") |>
    
    # Other structural features
    str_replace("^mean_rsa$", "Mean RSA") |>
    str_replace("^mean_disorder$", "Mean disorder") |>
    str_replace("^frac_disordered$", "Frac. disordered") |>
    str_replace("^frac_helix$", "Frac. helix") |>
    str_replace("^frac_sheet$", "Frac. sheet") |>
    str_replace("^frac_coil$", "Frac. coil") |>
    str_replace("^mean_plddt$", "Mean pLDDT") |>
    str_replace("^min_plddt$", "Minimum pLDDT") |>
    str_replace("^sd_plddt$", "SD pLDDT") |>
    
    # General features
    str_replace("^distance_from_n_terminus$", "Dist. from N terminus") |>
    str_replace("^distance_from_c_terminus$", "Dist. from C terminus") |>
    str_replace("^protein_length$", "Protein length") |>
    str_replace("^pep_length$", "Peptide length")
}

wilcox_feature_groups <- list(
  n_flank = names(df_raw)[
    str_detect(names(df_raw), "_nflank$") &
      !str_detect(names(df_raw), "^mean_p_q3_") &
      !str_detect(names(df_raw), "^frac_(helix|sheet|coil)_")
  ],
  peptide = names(df_raw)[
    str_detect(names(df_raw), "_peptide$") &
      !str_detect(names(df_raw), "^mean_p_q3_") &
      !str_detect(names(df_raw), "^frac_(helix|sheet|coil)_")
  ],
  c_flank = names(df_raw)[
    str_detect(names(df_raw), "_cflank$") &
      !str_detect(names(df_raw), "^mean_p_q3_") &
      !str_detect(names(df_raw), "^frac_(helix|sheet|coil)_")
  ],
  general = intersect(
    c(
      names(df_raw)[
        str_detect(names(df_raw), "_full_context$") &
          !str_detect(names(df_raw), "^mean_p_q3_") &
          !str_detect(names(df_raw), "^frac_(helix|sheet|coil)_")
      ],
      "distance_from_n_terminus",
      "distance_from_c_terminus",
      "protein_length",
      "pep_length"
    ),
    names(df_raw)
  )
)

numeric_cols <- names(df_raw)[sapply(df_raw, is.numeric)]
wilcox_feature_groups <- purrr::map(wilcox_feature_groups, ~ intersect(.x, numeric_cols))


# -----------------------------------------------------------------------------
# 1. IDENTIFY NETSURFP FEATURES & CORRELATION HEATMAP
# -----------------------------------------------------------------------------
message("Generating correlation heatmaps by group...")

heatmap_feature_groups <- list(
  n_flank = names(df_raw)[
    str_detect(names(df_raw), "_nflank$") &
      !str_detect(names(df_raw), "^mean_p_q3_") &
      !str_detect(names(df_raw), "^frac_(helix|sheet|coil)_") &
      !str_detect(names(df_raw), "^frac_q8_")
  ],
  peptide = names(df_raw)[
    str_detect(names(df_raw), "_peptide$") &
      !str_detect(names(df_raw), "^mean_p_q3_") &
      !str_detect(names(df_raw), "^frac_(helix|sheet|coil)_") &
      !str_detect(names(df_raw), "^frac_q8_")
  ],
  c_flank = names(df_raw)[
    str_detect(names(df_raw), "_cflank$") &
      !str_detect(names(df_raw), "^mean_p_q3_") &
      !str_detect(names(df_raw), "^frac_(helix|sheet|coil)_") &
      !str_detect(names(df_raw), "^frac_q8_")
  ],
  general = intersect(
    c(
      names(df_raw)[
        str_detect(names(df_raw), "_full_context$") &
          !str_detect(names(df_raw), "^mean_p_q3_") &
          !str_detect(names(df_raw), "^frac_(helix|sheet|coil)_") &
          !str_detect(names(df_raw), "^frac_q8_")
      ],
      "distance_from_n_terminus",
      "distance_from_c_terminus",
      "protein_length",
      "pep_length"
    ),
    names(df_raw)
  )
)

numeric_cols <- names(df_raw)[sapply(df_raw, is.numeric)]
heatmap_feature_groups <- purrr::map(heatmap_feature_groups, ~ intersect(.x, numeric_cols))

heatmap_titles <- c(
  peptide = "Correlation Heatmap — Peptide",
  n_flank = "Correlation Heatmap — N-flank",
  c_flank = "Correlation Heatmap — C-flank",
  general = "Correlation Heatmap — General"
)

for (group_name in names(heatmap_feature_groups)) {
  
  cols <- heatmap_feature_groups[[group_name]]
  
  if (length(cols) < 2) {
    message("  Skipping ", group_name, " — fewer than 2 columns found.")
    next
  }
  
  cor_mat <- cor(df_raw[cols], use = "pairwise.complete.obs")
  
  pretty_labels <- pretty_feature_label(cols)
  rownames(cor_mat) <- pretty_labels
  colnames(cor_mat) <- pretty_labels
  
  cor_long <- as.data.frame(as.table(cor_mat)) |>
    rename(Feature_1 = Var1, Feature_2 = Var2, Correlation = Freq) |>
    filter(!is.na(Correlation))
  
  p_heat <- ggplot(cor_long, aes(x = Feature_1, y = Feature_2, fill = Correlation)) +
    geom_tile() +
    scale_fill_gradient2(
      low = "#2ecc71",
      mid = "white",
      high = "#e74c3c",
      midpoint = 0,
      limits = c(-1, 1)
    ) +
    labs(
      title = heatmap_titles[group_name],
      subtitle = "Pearson correlation across features",
      x = NULL,
      y = NULL,
      fill = "Correlation"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 8),
      axis.text.y = element_text(size = 8),
      panel.grid = element_blank(),
      plot.title.position = "plot"
    )
  
  out_file <- file.path(figures_dir, paste0("correlation_heatmap_", group_name, ".png"))
  ggsave(out_file, plot = p_heat, width = 10, height = 9, dpi = 300)
  message("  Saved: ", out_file)
}


# -----------------------------------------------------------------------------
# 2. PCA: COMBINING COMPOSITIONAL FEATURES
# -----------------------------------------------------------------------------
message("Running PCA to combine compositional NetSurfP features...")
netsurfp_cols <- unique(c(
  heatmap_feature_groups$peptide,
  heatmap_feature_groups$n_flank,
  heatmap_feature_groups$c_flank,
  heatmap_feature_groups$general
))
netsurfp_cols <- intersect(netsurfp_cols, names(df_raw))
df_pca_input <- df_raw |> select(all_of(netsurfp_cols))

# Temporary median imputation for PCA calculation only
for (col in names(df_pca_input)) {
  df_pca_input[[col]][is.na(df_pca_input[[col]])] <- median(df_pca_input[[col]], na.rm = TRUE)
}

# Drop zero-variance columns
valid_pca_cols <- names(df_pca_input)[apply(df_pca_input, 2, var, na.rm = TRUE) > 0]

# Compute PCA
pca_res <- prcomp(df_pca_input[, valid_pca_cols], center = TRUE, scale. = TRUE)

# Extract top 5 PCs and append to main dataframe
pca_df <- as_tibble(pca_res$x[, 1:5])
names(pca_df) <- paste0("NetSurfP_PC", 1:5)
df_raw <- bind_cols(df_raw, pca_df)

write_csv(as_tibble(pca_res$rotation[, 1:5], rownames = "feature"), file.path(results_dir, "netsurfp_pca_loadings.csv"))

# --- PLOT PCA RESULTS ---
message("Plotting PCA results...")

# A. Scree Plot (Variance Explained)
pca_var <- tibble(
  PC = 1:length(pca_res$sdev),
  Variance = (pca_res$sdev^2) / sum(pca_res$sdev^2)
)

p_scree <- ggplot(pca_var |> slice_head(n = 20), aes(x = PC, y = Variance)) +
  geom_col(fill = "grey70", color = "black") +
  geom_line(color = "#e74c3c", linewidth = 1) +
  geom_point(size = 2, color = "#e74c3c") +
  scale_x_continuous(breaks = 1:20) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "PCA Scree Plot: NetSurfP Combinations", x = "Principal Component", y = "Variance Explained") +
  theme_bw()

ggsave(file.path(figures_dir, "netsurfp_pca_scree_plot.png"), plot = p_scree, width = 8, height = 5, dpi = 300)

# B. Scatter Plot (PC1 vs PC2)
pc1_var <- pca_var$Variance[1] * 100
pc2_var <- pca_var$Variance[2] * 100

# NEW: Shuffle the rows randomly so the dots are mixed, preventing one class from hiding the other
set.seed(42)
df_pca_plot <- df_raw |> slice_sample(prop = 1)

p_pca_scatter <- ggplot(df_pca_plot, aes(
  x = NetSurfP_PC1, 
  y = NetSurfP_PC2, 
  # NEW: using fill and color to control the inside and outside of the dots
  fill = factor(label, levels = c(0, 1), labels = c("0_not_presented", "1_presented")),
  color = factor(label, levels = c(0, 1), labels = c("0_not_presented", "1_presented"))
)) +
  # NEW: shape = 21 allows distinct borders. alpha = 0.3 makes everything highly transparent.
  geom_point(shape = 21, alpha = 1, size = 1.5, stroke = 0.5) +
  scale_fill_manual(values = c("0_not_presented" = "#e74c3c", "1_presented" = "#2ecc71")) +
  scale_color_manual(values = c("0_not_presented" = "#e74c3c", "1_presented" = "#2ecc71")) +
  labs(
    title = "Peptide Landscape by Secondary Structure",
    subtitle = "PCA of NetSurfP Compositional Features",
    x = sprintf("NetSurfP_PC1 (%.1f%% Variance)", pc1_var),
    y = sprintf("NetSurfP_PC2 (%.1f%% Variance)", pc2_var),
    fill = "Label",
    color = "Label"
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

ggsave(file.path(figures_dir, "netsurfp_pca_scatter.png"), plot = p_pca_scatter, width = 8, height = 7, dpi = 300)

# -----------------------------------------------------------------------------
# 3. SELECT ALL NUMERIC FEATURES FOR TESTING
# -----------------------------------------------------------------------------
exclude_cols <- c(
  "label", "start", "end", "peptide", "uniprot_id", "source_molecule",
  "molecule_parent", "sequence", "n_flank", "c_flank", "full_context",
  "c_term_P1", "c_term_P1_prime", "n_term_P1", "n_term_P1_prime"
)

numeric_features <- df_raw |>
  select(where(is.numeric)) |>
  select(-any_of(exclude_cols)) |>
  names()

message("Number of numeric features (including PCs): ", length(numeric_features))

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
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
    return(tibble(
      feature = feature, n_0 = length(x0), n_1 = length(x1),
      mean_0 = NA_real_, mean_1 = NA_real_, median_0 = NA_real_, median_1 = NA_real_,
      sd_0 = NA_real_, sd_1 = NA_real_, pooled_sd = NA_real_, smd = NA_real_,
      smd_se = NA_real_, smd_ci_low = NA_real_, smd_ci_high = NA_real_,
      wilcox_statistic = NA_real_, p_value = NA_real_, median_diff = NA_real_, mean_diff = NA_real_
    ))
  }
  
  wt <- tryCatch(
    suppressWarnings(wilcox.test(x1, x0, alternative = "two.sided", exact = FALSE)),
    error = function(e) NULL
  )
  
  mean_0 <- mean(x0)
  mean_1 <- mean(x1)
  sd_0 <- sd(x0)
  sd_1 <- sd(x1)
  
  pooled_sd <- sqrt(((length(x0) - 1) * sd_0^2 + (length(x1) - 1) * sd_1^2) / (length(x0) + length(x1) - 2))
  smd <- ifelse(is.finite(pooled_sd) && pooled_sd > 0, (mean_1 - mean_0) / pooled_sd, NA_real_)
  
  n0 <- length(x0)
  n1 <- length(x1)
  
  # Approximate SE and 95% CI for Cohen's d
  smd_se <- ifelse(
    is.finite(smd),
    sqrt((n0 + n1) / (n0 * n1) + (smd^2) / (2 * (n0 + n1 - 2))),
    NA_real_
  )
  
  smd_ci_low <- ifelse(is.finite(smd_se), smd - 1.96 * smd_se, NA_real_)
  smd_ci_high <- ifelse(is.finite(smd_se), smd + 1.96 * smd_se, NA_real_)
  
  tibble(
    feature = feature, n_0 = n0, n_1 = n1,
    mean_0 = mean_0, mean_1 = mean_1, median_0 = median(x0), median_1 = median(x1),
    sd_0 = sd_0, sd_1 = sd_1, pooled_sd = pooled_sd, smd = smd,
    smd_se = smd_se, smd_ci_low = smd_ci_low, smd_ci_high = smd_ci_high,
    wilcox_statistic = ifelse(is.null(wt), NA_real_, as.numeric(wt$statistic)),
    p_value = ifelse(is.null(wt), NA_real_, wt$p.value),
    median_diff = median(x1) - median(x0), mean_diff = mean_1 - mean_0
  )
}

safe_logit <- function(data, feature, scale_feature = FALSE) {
  sub <- data |>
    select(all_of(feature), label) |>
    filter(!is.na(.data[[feature]]), !is.na(label))
  
  if (nrow(sub) < 10 || sub |> summarise(n_unique = n_distinct(.data[[feature]])) |> pull(n_unique) < 2) {
    return(tibble(
      feature = feature, n = nrow(sub), scaled = scale_feature, estimate = NA_real_,
      std_error = NA_real_, statistic = NA_real_, p_value = NA_real_,
      odds_ratio = NA_real_, conf_low = NA_real_, conf_high = NA_real_
    ))
  }
  
  if (scale_feature) {
    sub <- sub |> mutate(x = as.numeric(scale(.data[[feature]])))
    formula_used <- label ~ x
    term_name <- "x"
  } else {
    formula_used <- as.formula(paste("label ~", feature))
    term_name <- feature
  }
  
  fit <- tryCatch(
    glm(formula = formula_used, data = sub, family = binomial()),
    error = function(e) NULL
  )
  
  if (is.null(fit)) return(tibble(feature = feature, n = nrow(sub), scaled = scale_feature, estimate = NA_real_, std_error = NA_real_, statistic = NA_real_, p_value = NA_real_, odds_ratio = NA_real_, conf_low = NA_real_, conf_high = NA_real_))
  
  tid <- broom::tidy(fit) |> filter(term == term_name)
  if (nrow(tid) == 0) return(tibble(feature = feature, n = nrow(sub), scaled = scale_feature, estimate = NA_real_, std_error = NA_real_, statistic = NA_real_, p_value = NA_real_, odds_ratio = NA_real_, conf_low = NA_real_, conf_high = NA_real_))
  
  est <- tid$estimate[1]
  se <- tid$std.error[1]
  
  tibble(
    feature = feature, n = nrow(sub), scaled = scale_feature, estimate = est,
    std_error = se, statistic = tid$statistic[1], p_value = tid$p.value[1],
    odds_ratio = exp(est), conf_low = exp(est - 1.96 * se), conf_high = exp(est + 1.96 * se)
  )
}

plot_numeric_feature <- function(data, feature, outdir) {
  sub <- data |>
    select(all_of(feature), label) |>
    filter(!is.na(.data[[feature]]), !is.na(label)) |>
    mutate(label = factor(label, levels = c(0, 1), labels = c("0_not_presented", "1_presented")))
  
  if (nrow(sub) == 0) return(NULL)
  fname <- sanitize_filename(feature)
  
  p_box <- ggplot(sub, aes(x = label, y = .data[[feature]], fill = label)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.85, width = 0.5) +
    scale_fill_manual(values = c("#e74c3c", "#2ecc71")) +
    labs(title = paste("Distribution of", feature, "by label"), x = "Label", y = feature) +
    theme_bw() + theme(legend.position = "none")
  
  ggsave(file.path(outdir, paste0(fname, "_boxplot.png")), plot = p_box, width = 6, height = 4, dpi = 300)
  
  p_ecdf <- ggplot(sub, aes(x = .data[[feature]], color = label)) +
    stat_ecdf(linewidth = 1) +
    scale_color_manual(values = c("#e74c3c", "#2ecc71")) +
    labs(title = paste("ECDF of", feature, "by label"), x = feature, y = "ECDF", color = "Label") +
    theme_bw()
  
  ggsave(file.path(outdir, paste0(fname, "_ecdf.png")), plot = p_ecdf, width = 6, height = 4, dpi = 300)
  
  list(box = p_box, ecdf = p_ecdf)
}

# -----------------------------------------------------------------------------
# RUN STATISTICAL TESTS
# -----------------------------------------------------------------------------
message("Running Wilcoxon tests by region...")
wilcox_results_by_group <- purrr::imap(wilcox_feature_groups, function(features, group_name) {
  
  if (length(features) == 0) {
    message("  No features found for ", group_name)
    return(NULL)
  }
  
  res <- purrr::map_dfr(features, ~ safe_wilcox(df_raw, .x)) |>
    mutate(
      group = group_name,
      p_adj = p.adjust(p_value, method = "BH"),
      direction = case_when(
        smd > 0 ~ "Higher in presented peptides",
        smd < 0 ~ "Lower in presented peptides",
        TRUE ~ "no_difference"
      ),
      abs_smd = abs(smd)
    ) |>
    arrange(p_adj, desc(abs_smd))
  
  out_file <- file.path(results_dir, paste0("wilcoxon_", group_name, "_features.csv"))
  write_csv(res, out_file)
  message("  Saved: ", out_file)
  
  res
})

numeric_test_results <- bind_rows(wilcox_results_by_group)
write_csv(numeric_test_results, file.path(results_dir, "numeric_feature_tests_by_group.csv"))

message("Running univariate logistic regressions...")
numeric_logit_results <- purrr::map_dfr(numeric_features, ~ safe_logit(df_raw, .x, scale_feature = FALSE)) |>
  mutate(p_adj = p.adjust(p_value, method = "BH")) |> arrange(p_adj)
write_csv(numeric_logit_results, file.path(results_dir, "numeric_logistic_regression.csv"))

message("Running standardized univariate logistic regressions...")
numeric_logit_scaled_results <- purrr::map_dfr(numeric_features, ~ safe_logit(df_raw, .x, scale_feature = TRUE)) |>
  mutate(p_adj = p.adjust(p_value, method = "BH")) |> arrange(p_adj)
write_csv(numeric_logit_scaled_results, file.path(results_dir, "numeric_logistic_regression_scaled.csv"))

# -----------------------------------------------------------------------------
# PLDDT QUADRATIC MODEL
# -----------------------------------------------------------------------------
message("Running quadratic model for mean_plddt_peptide...")
plddt_quad_df <- df_raw |> filter(!is.na(label), !is.na(mean_plddt_peptide))

plddt_quad_model <- glm(label ~ mean_plddt_peptide + I(mean_plddt_peptide^2), data = plddt_quad_df, family = binomial())
plddt_quad_results <- broom::tidy(plddt_quad_model) |>
  mutate(odds_ratio = exp(estimate), conf_low = exp(estimate - 1.96 * std.error), conf_high = exp(estimate + 1.96 * std.error))
write_csv(plddt_quad_results, file.path(results_dir, "mean_plddt_peptide_quadratic_model.csv"))

# -----------------------------------------------------------------------------
# SUMMARY PLOTS & INDIVIDUAL FEATURE PLOTS
# -----------------------------------------------------------------------------
message("Making Wilcoxon plots by group...")

for (group_name in names(wilcox_feature_groups)) {
  
  plot_df <- numeric_test_results |>
    filter(group == group_name, !is.na(smd), !is.na(p_adj), is.finite(smd), is.finite(p_adj)) |>
    mutate(feature_label = pretty_feature_label(feature),
           feature_label = forcats::fct_reorder(feature_label, smd))
  
  if (nrow(plot_df) == 0) {
    message("  Skipping Wilcoxon plot for ", group_name, " — no valid results.")
    next
  }
  
  p_wilcox <- ggplot(plot_df, aes(x = smd, y = feature_label, color = direction)) +
    geom_errorbarh(aes(xmin = smd_ci_low, xmax = smd_ci_high), height = 0.2, linewidth = 0.7, alpha = 0.8) +
    geom_point(size = 3) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
    scale_color_manual(values = c(
      "Higher in presented peptides" = "#2ecc71",
      "Lower in presented peptides" = "#e74c3c",
      "No difference" = "grey50"
    )) +
    coord_cartesian(xlim = c(-0.3, 0.3)) +
    labs(
      title = paste("Wilcoxon results —", group_name),
      subtitle = "Bars show 95% confidence intervals",
      x = "Standardized mean difference (95% CI)",
      y = NULL,
      color = "Direction"
    ) +
    theme_bw() +
    theme(
      legend.position = "bottom"
    )
  
  out_file <- file.path(figures_dir, paste0("wilcoxon_", group_name, "_all_features.png"))
  ggsave(out_file, plot = p_wilcox, width = 8, height = 6, dpi = 300)
  message("  Saved: ", out_file)
}

top_or_scaled <- numeric_logit_scaled_results |>
  filter(!is.na(odds_ratio), !is.na(conf_low), !is.na(conf_high), !is.na(p_value)) |>
  slice_head(n = 20) |> mutate(feature = forcats::fct_reorder(feature, odds_ratio))

p_or_scaled <- ggplot(top_or_scaled, aes(x = odds_ratio, y = feature)) +
  geom_point(size = 3, color = "#2ecc71") +
  geom_errorbarh(aes(xmin = conf_low, xmax = conf_high), height = 0.2, color = "grey30") +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey40") + scale_x_log10() +
  labs(title = "Top numeric features by univariate logistic regression", x = "Odds ratio (log scale)", y = "Feature") + theme_bw()
ggsave(file.path(figures_dir, "top_numeric_features_odds_ratios_scaled.png"), plot = p_or_scaled, width = 8, height = 6, dpi = 300)

volcano_data <- numeric_test_results |>
  filter(!is.na(smd), !is.na(p_adj), is.finite(smd), is.finite(p_adj), p_adj > 0) |>
  mutate(log_p = -log10(p_adj))

p_volcano_adj <- ggplot(volcano_data, aes(x = smd, y = log_p)) +
  geom_point(alpha = 0.8, color = "#e74c3c") + 
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  geom_text_repel(
    data = filter(volcano_data, log_p > 100),
    aes(label = feature),
    size = 3,
    color = "black",
    box.padding = 0.5,
    max.overlaps = Inf
  ) +
  labs(
    title = "Numeric feature association summary", 
    x = "Standardized mean difference", 
    y = "-log10(FDR-adjusted p-value)"
  ) + 
  theme_bw()

ggsave(file.path(figures_dir, "numeric_features_volcano_adjusted_p.png"), plot = p_volcano_adj, width = 7, height = 5, dpi = 300)

# Per-feature plots
message("Making per-feature plots...")

top_plot_features <- c(
  "frac_disordered_peptide",
  "mean_plddt_peptide",
  "min_plddt_peptide",
  "NetSurfP_PC1",
  "NetSurfP_PC2"
)

top_plot_features <- intersect(top_plot_features, names(df_raw))

selected_plot_objects <- purrr::map(top_plot_features, ~ plot_numeric_feature(df_raw, .x, figures_dir))
names(selected_plot_objects) <- top_plot_features

message("\nDone.")

