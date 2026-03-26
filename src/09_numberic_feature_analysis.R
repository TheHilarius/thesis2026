#!/usr/bin/env Rscript
library(tidyverse)
library(broom)
library(ggplot2)
source("src/functions.R")
set_working_directory()

input_file <- "data/processed/df_all.csv"
results_dir <- "results"
figures_dir <- file.path("figures", "numeric")

dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

# Load data
df_raw <- read_csv(input_file, show_col_types = FALSE)

df_raw <- df_raw |>
  filter(!is.na(label)) |>
  mutate(label = as.integer(label)) |>
  filter(label %in% c(0, 1))

message("Label counts:")
print(table(df_raw$label))

# Select numeric features
exclude_cols <- c(
  "label",
  "start",
  "end",
  "peptide",
  "uniprot_id",
  "source_molecule",
  "molecule_parent",
  "sequence",
  "n_flank",
  "c_flank",
  "full_context",
  "c_term_P1",
  "c_term_P1_prime",
  "n_term_P1",
  "n_term_P1_prime"
)

numeric_features <- df_raw |>
  select(where(is.numeric)) |>
  select(-any_of(exclude_cols)) |>
  names()

message("Number of numeric features: ", length(numeric_features))

# Helpers
sanitize_filename <- function(x) {
  x |>
    str_replace_all("[^A-Za-z0-9_\\-]", "_")
}

safe_wilcox <- function(data, feature) {
  sub <- data |>
    select(all_of(feature), label) |>
    filter(!is.na(.data[[feature]]), !is.na(label))
  
  x0 <- sub |>
    filter(label == 0) |>
    pull(all_of(feature))
  
  x1 <- sub |>
    filter(label == 1) |>
    pull(all_of(feature))
  
  if (length(x0) < 2 || length(x1) < 2) {
    return(tibble(
      feature = feature,
      n_0 = length(x0),
      n_1 = length(x1),
      mean_0 = NA_real_,
      mean_1 = NA_real_,
      median_0 = NA_real_,
      median_1 = NA_real_,
      sd_0 = NA_real_,
      sd_1 = NA_real_,
      pooled_sd = NA_real_,
      smd = NA_real_,
      wilcox_statistic = NA_real_,
      p_value = NA_real_,
      median_diff = NA_real_,
      mean_diff = NA_real_
    ))
  }
  
  wt <- tryCatch(
    suppressWarnings(wilcox.test(x1, x0, alternative = "two.sided", exact = FALSE)),
    error = function(e) NULL
  )
  
  mean_0 <- mean(x0)
  mean_1 <- mean(x1)
  median_0 <- median(x0)
  median_1 <- median(x1)
  sd_0 <- sd(x0)
  sd_1 <- sd(x1)
  
  pooled_sd <- sqrt(((length(x0) - 1) * sd_0^2 + (length(x1) - 1) * sd_1^2) / (length(x0) + length(x1) - 2))
  smd <- ifelse(is.finite(pooled_sd) && pooled_sd > 0, (mean_1 - mean_0) / pooled_sd, NA_real_)
  
  tibble(
    feature = feature,
    n_0 = length(x0),
    n_1 = length(x1),
    mean_0 = mean_0,
    mean_1 = mean_1,
    median_0 = median_0,
    median_1 = median_1,
    sd_0 = sd_0,
    sd_1 = sd_1,
    pooled_sd = pooled_sd,
    smd = smd,
    wilcox_statistic = ifelse(is.null(wt), NA_real_, as.numeric(wt$statistic)),
    p_value = ifelse(is.null(wt), NA_real_, wt$p.value),
    median_diff = median_1 - median_0,
    mean_diff = mean_1 - mean_0
  )
}

safe_logit <- function(data, feature, scale_feature = FALSE) {
  sub <- data |>
    select(all_of(feature), label) |>
    filter(!is.na(.data[[feature]]), !is.na(label))
  
  if (nrow(sub) < 10) {
    return(tibble(
      feature = feature,
      n = nrow(sub),
      scaled = scale_feature,
      estimate = NA_real_,
      std_error = NA_real_,
      statistic = NA_real_,
      p_value = NA_real_,
      odds_ratio = NA_real_,
      conf_low = NA_real_,
      conf_high = NA_real_
    ))
  }
  
  if (sub |> summarise(n_unique = n_distinct(.data[[feature]])) |> pull(n_unique) < 2) {
    return(tibble(
      feature = feature,
      n = nrow(sub),
      scaled = scale_feature,
      estimate = NA_real_,
      std_error = NA_real_,
      statistic = NA_real_,
      p_value = NA_real_,
      odds_ratio = NA_real_,
      conf_low = NA_real_,
      conf_high = NA_real_
    ))
  }
  
  if (scale_feature) {
    sub <- sub |>
      mutate(x = as.numeric(scale(.data[[feature]])))
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
  
  if (is.null(fit)) {
    return(tibble(
      feature = feature,
      n = nrow(sub),
      scaled = scale_feature,
      estimate = NA_real_,
      std_error = NA_real_,
      statistic = NA_real_,
      p_value = NA_real_,
      odds_ratio = NA_real_,
      conf_low = NA_real_,
      conf_high = NA_real_
    ))
  }
  
  tid <- broom::tidy(fit) |>
    filter(term == term_name)
  
  if (nrow(tid) == 0) {
    return(tibble(
      feature = feature,
      n = nrow(sub),
      scaled = scale_feature,
      estimate = NA_real_,
      std_error = NA_real_,
      statistic = NA_real_,
      p_value = NA_real_,
      odds_ratio = NA_real_,
      conf_low = NA_real_,
      conf_high = NA_real_
    ))
  }
  
  est <- tid$estimate[1]
  se <- tid$std.error[1]
  ci_low <- est - 1.96 * se
  ci_high <- est + 1.96 * se
  
  tibble(
    feature = feature,
    n = nrow(sub),
    scaled = scale_feature,
    estimate = est,
    std_error = se,
    statistic = tid$statistic[1],
    p_value = tid$p.value[1],
    odds_ratio = exp(est),
    conf_low = exp(ci_low),
    conf_high = exp(ci_high)
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
    scale_fill_manual(values = c("#D55E00", "#0072B2")) +
    labs(
      title = paste("Distribution of", feature, "by label"),
      x = "Label",
      y = feature
    ) +
    theme_bw() +
    theme(legend.position = "none")
  
  ggsave(
    filename = file.path(outdir, paste0(fname, "_boxplot.png")),
    plot = p_box,
    width = 6,
    height = 4,
    dpi = 300
  )
  
  p_ecdf <- ggplot(sub, aes(x = .data[[feature]], color = label)) +
    stat_ecdf(linewidth = 1) +
    scale_color_manual(values = c("#D55E00", "#0072B2")) +
    labs(
      title = paste("ECDF of", feature, "by label"),
      subtitle = "Empirical cumulative distribution function",
      x = feature,
      y = "ECDF",
      color = "Label"
    ) +
    theme_bw()
  
  ggsave(
    filename = file.path(outdir, paste0(fname, "_ecdf.png")),
    plot = p_ecdf,
    width = 6,
    height = 4,
    dpi = 300
  )
  
  list(box = p_box, ecdf = p_ecdf)
}

# Wilcoxon tests
message("Running Wilcoxon tests...")

numeric_test_results <- purrr::map_dfr(numeric_features, ~ safe_wilcox(df_raw, .x)) |>
  mutate(
    p_adj = p.adjust(p_value, method = "BH"),
    direction = case_when(
      smd > 0 ~ "higher_in_label_1",
      smd < 0 ~ "lower_in_label_1",
      TRUE ~ "no_difference"
    ),
    abs_smd = abs(smd)
  ) |>
  arrange(p_adj, desc(abs_smd))

write_csv(
  numeric_test_results,
  file.path(results_dir, "numeric_feature_tests.csv")
)

message("Saved: ", file.path(results_dir, "numeric_feature_tests.csv"))

# Logistic regression, raw scale
message("Running univariate logistic regressions...")

numeric_logit_results <- purrr::map_dfr(numeric_features, ~ safe_logit(df_raw, .x, scale_feature = FALSE)) |>
  mutate(
    p_adj = p.adjust(p_value, method = "BH")
  ) |>
  arrange(p_adj)

write_csv(
  numeric_logit_results,
  file.path(results_dir, "numeric_logistic_regression.csv")
)

message("Saved: ", file.path(results_dir, "numeric_logistic_regression.csv"))

# Logistic regression, standardized
message("Running standardized univariate logistic regressions...")

numeric_logit_scaled_results <- purrr::map_dfr(numeric_features, ~ safe_logit(df_raw, .x, scale_feature = TRUE)) |>
  mutate(
    p_adj = p.adjust(p_value, method = "BH")
  ) |>
  arrange(p_adj)

write_csv(
  numeric_logit_scaled_results,
  file.path(results_dir, "numeric_logistic_regression_scaled.csv")
)

message("Saved: ", file.path(results_dir, "numeric_logistic_regression_scaled.csv"))

# Quadratic model for mean_plddt_peptide
message("Running quadratic model for mean_plddt_peptide...")

plddt_quad_df <- df_raw |>
  select(label, mean_plddt_peptide) |>
  filter(!is.na(label), !is.na(mean_plddt_peptide))

plddt_quad_model <- glm(
  label ~ mean_plddt_peptide + I(mean_plddt_peptide^2),
  data = plddt_quad_df,
  family = binomial()
)

plddt_quad_results <- broom::tidy(plddt_quad_model) |>
  mutate(
    odds_ratio = exp(estimate),
    conf_low = exp(estimate - 1.96 * std.error),
    conf_high = exp(estimate + 1.96 * std.error)
  )

write_csv(
  plddt_quad_results,
  file.path(results_dir, "mean_plddt_peptide_quadratic_model.csv")
)

message("Saved: ", file.path(results_dir, "mean_plddt_peptide_quadratic_model.csv"))

# Summary plots
message("Making summary plots...")

# Wilcoxon-style summary using standardized mean difference
top_smd <- numeric_test_results |>
  filter(!is.na(smd), !is.na(p_value), is.finite(smd), is.finite(p_value)) |>
  slice_head(n = 20) |>
  mutate(feature = forcats::fct_reorder(feature, smd))

max_abs_smd <- max(abs(top_smd$smd), na.rm = TRUE)

p_wilcox <- ggplot(top_smd, aes(x = smd, y = feature, color = direction)) +
  geom_point(size = 3) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  scale_color_manual(values = c(
    "higher_in_label_1" = "#0072B2",
    "lower_in_label_1" = "#D55E00",
    "no_difference" = "grey50"
  )) +
  scale_x_continuous(limits = c(-max_abs_smd, max_abs_smd)) +
  labs(
    title = "Top numeric features by Wilcoxon test",
    x = "Standardized mean difference",
    y = "Feature",
    color = "Direction"
  ) +
  theme_bw()

ggsave(
  filename = file.path(figures_dir, "top_numeric_features_wilcoxon.png"),
  plot = p_wilcox,
  width = 8,
  height = 6,
  dpi = 300
)

# Logistic regression summary using standardized features
top_or_scaled <- numeric_logit_scaled_results |>
  filter(!is.na(odds_ratio), !is.na(conf_low), !is.na(conf_high), !is.na(p_value)) |>
  slice_head(n = 20) |>
  mutate(feature = forcats::fct_reorder(feature, odds_ratio))

log_max <- max(abs(log(c(top_or_scaled$conf_low, top_or_scaled$conf_high))), na.rm = TRUE)
x_limits_or <- exp(c(-log_max, log_max))

p_or_scaled <- ggplot(top_or_scaled, aes(x = odds_ratio, y = feature)) +
  geom_point(size = 3, color = "#0072B2") +
  geom_errorbarh(aes(xmin = conf_low, xmax = conf_high), height = 0.2, color = "grey30") +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey40") +
  scale_x_log10(limits = x_limits_or) +
  labs(
    title = "Top numeric features by univariate logistic regression",
    subtitle = "Odds ratio per 1 standard deviation increase in feature",
    x = "Odds ratio (log scale)",
    y = "Feature"
  ) +
  theme_bw()

ggsave(
  filename = file.path(figures_dir, "top_numeric_features_odds_ratios_scaled.png"),
  plot = p_or_scaled,
  width = 8,
  height = 6,
  dpi = 300
)

# Volcano plot with raw p-values
p_volcano_raw <- numeric_test_results |>
  filter(!is.na(smd), !is.na(p_value), is.finite(smd), is.finite(p_value), p_value > 0) |>
  ggplot(aes(x = smd, y = -log10(p_value))) +
  geom_point(alpha = 0.8, color = "#0072B2") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  labs(
    title = "Numeric feature association summary",
    subtitle = "Standardized mean difference vs raw Wilcoxon p-value",
    x = "Standardized mean difference",
    y = "-log10(raw p-value)"
  ) +
  theme_bw()

ggsave(
  filename = file.path(figures_dir, "numeric_features_volcano_raw_p.png"),
  plot = p_volcano_raw,
  width = 7,
  height = 5,
  dpi = 300
)

# Volcano plot with adjusted p-values
p_volcano_adj <- numeric_test_results |>
  filter(!is.na(smd), !is.na(p_adj), is.finite(smd), is.finite(p_adj), p_adj > 0) |>
  ggplot(aes(x = smd, y = -log10(p_adj))) +
  geom_point(alpha = 0.8, color = "#D55E00") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  labs(
    title = "Numeric feature association summary",
    subtitle = "Standardized mean difference vs FDR-adjusted Wilcoxon p-value",
    x = "Standardized mean difference",
    y = "-log10(FDR-adjusted p-value)"
  ) +
  theme_bw()

ggsave(
  filename = file.path(figures_dir, "numeric_features_volcano_adjusted_p.png"),
  plot = p_volcano_adj,
  width = 7,
  height = 5,
  dpi = 300
)

# Per-feature plots
message("Making per-feature plots...")

top_plot_features <- c(
  "frac_disordered_peptide",
  "mean_plddt_peptide",
  "mean_plddt_full_context",
  "min_plddt_peptide",
  "sd_plddt_peptide",
  "pep_length"
)

selected_plot_objects <- purrr::map(top_plot_features, ~ plot_numeric_feature(df_raw, .x, figures_dir))
names(selected_plot_objects) <- top_plot_features

# Display selected plots in RStudio
message("Displaying selected plots in RStudio...")

print(p_wilcox)
print(p_or_scaled)
print(p_volcano_raw)
print(p_volcano_adj)

for (feat in top_plot_features) {
  plots <- selected_plot_objects[[feat]]
  if (!is.null(plots)) {
    print(plots$box)
    print(plots$ecdf)
  }
}

# Console summary
message("\nTop numeric features from Wilcoxon analysis:")
numeric_test_results |>
  select(feature, p_value, p_adj, smd, mean_0, mean_1, median_0, median_1, direction) |>
  slice_head(n = 15) |>
  print(n = 15)

message("\nTop numeric features from standardized logistic regression:")
numeric_logit_scaled_results |>
  select(feature, p_value, p_adj, odds_ratio, conf_low, conf_high) |>
  slice_head(n = 15) |>
  print(n = 15)

message("\nQuadratic model for mean_plddt_peptide:")
print(plddt_quad_results)

message("\nDone.")