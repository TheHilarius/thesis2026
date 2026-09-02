#!/usr/bin/env Rscript
# =============================================================================
# 02_pca_optimization_analysis.R
# Investigate PCA optimization results: selection logic, sensitivity, and
# whether performance differences between configurations are meaningful.
#
# Input:  results/figures/models/pca_optimization/joint_pca_results_handcrafted_sparse_{esmc,esmif}.json
# Output: results/figures/models/pca_optimization/pca_sensitivity_analysis.png
# =============================================================================

library(jsonlite)
library(ggplot2)
library(patchwork)
library(dplyr)

# ── Paths ─────────────────────────────────────────────────────────────────────
base   <- "results/figures/models/pca_optimization"
out_png <- file.path(base, "pca_sensitivity_analysis.png")

# ── Load JSON helpers ─────────────────────────────────────────────────────────
parse_tuple <- function(s) {
  cleaned <- gsub("[() ]", "", s)
  as.numeric(strsplit(cleaned, ",")[[1]])
}

load_results <- function(json_path) {
  raw <- jsonlite::fromJSON(json_path, simplifyVector = FALSE)
  feature_set <- raw$feature_set
  agg <- raw$aggregated

  parse_model <- function(model_data, model_name) {
    ps <- model_data$pooled_scores

    # Pooled scores data frame (125 rows)
    pooled_df <- data.frame(
      combo_str  = names(ps),
      pooled_auc = as.numeric(unlist(ps)),
      stringsAsFactors = FALSE
    )
    pooled_df$peptide_pca <- sapply(pooled_df$combo_str, function(s) parse_tuple(s)[1])
    pooled_df$nflank_pca  <- sapply(pooled_df$combo_str, function(s) parse_tuple(s)[2])
    pooled_df$cflank_pca  <- sapply(pooled_df$combo_str, function(s) parse_tuple(s)[3])
    pooled_df$feature_set <- feature_set
    pooled_df$model       <- model_name

    # Fold-level data frame (6 rows)
    fold_df <- data.frame(
      outer_fold  = 0:5,
      peptide_pca = sapply(model_data$best_combos, function(x) x[[1]]),
      nflank_pca  = sapply(model_data$best_combos, function(x) x[[2]]),
      cflank_pca  = sapply(model_data$best_combos, function(x) x[[3]]),
      inner_auc   = as.numeric(model_data$best_inner_aucs),
      stringsAsFactors = FALSE
    )
    fold_df$combo_label <- sprintf("(%d,%d,%d)",
                                   fold_df$peptide_pca,
                                   fold_df$nflank_pca,
                                   fold_df$cflank_pca)
    fold_df$feature_set <- feature_set
    fold_df$model       <- model_name

    # Optimal info
    best_combo <- unlist(model_data$best_pooled_combo)
    best_auc   <- model_data$best_pooled_auc
    best_label <- sprintf("(%s)", paste(best_combo, collapse = ","))

    list(
      pooled = pooled_df,
      folds  = fold_df,
      best_combo = best_combo,
      best_auc   = best_auc,
      best_label = best_label
    )
  }

  rf <- parse_model(agg$rf, "rf")
  lr <- parse_model(agg$lr, "lr")

  list(
    pooled = bind_rows(rf$pooled, lr$pooled),
    folds  = bind_rows(rf$folds, lr$folds),
    rf = rf, lr = lr
  )
}

esmc  <- load_results(file.path(base, "joint_pca_results_handcrafted_sparse_esmc.json"))
esmif <- load_results(file.path(base, "joint_pca_results_handcrafted_sparse_esmif.json"))

# ── Plot 1: Heatmap — Pooled AUC by region pair ──────────────────────────────
plot_heatmap <- function(result, title_suffix) {
  df <- result$pooled
  best_combo <- result$rf$best_combo  # just use RF for now, will facet by model

  plots <- list()
  for (mdl in c("rf", "lr")) {
    mdf <- df %>% filter(model == mdl)
    best <- result[[mdl]]$best_combo
    best_label <- result[[mdl]]$best_label

    combos <- list(
      list(x = "peptide_pca", y = "nflank_pca",
           xlab = "Peptide PCs", ylab = "N-flank PCs"),
      list(x = "peptide_pca", y = "cflank_pca",
           xlab = "Peptide PCs", ylab = "C-flank PCs"),
      list(x = "nflank_pca", y = "cflank_pca",
           xlab = "N-flank PCs", ylab = "C-flank PCs")
    )

    for (cb in combos) {
      heat_df <- mdf %>%
        group_by(.data[[cb$x]], .data[[cb$y]]) %>%
        summarise(mean_auc = mean(pooled_auc), .groups = "drop")

      # Best cell coords
      best_x <- best[match(cb$x, c("peptide_pca", "nflank_pca", "cflank_pca"))]
      best_y <- best[match(cb$y, c("peptide_pca", "nflank_pca", "cflank_pca"))]

      p <- ggplot(heat_df, aes(x = .data[[cb$x]], y = .data[[cb$y]], fill = mean_auc)) +
        geom_tile(color = "white", linewidth = 0.3) +
        geom_text(aes(label = sprintf("%.4f", mean_auc)), size = 2.2) +
        annotate("point", x = best_x, y = best_y,
                 shape = 8, size = 3.5, color = "red", stroke = 1.2) +
        scale_fill_viridis_c(option = "C", name = "AUC") +
        scale_x_continuous(breaks = seq(5, 25, 5)) +
        scale_y_continuous(breaks = seq(5, 25, 5)) +
        labs(x = cb$xlab, y = cb$ylab,
             title = toupper(mdl),
             subtitle = paste0("Fix: ", setdiff(c("peptide_pca","nflank_pca","cflank_pca"),
                                                c(cb$x, cb$y)))) +
        theme_minimal(base_size = 8) +
        theme(
          plot.title = element_text(size = 8, face = "bold"),
          plot.subtitle = element_text(size = 7),
          panel.grid = element_blank(),
          axis.title = element_text(size = 7),
          axis.text = element_text(size = 6),
          legend.key.height = unit(0.5, "cm")
        )
      plots[[paste0(mdl, "_", cb$x, "_", cb$y)]] <- p
    }
  }

  # Arrange: 2 rows (rf, lr) × 3 cols (3 region pairs)
  wrap_plots(plots, ncol = 3) +
    plot_annotation(
      title = paste("Region Pair Sensitivity —", title_suffix),
      subtitle = "Each cell = mean pooled AUC over held-out region. Star = winning combo.",
      theme = theme(
        plot.title = element_text(size = 11, face = "bold"),
        plot.subtitle = element_text(size = 9)
      )
    )
}

p_heatmap_esmc  <- plot_heatmap(esmc, "ESM-C")
p_heatmap_esmif <- plot_heatmap(esmif, "ESM-IF")

# ── Plot 2: Per-fold winners vs pooled winner ────────────────────────────────
plot_fold_winners <- function(result, title_suffix) {
  fold_df <- result$folds

  # Add best labels per model
  best_labels <- data.frame(
    model = c("rf", "lr"),
    best_label = c(result$rf$best_label, result$lr$best_label),
    best_auc = c(result$rf$best_auc, result$lr$best_auc),
    stringsAsFactors = FALSE
  )

  fold_df <- fold_df %>%
    left_join(best_labels, by = "model") %>%
    mutate(matches_best = (combo_label == best_label))

  ggplot(fold_df, aes(x = factor(outer_fold), y = inner_auc, fill = matches_best)) +
    geom_col(width = 0.7) +
    geom_text(aes(label = combo_label), size = 2.5, angle = 90, hjust = 1.1, vjust = 0.5) +
    geom_hline(data = best_labels, aes(yintercept = best_auc),
               linetype = "dashed", color = "red", linewidth = 0.5, inherit.aes = FALSE) +
    geom_text(data = best_labels,
              aes(x = 5.7, y = best_auc, label = paste("Pooled:", best_label)),
              hjust = 1, vjust = -0.5, size = 2.8, color = "red", inherit.aes = FALSE) +
    facet_wrap(~ model, ncol = 1) +
    scale_fill_manual(values = c("TRUE" = "#2196F3", "FALSE" = "#FF9800"),
                      labels = c("FALSE" = "Differs from pooled", "TRUE" = "Matches pooled"),
                      name = NULL) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.15))) +
    labs(x = "Outer Fold", y = "Inner CV AUC",
         title = paste("Per-Fold Winners vs Pooled Winner —", title_suffix),
         subtitle = "Bars show each fold's best 3-tuple. Orange = disagrees with pooled winner.") +
    theme_minimal(base_size = 10) +
    theme(
      plot.title = element_text(size = 11, face = "bold"),
      legend.position = "top"
    )
}

p_fold_esmc  <- plot_fold_winners(esmc, "ESM-C")
p_fold_esmif <- plot_fold_winners(esmif, "ESM-IF")

# ── Plot 3: Score distribution — all 125 combos ranked ───────────────────────
plot_score_dist <- function(result, title_suffix) {
  plots <- list()
  for (mdl in c("rf", "lr")) {
    df <- result[[mdl]]$pooled %>% filter(model == mdl)
    best_label <- result[[mdl]]$best_label
    best_auc   <- result[[mdl]]$best_auc

    ranked <- df %>% arrange(desc(pooled_auc)) %>% mutate(rank = row_number())
    med <- ranked %>% slice(63)

    spread <- max(ranked$pooled_auc) - min(ranked$pooled_auc)
    spread_pct <- spread / min(ranked$pooled_auc) * 100

    p <- ggplot(ranked, aes(x = rank, y = pooled_auc)) +
      geom_point(size = 1.5, alpha = 0.7, color = "grey40") +
      geom_point(data = data.frame(rank = 1, pooled_auc = best_auc),
                 size = 4, shape = 18, color = "red") +
      geom_hline(yintercept = best_auc, linetype = "dashed", color = "red", alpha = 0.5) +
      annotate("text", x = 2, y = best_auc,
               label = sprintf("Winner: %s (%.4f)", best_label, best_auc),
               hjust = 0, vjust = -1, size = 3, color = "red") +
      annotate("text", x = 10, y = med$pooled_auc + 0.0003,
               label = sprintf("Median: %.4f", med$pooled_auc),
               size = 3, color = "blue") +
      annotate("text", x = 10, y = min(ranked$pooled_auc),
               label = sprintf("Spread: %.4f (%.2f%%)", spread, spread_pct),
               size = 3, hjust = 0, vjust = 1, color = "grey30") +
      labs(x = "Combo Rank (1 = best)", y = "Pooled AUC",
           title = toupper(mdl)) +
      theme_minimal(base_size = 9) +
      theme(plot.title = element_text(size = 10, face = "bold"))

    plots[[mdl]] <- p
  }

  wrap_plots(plots, ncol = 1) +
    plot_annotation(
      title = paste("Score Distribution —", title_suffix),
      subtitle = "All 125 combinations ranked. Red = winner. Small spread = PCA choice has limited impact.",
      theme = theme(
        plot.title = element_text(size = 11, face = "bold"),
        plot.subtitle = element_text(size = 9)
      )
    )
}

p_dist_esmc  <- plot_score_dist(esmc, "ESM-C")
p_dist_esmif <- plot_score_dist(esmif, "ESM-IF")

# ── Plot 4: Region marginal effects ──────────────────────────────────────────
plot_marginal <- function(result, title_suffix) {
  plots <- list()
  for (mdl in c("rf", "lr")) {
    df <- result[[mdl]]$pooled
    best <- result[[mdl]]$best_combo

    # Marginal means per PC count per region
    marginal <- bind_rows(
      df %>% group_by(peptide_pca) %>%
        summarise(mean_auc = mean(pooled_auc), .groups = "drop") %>%
        mutate(region = "Peptide", pc_count = peptide_pca, optimal = pc_count == best[1]),
      df %>% group_by(nflank_pca) %>%
        summarise(mean_auc = mean(pooled_auc), .groups = "drop") %>%
        mutate(region = "N-flank", pc_count = nflank_pca, optimal = pc_count == best[2]),
      df %>% group_by(cflank_pca) %>%
        summarise(mean_auc = mean(pooled_auc), .groups = "drop") %>%
        mutate(region = "C-flank", pc_count = cflank_pca, optimal = pc_count == best[3])
    )

    # Compute range for annotation
    region_ranges <- marginal %>%
      group_by(region) %>%
      summarise(
        auc_range = max(mean_auc) - min(mean_auc),
        .groups = "drop"
      )

    p <- ggplot(marginal, aes(x = factor(pc_count), y = mean_auc, fill = optimal)) +
      geom_col(width = 0.6) +
      geom_text(aes(label = sprintf("%.4f", mean_auc)), size = 2.5, vjust = -0.3) +
      geom_text(data = region_ranges,
                aes(x = 3, y = Inf, label = sprintf("Range: %.4f", auc_range)),
                vjust = 2, size = 2.8, color = "grey30", inherit.aes = FALSE) +
      facet_wrap(~ region, scales = "free_y") +
      scale_fill_manual(values = c("TRUE" = "#4CAF50", "FALSE" = "#BDBDBD"),
                        labels = c("TRUE" = "Optimal", "FALSE" = "Other"),
                        name = NULL) +
      labs(x = "PC Count", y = "Mean Pooled AUC",
           title = toupper(mdl),
           subtitle = "Green = optimal. Range = max - min mean AUC across PC counts.") +
      theme_minimal(base_size = 9) +
      theme(
        plot.title = element_text(size = 10, face = "bold"),
        legend.position = "top"
      )
    plots[[mdl]] <- p
  }

  wrap_plots(plots, ncol = 1) +
    plot_annotation(
      title = paste("Region Marginal Effects —", title_suffix),
      subtitle = "Mean AUC per PC count, averaged over other regions. Small range = region PC count barely matters.",
      theme = theme(
        plot.title = element_text(size = 11, face = "bold"),
        plot.subtitle = element_text(size = 9)
      )
    )
}

p_marginal_esmc  <- plot_marginal(esmc, "ESM-C")
p_marginal_esmif <- plot_marginal(esmif, "ESM-IF")

# ── Plot 5: Sensitivity — ΔAUC when varying one region ───────────────────────
plot_sensitivity <- function(result, title_suffix) {
  plots <- list()
  for (mdl in c("rf", "lr")) {
    df <- result[[mdl]]$pooled
    best <- result[[mdl]]$best_combo

    vary <- bind_rows(
      df %>% filter(nflank_pca == best[2], cflank_pca == best[3]) %>%
        mutate(region = "Peptide", varied_pc = peptide_pca),
      df %>% filter(peptide_pca == best[1], cflank_pca == best[3]) %>%
        mutate(region = "N-flank", varied_pc = nflank_pca),
      df %>% filter(peptide_pca == best[1], nflank_pca == best[2]) %>%
        mutate(region = "C-flank", varied_pc = cflank_pca)
    ) %>%
      mutate(is_optimal = (varied_pc == best[match(region,
                  c("Peptide","N-flank","C-flank"))[1]]))

    # Compute ΔAUC for annotation
    delta_df <- vary %>%
      group_by(region) %>%
      summarise(
        delta_auc = max(pooled_auc) - min(pooled_auc),
        .groups = "drop"
      )

    p <- ggplot(vary, aes(x = factor(varied_pc), y = pooled_auc, group = region)) +
      geom_line(linewidth = 0.8, color = "grey40") +
      geom_point(aes(color = is_optimal), size = 3) +
      geom_text(aes(label = sprintf("%.4f", pooled_auc)), size = 2.5, vjust = -1) +
      geom_text(data = delta_df,
                aes(x = 3, y = Inf, label = sprintf("ΔAUC: %.4f", delta_auc)),
                vjust = 2, size = 2.8, color = "grey30", inherit.aes = FALSE) +
      facet_wrap(~ region, scales = "free_x") +
      scale_color_manual(values = c("TRUE" = "red", "FALSE" = "grey40"),
                         labels = c("TRUE" = "Optimal", "FALSE" = "Other"),
                         name = NULL) +
      labs(x = "PC Count (other two fixed at optimal)",
           y = "Pooled AUC",
           title = toupper(mdl),
           subtitle = sprintf("Fix: (%s). ΔAUC = spread when varying this region.",
                              paste(best, collapse = ","))) +
      theme_minimal(base_size = 9) +
      theme(
        plot.title = element_text(size = 10, face = "bold"),
        legend.position = "top"
      )
    plots[[mdl]] <- p
  }

  wrap_plots(plots, ncol = 1) +
    plot_annotation(
      title = paste("Sensitivity —", title_suffix),
      subtitle = "How much does each region's PC count affect performance? Flat line = doesn't matter.",
      theme = theme(
        plot.title = element_text(size = 11, face = "bold"),
        plot.subtitle = element_text(size = 9)
      )
    )
}

p_sens_esmc  <- plot_sensitivity(esmc, "ESM-C")
p_sens_esmif <- plot_sensitivity(esmif, "ESM-IF")

# ── Combine and save ─────────────────────────────────────────────────────────
combined <- (
  p_heatmap_esmc / p_fold_esmc / p_dist_esmc / p_marginal_esmc / p_sens_esmc
) | (
  p_heatmap_esmif / p_fold_esmif / p_dist_esmif / p_marginal_esmif / p_sens_esmif
)

ggsave(
  filename = out_png,
  plot = combined,
  width = 20,
  height = 35,
  dpi = 300,
  bg = "white"
)

cat("Saved:", out_png, "\n")
