#!/usr/bin/env Rscript

# =============================================================================
# 12_sequence_logos.R — Sequence logo set (overall results, not run-specific)
#
# Outputs (all results/figures/logo/, no overwrites of existing files):
#   differential_logo_29mer.png    diff PPM (Presented - Not Presented)
#   logo_presented_29mer.png       bits logo, label=1 only
#   logo_not_presented_29mer.png   bits logo, label=0 only
#   differential_logo_by_fold.png  6-panel diff logo (fold stability QC)
#
# PPM math copied verbatim from 10_categorical_residue_analysis.R (sec. 6):
# per-column NA drop so terminus gaps don't skew probabilities.
# Fold panels come from df_all_with_folds.csv (homology-aware folds 0-5).
# =============================================================================

library(tidyverse)
library(ggseqlogo)
source("src/pipeline/r/functions.R")
set_working_directory()

dir.create("results/figures/logo", showWarnings = FALSE, recursive = TRUE)

# -----------------------------------------------------------------------------
# 1. Data
# -----------------------------------------------------------------------------
df_folds <- read_csv("data/processed/df_all_with_folds.csv", show_col_types = FALSE)

n_flank_features <- paste0("N", 10:1)
peptide_features <- paste0("P", 1:9)
c_flank_features <- paste0("C", 1:10)
all_pos_features <- c(n_flank_features, peptide_features, c_flank_features)

aa_alphabet <- c("A","C","D","E","F","G","H","I","K","L",
                 "M","N","P","Q","R","S","T","V","W","Y")

df_clean <- df_folds %>%
  filter(!is.na(label)) %>%
  mutate(label = factor(label, levels = c(0, 1),
                        labels = c("Not Presented", "Presented"))) %>%
  mutate(across(all_of(all_pos_features), as.character))

cat("Rows:", nrow(df_clean), "\n")
print(table(df_clean$label, df_clean$fold))

# -----------------------------------------------------------------------------
# 2. PPM helper (verbatim from script 10)
# -----------------------------------------------------------------------------
get_ppm_for_label <- function(df, label_val, cols, alphabet) {
  sub_df <- df |> filter(label == label_val)
  ppm <- sapply(cols, function(col) {
    vals <- na.omit(sub_df[[col]])
    counts <- table(factor(vals, levels = alphabet))
    counts / sum(counts)
  })
  rownames(ppm) <- alphabet
  return(ppm)
}

theme_logo_slide <- function() {
  theme_logo() +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5, color = "grey30"),
      axis.text.x = element_text(size = 11, face = "bold", angle = 0),
      panel.grid.major.y = element_line(color = "grey90")
    )
}

lab_slide <- function(title, subtitle, y) {
  labs(title = title, subtitle = subtitle, y = y,
       x = "Sequence Position (N-Flank → Peptide → C-Flank)")
}

pos_dividers <- list(
  geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 0.5),
  geom_vline(xintercept = 10.5, linetype = "dashed", color = "grey50"),
  geom_vline(xintercept = 19.5, linetype = "dashed", color = "grey50"),
  scale_x_continuous(breaks = 1:29, labels = all_pos_features)
)

# -----------------------------------------------------------------------------
# 3. Full-data PPMs
# -----------------------------------------------------------------------------
ppm_1 <- get_ppm_for_label(df_clean, "Presented",    all_pos_features, aa_alphabet)
ppm_0 <- get_ppm_for_label(df_clean, "Not Presented", all_pos_features, aa_alphabet)

# 3a. Differential logo (regeneration of the original, same math)
diff_ppm <- ppm_1 - ppm_0

p_diff <- ggseqlogo(diff_ppm, method = "custom") +
  pos_dividers +
  theme_logo_slide() +
  lab_slide(
    "Differential Sequence Logo: Presented vs. Not Presented",
    "Top (Positive) = Enriched in Presented | Bottom (Negative) = Depleted in Presented",
    "Difference in Probability")

ggsave("results/figures/logo/differential_logo_29mer.png",
       plot = p_diff, width = 20, height = 5, dpi = 300)
cat("Saved: results/figures/logo/differential_logo_29mer.png\n")

# 3b. Per-class bits logos (absolute 0-2 bit scale)
p_pos <- ggseqlogo(ppm_1, method = "bits") +
  pos_dividers +
  theme_logo_slide() +
  lab_slide(
    "Sequence Logo: Presented Peptides (label = 1)",
    "Bits of conservation within the presented class",
    "Bits")

ggsave("results/figures/logo/logo_presented_29mer.png",
       plot = p_pos, width = 20, height = 5, dpi = 300)
cat("Saved: results/figures/logo/logo_presented_29mer.png\n")

p_neg <- ggseqlogo(ppm_0, method = "bits") +
  pos_dividers +
  theme_logo_slide() +
  lab_slide(
    "Sequence Logo: Not Presented Peptides (label = 0)",
    "Bits of conservation within the not-presented class",
    "Bits")

ggsave("results/figures/logo/logo_not_presented_29mer.png",
       plot = p_neg, width = 20, height = 5, dpi = 300)
cat("Saved: results/figures/logo/logo_not_presented_29mer.png\n")

# -----------------------------------------------------------------------------
# 4. Per-fold differential logo (fold-stability QC)
#    ~1/6 the counts per panel -> noisier by construction; this checks that
#    no single fold drives the residue signal, not per-position precision.
# -----------------------------------------------------------------------------
fold_ids <- sort(unique(df_clean$fold))
diff_list <- lapply(fold_ids, function(f) {
  sub <- df_clean |> filter(fold == f)
  get_ppm_for_label(sub, "Presented", all_pos_features, aa_alphabet) -
    get_ppm_for_label(sub, "Not Presented", all_pos_features, aa_alphabet)
})
names(diff_list) <- paste0("Fold ", fold_ids)

p_folds <- ggseqlogo(diff_list, method = "custom") +
  facet_wrap(~seq_group, ncol = 2) +
  geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 0.4) +
  geom_vline(xintercept = 10.5, linetype = "dashed", color = "grey65") +
  geom_vline(xintercept = 19.5, linetype = "dashed", color = "grey65") +
  scale_x_continuous(breaks = seq(2, 28, by = 3),
                     labels = all_pos_features[seq(2, 28, by = 3)]) +
  theme_logo() +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "grey30"),
    axis.text.x = element_text(size = 8, face = "bold"),
    strip.text = element_text(face = "bold", size = 11)
  ) +
  labs(title = "Differential Logo by Fold (stability check)",
       subtitle = "Presented - Not Presented per homology-aware CV fold; panels should look alike",
       x = "Sequence Position (N-Flank → Peptide → C-Flank)",
       y = "Difference in Probability")

ggsave("results/figures/logo/differential_logo_by_fold.png",
       plot = p_folds, width = 14, height = 12, dpi = 300)
cat("Saved: results/figures/logo/differential_logo_by_fold.png\n")

cat("\n--- All logos done ---\n")
