#!/usr/bin/env Rscript

# =============================================================================
# 13_difflogo_significance.R — Differential logo with its OWN significance test
#
# Why this exists: ggseqlogo (used by 10_/11_/12_) is pure visualisation — it
# has no statistical test, and 12_sequence_logos.R feeds it a hand-computed
# ppm_1 - ppm_0 matrix, so the package never even sees n1/n2.
#
# This script instead uses DiffLogo (Schreiber et al., BMC Bioinformatics 2015)
# — the reference differential-logo tool — which computes, for every sequence
# position, its own p-value:
#   H0: the two position distributions are the same
#   null = numberOfPermutations label shuffles, stack height = Shannon (JS)
#   divergence, gamma fit of the null, p = 1 - CDF(observed)
# and marks p < 0.05 positions with '*' inside the plot.
#
# NOTE on interpretation: with n1/n2 ~ 20k+ the permutation null is sampling
# noise only, so essentially every position lands at p ~ 0. The informative
# numbers are therefore the effect sizes in the TSV (js_divergence,
# max_abs_delta), not the asterisks alone.
#
# Outputs (differential_logo_29mer_difflogo.png is rewritten — same deliverable,
# user-approved overwrite; nothing else is touched):
#   results/figures/logo/differential_logo_29mer_difflogo.png    letters = p1 - p0
#   results/figures/logo/differential_logo_29mer_difflogo_js.png letters = JS div.
#   results/figures/logo/difflogo_pvalues.tsv
# Letters are drawn with ggseqlogo (method = "custom"), so negative (depleted)
# stacks stay UPRIGHT below zero like differential_logo_29mer.png. DiffLogo is
# used only for its permutation p-values ('*'), never for letter polygons
# (DiffLogo's addLetter flips ht < 0 letters upside-down by design).
#
# Data + PPM math copied verbatim from 12_sequence_logos.R (sec. 1-2):
# per-column NA drop so terminus gaps don't skew probabilities; X residues are
# outside the 20-letter alphabet and are dropped by factor(v, levels = aa).
#
# Requires: DiffLogo (BiocManager::install("DiffLogo"))
# Run from repo root:
#   Rscript src/investigation/13_difflogo_significance.R [OUT] [IN_CSV]
#   OUT    output folder (default results/figures/logo)
#   IN_CSV input data (default data/processed/df_all_with_folds.csv)
# Example:
#   Rscript src/investigation/13_difflogo_significance.R \
#     results/figures/logo/flank10_pre_remove
# =============================================================================

if (!requireNamespace("DiffLogo", quietly = TRUE)) {
  stop("DiffLogo not installed. Run: install.packages('BiocManager'); ",
       "BiocManager::install('DiffLogo')")
}

library(tidyverse)
library(ggseqlogo)
library(DiffLogo)
source("src/pipeline/r/functions.R")
set_working_directory()

args <- commandArgs(trailingOnly = TRUE)
OUT <- if (length(args) >= 1) args[1] else "results/figures/logo"
IN_CSV <- if (length(args) >= 2) args[2] else "data/processed/df_all_with_folds.csv"

dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# -----------------------------------------------------------------------------
# 1. Data (verbatim from 12_sequence_logos.R)
# -----------------------------------------------------------------------------
df_folds <- read_csv(IN_CSV, show_col_types = FALSE)

# Flank length adapts to the input CSV (29-mer default, 19-mer for flank5):
# keep N10..N1 / C1..C10 that actually exist in the file.
n_flank_features  <- intersect(paste0("N", 10:1), names(df_folds))
peptide_features  <- paste0("P", 1:9)
c_flank_features  <- intersect(paste0("C", 1:10),  names(df_folds))
all_pos_features  <- c(n_flank_features, peptide_features, c_flank_features)

# Same order as DiffLogo's built-in 20-letter protein alphabet (ASN).
aa_alphabet <- c("A","C","D","E","F","G","H","I","K","L",
                 "M","N","P","Q","R","S","T","V","W","Y")

df_clean <- df_folds %>%
  filter(!is.na(label)) %>%
  mutate(label = factor(label, levels = c(0, 1),
                        labels = c("Not Presented", "Presented"))) %>%
  mutate(across(all_of(all_pos_features), as.character))

n1 <- sum(df_clean$label == "Presented")
n2 <- sum(df_clean$label == "Not Presented")
n_x <- sum(unlist(df_clean[all_pos_features], use.names = FALSE) == "X",
           na.rm = TRUE)

cat("Rows:", nrow(df_clean), "| n1 (Presented):", n1,
    "| n2 (Not Presented):", n2, "| X residues dropped:", n_x, "\n")

stopifnot(identical(aa_alphabet, ASN$chars))   # row order must match ASN

# -----------------------------------------------------------------------------
# 2. Per-class PPM (verbatim from 12_sequence_logos.R sec. 2) -> PWM for DiffLogo
# -----------------------------------------------------------------------------
get_ppm_for_label <- function(df, label_val, cols, alphabet) {
  sub_df <- df |> filter(label == label_val)
  ppm <- sapply(cols, function(col) {
    vals <- na.omit(sub_df[[col]])
    counts <- table(factor(vals, levels = alphabet))
    counts / sum(counts)
  })
  rownames(ppm) <- alphabet
  ppm
}

# data.frame (not matrix) input keeps DiffLogo's matrix-conversion chatter quiet
ppm_1 <- as.data.frame(get_ppm_for_label(df_clean, "Presented",
                                         all_pos_features, aa_alphabet))
ppm_0 <- as.data.frame(get_ppm_for_label(df_clean, "Not Presented",
                                         all_pos_features, aa_alphabet))

# -----------------------------------------------------------------------------
# 3. DiffLogo objects + the package's own per-position significance test
# -----------------------------------------------------------------------------
# Chemistry colour scheme = ggseqlogo's default "chemistry" palette (the one
# differential_logo_29mer.png uses), so AAs colour identically in both figures.
chem_col <- c(A = "#221E22", C = "#109648", D = "#D62839", E = "#D62839",
              F = "#221E22", G = "#109648", H = "#255C99", I = "#221E22",
              K = "#255C99", L = "#221E22", M = "#221E22", N = "#5E239D",
              P = "#221E22", Q = "#5E239D", R = "#255C99", S = "#109648",
              T = "#109648", V = "#221E22", W = "#221E22", Y = "#109648")
stopifnot(identical(names(chem_col), aa_alphabet))
alphabet_chem <- Alphabet(aa_alphabet, unname(chem_col), FALSE)

# Two display scales, ONE test:
#   obj_ppm  stackHeight = sumOfAbsProbabilityDifferences with the default
#            normalized-difference base distribution gives letter heights of
#            exactly p1 - p0 -> y-axis matches the old ggseqlogo figure
#            ("Difference in Probability").
#   obj_js   DiffLogo default (Shannon/JS divergence scaled stacks).
# align_pwms = FALSE: columns already fixed (N10..N1 | P1..P9 | C1..C10)
obj_ppm <- createDiffLogoObject(pwm1 = ppm_1, pwm2 = ppm_0,
                                stackHeight = sumOfAbsProbabilityDifferences,
                                alphabet = alphabet_chem,
                                try_reverse_complement = FALSE)
obj_ppm$ylab <- "Difference in Probability"
obj_js <- createDiffLogoObject(pwm1 = ppm_1, pwm2 = ppm_0,
                               alphabet = alphabet_chem,
                               try_reverse_complement = FALSE)

cat("Running DiffLogo permutation test",
    "(1000 permutations x", ncol(ppm_1), "positions) ...\n")
t0 <- Sys.time()
obj_ppm <- enrichDiffLogoObjectWithPvalues(obj_ppm, n1 = n1, n2 = n2,
                                           numberOfPermutations = 1000)
cat("  done in", round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1),
    "s\n")
# The p-value depends only on n1/n2/p1/p2 and shannonDivergence (enrich's own
# default argument, independent of the display stackHeight) — reuse it instead
# of paying the 97 s permutation run a second time.
obj_js$pvals <- obj_ppm$pvals

# -----------------------------------------------------------------------------
# 4. Figures — ggseqlogo draws the letters (upright below zero, same style as
#    12_sequence_logos.R); DiffLogo supplies only the '*' significance marks.
# -----------------------------------------------------------------------------
theme_logo_slide <- function() {
  theme_logo() +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5, color = "grey30"),
      axis.text.x = element_text(size = 11, face = "bold", angle = 0),
      panel.grid.major.y = element_line(color = "grey90")
    )
}

# region dividers between N-flank / peptide / C-flank (flank-length aware)
n_len <- length(n_flank_features)
pos_dividers <- list(
  geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 0.5),
  geom_vline(xintercept = n_len + 0.5, linetype = "dashed", color = "grey50"),
  geom_vline(xintercept = n_len + 9.5, linetype = "dashed", color = "grey50"),
  scale_x_continuous(breaks = seq_along(all_pos_features),
                     labels = all_pos_features)
)

# Letter matrices reproducing DiffLogo's two display scales exactly:
#   diff_mat: p1 - p0 (== sumOfAbsProbDiffs x normalized-difference)
#   js_mat:   (p1 - p0)/sum|p1 - p0| * JS height (== DiffLogo JS default)
m1 <- as.matrix(ppm_1)
m0 <- as.matrix(ppm_0)
rownames(m1) <- aa_alphabet
rownames(m0) <- aa_alphabet
diff_mat <- m1 - m0
js_h <- vapply(seq_along(all_pos_features), function(j)
  shannonDivergence(m1[, j], m0[, j])$height, numeric(1))
js_mat <- sapply(seq_along(all_pos_features), function(j) {
  d <- m1[, j] - m0[, j]
  s <- sum(abs(d))
  if (s == 0 || js_h[j] == 0) rep(0, length(d)) else d / s * js_h[j]
})
rownames(js_mat) <- aa_alphabet
colnames(js_mat) <- all_pos_features

render_ggdiff <- function(mat, y_label, out_png, scale_note) {
  pvals <- obj_ppm$pvals
  sig_x <- which(pvals < 0.05)
  # ggseqlogo stacks letters cumulatively per column, so the real top/bottom
  # is the sum of the positive / negative values, not max(mat) / min(mat)
  y_hi <- max(apply(mat, 2, function(v) sum(pmax(v, 0))))
  y_lo <- min(apply(mat, 2, function(v) sum(pmin(v, 0))))
  span <- y_hi - y_lo
  if (span == 0) span <- 1
  y_star <- y_lo - 0.15 * span
  star_df <- tibble(x = sig_x, y = y_star, label = "*")
  p <- ggseqlogo(mat, method = "custom") +
    pos_dividers +
    geom_text(data = star_df, aes(x = x, y = y, label = label),
              inherit.aes = FALSE, size = 6) +
    coord_cartesian(ylim = c(y_star - 0.08 * span, y_hi + 0.06 * span)) +
    theme_logo_slide() +
    labs(
      title = "Differential Sequence Logo: Presented vs. Not Presented",
      subtitle = paste0(
        "Top (Positive) = Enriched in Presented | Bottom (Negative) = Depleted in Presented\n",
        sprintf("* = DiffLogo permutation p < 0.05 (1000 shuffles; n1 = %s, n2 = %s) | %s",
                format(n1, big.mark = ","), format(n2, big.mark = ","),
                scale_note)),
      x = "Sequence Position (N-Flank \u2192 Peptide \u2192 C-Flank)",
      y = y_label)
  ggsave(out_png, plot = p, width = 20, height = 5.5, dpi = 300)
  cat("Saved:", out_png, "\n")
}

# output names derive from the actual position count (29-mer flank10, 19-mer flank5)
logo_mer <- paste0(length(all_pos_features), "mer")

render_ggdiff(diff_mat, "Difference in Probability",
              file.path(OUT, sprintf("differential_logo_%s_difflogo.png", logo_mer)),
              "letters = difference in probability")
render_ggdiff(js_mat, "JS divergence",
              file.path(OUT, sprintf("differential_logo_%s_difflogo_js.png", logo_mer)),
              "letters = JS divergence")

# -----------------------------------------------------------------------------
# 5. Per-position table — DiffLogo's own p-values + effect sizes
#    (independent of results/categorical/categorical_feature_tests.csv)
# -----------------------------------------------------------------------------
js_height <- vapply(seq_along(all_pos_features), function(j)
  shannonDivergence(ppm_1[, j], ppm_0[, j])$height, numeric(1))
max_delta <- vapply(seq_along(all_pos_features), function(j)
  max(abs(ppm_1[, j] - ppm_0[, j])), numeric(1))
# n_obs = residues that actually enter the PPM (20-letter alphabet; X dropped) —
# matches the convention of categorical_feature_tests.csv
n_obs <- vapply(all_pos_features, function(col)
  sum(df_clean[[col]] %in% aa_alphabet), numeric(1))

pvals <- obj_ppm$pvals
tib <- tibble(
  position       = all_pos_features,
  n_obs          = as.integer(n_obs),
  js_divergence  = round(js_height, 5),
  max_abs_delta  = round(max_delta, 5),
  p_raw          = signif(pvals, 4),
  p_BH           = signif(p.adjust(pvals, method = "BH"), 4),
  significant_05 = pvals < 0.05
)

out_tsv <- file.path(OUT, "difflogo_pvalues.tsv")
write_tsv(tib, out_tsv)
cat("Saved:", out_tsv, "\n")

cat("\nPer-position summary (sorted by effect size):\n")
print(as.data.frame(tib %>% arrange(desc(js_divergence)) %>%
                      select(position, n_obs, js_divergence, max_abs_delta,
                             p_raw, significant_05)),
      row.names = FALSE)
cat("\nSignificant positions (raw p < 0.05):",
    sum(tib$significant_05), "/", nrow(tib), "\n")
