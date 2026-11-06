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
# Both figures use ggseqlogo's "chemistry" colour scheme so they match
# differential_logo_29mer.png.
#
# Data + PPM math copied verbatim from 12_sequence_logos.R (sec. 1-2):
# per-column NA drop so terminus gaps don't skew probabilities; X residues are
# outside the 20-letter alphabet and are dropped by factor(v, levels = aa).
#
# Requires: DiffLogo (BiocManager::install("DiffLogo"))
# Run from repo root:  Rscript src/investigation/13_difflogo_significance.R
# =============================================================================

if (!requireNamespace("DiffLogo", quietly = TRUE)) {
  stop("DiffLogo not installed. Run: install.packages('BiocManager'); ",
       "BiocManager::install('DiffLogo')")
}

library(tidyverse)
library(DiffLogo)
source("src/pipeline/r/functions.R")
set_working_directory()

dir.create("results/figures/logo", showWarnings = FALSE, recursive = TRUE)

# -----------------------------------------------------------------------------
# 1. Data (verbatim from 12_sequence_logos.R)
# -----------------------------------------------------------------------------
df_folds <- read_csv("data/processed/df_all_with_folds.csv", show_col_types = FALSE)

n_flank_features  <- paste0("N", 10:1)
peptide_features  <- paste0("P", 1:9)
c_flank_features  <- paste0("C", 1:10)
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
# 4. Figures (DiffLogo draws base graphics; sparse = blank x tick labels so we
#    can put the real N/P/C feature names on the axis instead of 1..29)
# -----------------------------------------------------------------------------
render_logo <- function(obj, out_png, scale_note) {
  png(out_png, width = 20, height = 5.5, units = "in", res = 300)
  op <- par(mar = c(6, 5.5, 6.5, 1.5))

  # diffLogo() defaults ymin/ymax to c(posMax, negMax), which reverses the
  # y-axis (positive/enriched would point down) — pass them in normal order.
  diffLogo(obj, sparse = TRUE, ymin = obj$ylim.negMax, ymax = obj$ylim.posMax)
  axis(1, at = seq_along(all_pos_features), labels = all_pos_features,
       tick = FALSE, cex.axis = 0.8, font = 2, line = -1)
  # region names at the real column centres (N=1-10, P=11-19, C=20-29)
  axis(1, at = c(5.5, 15, 24.5),
       labels = c("N-Flank", "Peptide", "C-Flank"),
       tick = FALSE, cex.axis = 0.8, line = 3.5, col.ticks = NA,
       col.axis = "grey30")
  title(ylab = obj$ylab)
  title(main = "Differential Sequence Logo: Presented vs. Not Presented",
        font.main = 2)
  mtext("Top (Positive) = Enriched in Presented | Bottom (Negative) = Depleted in Presented",
        side = 3, line = 2.0, cex = 0.75, col = "grey30")
  mtext(sprintf(paste0("* = DiffLogo permutation p < 0.05 (1000 shuffles; ",
                       "n1 = %s, n2 = %s) | %s"),
                format(n1, big.mark = ","), format(n2, big.mark = ","),
                scale_note),
        side = 3, line = 1.0, cex = 0.7, col = "grey30")
  abline(v = c(10.5, 19.5), lty = 2, col = "grey50")
  # chemistry legend (ggseqlogo drew this below the axis; bottom row here is
  # taken by the region names, so it goes into the empty top-left corner)
  legend("topleft",
         legend = c("Acidic", "Basic", "Hydrophobic", "Neutral", "Polar"),
         fill = c("#D62839", "#255C99", "#221E22", "#5E239D", "#109648"),
         border = NA, bty = "n", cex = 0.7)

  par(op)
  dev.off()
  cat("Saved:", out_png, "\n")
}

render_logo(obj_ppm,
            "results/figures/logo/differential_logo_29mer_difflogo.png",
            "letters = difference in probability")
render_logo(obj_js,
            "results/figures/logo/differential_logo_29mer_difflogo_js.png",
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

out_tsv <- "results/figures/logo/difflogo_pvalues.tsv"
write_tsv(tib, out_tsv)
cat("Saved:", out_tsv, "\n")

cat("\nPer-position summary (sorted by effect size):\n")
print(as.data.frame(tib %>% arrange(desc(js_divergence)) %>%
                      select(position, n_obs, js_divergence, max_abs_delta,
                             p_raw, significant_05)),
      row.names = FALSE)
cat("\nSignificant positions (raw p < 0.05):",
    sum(tib$significant_05), "/", nrow(tib), "\n")
