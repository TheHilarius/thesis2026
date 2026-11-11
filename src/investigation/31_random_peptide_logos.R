#!/usr/bin/env Rscript

# =============================================================================
# 31_random_peptide_logos.R — logos for the confident extremes of 1M RANDOM
# context windows scored by the baseline XGB BLOSUM50 ensemble (lr=0.1,
# 30 nested-CV pickles, runs/encodings_sweep).
#
# Input (produced by 31_random_peptide_logo.py):
#   runs/random_peptide_logo/top10.csv     100k highest-P(presented) sequences
#   runs/random_peptide_logo/bottom10.csv  100k lowest-P sequences
#
# Outputs (runs/random_peptide_logo/, new files only):
#   logo_random_top10.png            bits logo, top decile
#   logo_random_bottom10.png         bits logo, bottom decile
#   differential_logo_random.png     custom PPM: top - bottom (ggseqlogo)
#   differential_logo_random_difflogo.png     DiffLogo, letters = p1 - p2
#   differential_logo_random_difflogo_js.png  DiffLogo, letters = JS divergence
#   difflogo_random_pvalues.tsv      DiffLogo permutation p-values
#
# ggseqlogo scaffolding from 12_sequence_logos.R; DiffLogo block copied from
# 13_difflogo_significance.R (both proven in this repo).
# Run from repo root:  Rscript src/investigation/31_random_peptide_logos.R
# =============================================================================

library(tidyverse)
library(ggseqlogo)
source("src/pipeline/r/functions.R")
set_working_directory()

args <- commandArgs(trailingOnly = TRUE)
OUT <- if (length(args) >= 1) args[1] else "runs/random_peptide_logo"
stopifnot(file.exists(file.path(OUT, "top10.csv")),
          file.exists(file.path(OUT, "bottom10.csv")))

aa_alphabet <- c("A","C","D","E","F","G","H","I","K","L",
                 "M","N","P","Q","R","S","T","V","W","Y")
n_flank_features  <- paste0("N", 10:1)
peptide_features  <- paste0("P", 1:9)
c_flank_features  <- paste0("C", 1:10)
all_pos_features  <- c(n_flank_features, peptide_features, c_flank_features)

# (ASN row-order guard lives in sec. 3, after library(DiffLogo))

# -----------------------------------------------------------------------------
# 1. Sequences -> PPM (per-column normalisation over the 20-letter alphabet;
#    identical math to get_ppm_for_label in 12_sequence_logos.R)
# -----------------------------------------------------------------------------
read_seqs <- function(fname) {
  df <- read_csv(file.path(OUT, fname),
                 col_types = cols(sequence = col_character(),
                                  prob = col_double()))
  stopifnot(all(nchar(df$sequence) == 29))
  df$sequence
}

seqs_top <- read_seqs("top10.csv")
seqs_bot <- read_seqs("bottom10.csv")
n1 <- length(seqs_top); n2 <- length(seqs_bot)
cat("n1 (top10):", n1, "| n2 (bottom10):", n2, "\n")

ppm_from_seqs <- function(seqs) {
  ppm <- sapply(all_pos_features, function(pos) {
    chars <- substr(seqs, match(pos, all_pos_features),
                    match(pos, all_pos_features))
    counts <- table(factor(chars, levels = aa_alphabet))
    counts / sum(counts)
  })
  rownames(ppm) <- aa_alphabet
  ppm
}

ppm_top <- ppm_from_seqs(seqs_top)
ppm_bot <- ppm_from_seqs(seqs_bot)

# -----------------------------------------------------------------------------
# 2. ggseqlogo figures (scaffolding verbatim from 12_)
# -----------------------------------------------------------------------------
theme_logo_slide <- theme_logo() +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "grey30"),
    axis.text.x = element_text(size = 11, face = "bold", angle = 0),
    panel.grid.major.y = element_line(color = "grey90")
  )

pos_dividers <- list(
  geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 0.5),
  geom_vline(xintercept = 10.5, linetype = "dashed", color = "grey50"),
  geom_vline(xintercept = 19.5, linetype = "dashed", color = "grey50"),
  scale_x_continuous(breaks = 1:29, labels = all_pos_features)
)

lab_rand <- function(title, subtitle, y) {
  labs(title = title, subtitle = subtitle, y = y,
       x = "Sequence Position (N-Flank \u2192 Peptide \u2192 C-Flank)")
}

# Fixed 0-2 bits (the classical logo scale), but fall back to auto-scaling
# when a stack exceeds 2 bits so nothing gets clipped.
max_stack_bits <- function(m) {
  max(vapply(seq_len(ncol(m)), function(j) {
    p <- m[, j]
    p <- p[p > 0]
    sum(p * log2(p / (1 / length(m[, j]))))   # IC, uniform background
  }, numeric(1)))
}
bits_ylim <- function(m, plot) {
  if (max_stack_bits(m) <= 2) plot + coord_cartesian(ylim = c(0, 2)) else plot
}

p_top <- ggseqlogo(ppm_top, method = "bits") +
  pos_dividers + theme_logo_slide +
  lab_rand("Random windows the model calls PRESENTED (top 10%)",
           "1M random 29-mers scored by the XGB BLOSUM50 ensemble (lr = 0.1)",
           "Bits")
p_top <- bits_ylim(ppm_top, p_top)
ggsave(file.path(OUT, "logo_random_top10.png"), plot = p_top,
       width = 20, height = 5, dpi = 300)
cat("Saved: logo_random_top10.png\n")

p_bot <- ggseqlogo(ppm_bot, method = "bits") +
  pos_dividers + theme_logo_slide +
  lab_rand("Random windows the model calls NOT PRESENTED (bottom 10%)",
           "1M random 29-mers scored by the XGB BLOSUM50 ensemble (lr = 0.1)",
           "Bits")
p_bot <- bits_ylim(ppm_bot, p_bot)
ggsave(file.path(OUT, "logo_random_bottom10.png"), plot = p_bot,
       width = 20, height = 5, dpi = 300)
cat("Saved: logo_random_bottom10.png\n")

diff_ppm <- ppm_top - ppm_bot
p_diff <- ggseqlogo(diff_ppm, method = "custom") +
  pos_dividers + theme_logo_slide +
  lab_rand("Differential Logo: top 10% vs bottom 10% (random windows)",
           "Top (Positive) = enriched in model-PRESENTED random windows",
           "Difference in Probability")
ggsave(file.path(OUT, "differential_logo_random.png"), plot = p_diff,
       width = 20, height = 5, dpi = 300)
cat("Saved: differential_logo_random.png\n")

# -----------------------------------------------------------------------------
# 3. DiffLogo + permutation test (block verbatim from 13_difflogo_significance.R)
# -----------------------------------------------------------------------------
if (!requireNamespace("DiffLogo", quietly = TRUE)) {
  stop("DiffLogo not installed. Run: install.packages('BiocManager'); ",
       "BiocManager::install('DiffLogo')")
}
library(DiffLogo)
stopifnot(identical(aa_alphabet, ASN$chars))   # row order must match ASN

# data.frame (not matrix) input keeps DiffLogo's matrix-conversion chatter quiet
ppm_top_df <- as.data.frame(ppm_top)
ppm_bot_df <- as.data.frame(ppm_bot)

# Chemistry colour scheme = ggseqlogo's default "chemistry" palette so AAs
# colour identically in all three figures.
chem_col <- c(A = "#221E22", C = "#109648", D = "#D62839", E = "#D62839",
              F = "#221E22", G = "#109648", H = "#255C99", I = "#221E22",
              K = "#255C99", L = "#221E22", M = "#221E22", N = "#5E239D",
              P = "#221E22", Q = "#5E239D", R = "#255C99", S = "#109648",
              T = "#109648", V = "#221E22", W = "#221E22", Y = "#109648")
stopifnot(identical(names(chem_col), aa_alphabet))
alphabet_chem <- Alphabet(aa_alphabet, unname(chem_col), FALSE)

obj_ppm <- createDiffLogoObject(pwm1 = ppm_top_df, pwm2 = ppm_bot_df,
                                stackHeight = sumOfAbsProbabilityDifferences,
                                alphabet = alphabet_chem,
                                try_reverse_complement = FALSE)
obj_ppm$ylab <- "Difference in Probability"
obj_js <- createDiffLogoObject(pwm1 = ppm_top_df, pwm2 = ppm_bot_df,
                               alphabet = alphabet_chem,
                               try_reverse_complement = FALSE)

cat("Running DiffLogo permutation test",
    "(1000 permutations x", ncol(ppm_top_df), "positions) ...\n")
t0 <- Sys.time()
obj_ppm <- enrichDiffLogoObjectWithPvalues(obj_ppm, n1 = n1, n2 = n2,
                                           numberOfPermutations = 1000)
cat("  done in", round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1),
    "s\n")
obj_js$pvals <- obj_ppm$pvals   # p-value depends only on n1/n2/p1/p2, not scale

render_logo <- function(obj, out_png, scale_note, main_title) {
  png(out_png, width = 20, height = 5.5, units = "in", res = 300)
  op <- par(mar = c(6, 5.5, 6.5, 1.5))
  diffLogo(obj, sparse = TRUE, ymin = obj$ylim.negMax, ymax = obj$ylim.posMax)
  axis(1, at = seq_along(all_pos_features), labels = all_pos_features,
       tick = FALSE, cex.axis = 0.8, font = 2, line = -1)
  axis(1, at = c(5.5, 15, 24.5),
       labels = c("N-Flank", "Peptide", "C-Flank"),
       tick = FALSE, cex.axis = 0.8, line = 3.5, col.ticks = NA,
       col.axis = "grey30")
  title(ylab = obj$ylab)
  title(main = main_title, font.main = 2)
  mtext("Top (Positive) = enriched in model-PRESENTED random windows",
        side = 3, line = 2.0, cex = 0.75, col = "grey30")
  mtext(sprintf(paste0("* = DiffLogo permutation p < 0.05 (1000 shuffles; ",
                       "n1 = %s, n2 = %s) | %s"),
                format(n1, big.mark = ","), format(n2, big.mark = ","),
                scale_note),
        side = 3, line = 1.0, cex = 0.7, col = "grey30")
  abline(v = c(10.5, 19.5), lty = 2, col = "grey50")
  legend("topleft",
         legend = c("Acidic", "Basic", "Hydrophobic", "Neutral", "Polar"),
         fill = c("#D62839", "#255C99", "#221E22", "#5E239D", "#109648"),
         border = NA, bty = "n", cex = 0.7)
  par(op)
  dev.off()
  cat("Saved:", out_png, "\n")
}

render_logo(obj_ppm,
            file.path(OUT, "differential_logo_random_difflogo.png"),
            "letters = difference in probability",
            "DiffLogo: top 10% vs bottom 10% (random windows)")
render_logo(obj_js,
            file.path(OUT, "differential_logo_random_difflogo_js.png"),
            "letters = JS divergence",
            "DiffLogo (JS): top 10% vs bottom 10% (random windows)")

# -----------------------------------------------------------------------------
# 4. Per-position table (pattern from 13_)
# -----------------------------------------------------------------------------
js_height <- vapply(seq_along(all_pos_features), function(j)
  shannonDivergence(ppm_top[, j], ppm_bot[, j])$height, numeric(1))
max_delta <- vapply(seq_along(all_pos_features), function(j)
  max(abs(ppm_top[, j] - ppm_bot[, j])), numeric(1))

pvals <- obj_ppm$pvals
tib <- tibble(
  position       = all_pos_features,
  js_divergence  = round(js_height, 5),
  max_abs_delta  = round(max_delta, 5),
  p_raw          = signif(pvals, 4),
  p_BH           = signif(p.adjust(pvals, method = "BH"), 4),
  significant_05 = pvals < 0.05
)

out_tsv <- file.path(OUT, "difflogo_random_pvalues.tsv")
write_tsv(tib, out_tsv)
cat("Saved:", out_tsv, "\n")

cat("\nPer-position summary (sorted by effect size):\n")
print(as.data.frame(tib %>% arrange(desc(js_divergence))),
      row.names = FALSE)
cat("\nSignificant positions (raw p < 0.05):",
    sum(tib$significant_05), "/", nrow(tib), "\n")
cat("\n--- All random-window logos done ---\n")
