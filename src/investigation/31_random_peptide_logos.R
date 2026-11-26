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
#   differential_logo_random_difflogo.png     ggseqlogo, letters = p_top - p_bot
#   differential_logo_random_difflogo_js.png  ggseqlogo, letters = JS divergence
#   (DiffLogo supplies only the '*' p-values; its renderer flips negative
#   letters upside-down, so letters use ggseqlogo to stay upright)
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

# Positions come from the PPM CSV the Python step wrote (19-mer rmdup or
# 29-mer baseline); sequences in top10/bottom10.csv must match it.
ppm_ref <- read_csv(file.path(OUT, "ppm_top10.csv"),
                    col_types = cols(.default = col_double(), ...1 = col_character()))
all_pos_features <- tail(names(ppm_ref), -1)

# (ASN row-order guard lives in sec. 3, after library(DiffLogo))

# -----------------------------------------------------------------------------
# 1. Sequences -> PPM (per-column normalisation over the 20-letter alphabet;
#    identical math to get_ppm_for_label in 12_sequence_logos.R)
# -----------------------------------------------------------------------------
read_seqs <- function(fname) {
  df <- read_csv(file.path(OUT, fname),
                 col_types = cols(sequence = col_character(),
                                  prob = col_double()))
  stopifnot(all(nchar(df$sequence) == length(all_pos_features)))
  df$sequence
}

seqs_top <- read_seqs("top10.csv")
seqs_bot <- read_seqs("bottom10.csv")
n1 <- length(seqs_top); n2 <- length(seqs_bot)
n_chars <- unique(nchar(seqs_top))
stopifnot(length(n_chars) == 1, n_chars == length(all_pos_features))
cat("n1 (top10):", n1, "| n2 (bottom10):", n2, "| window:", n_chars, "\n")

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
n_flank <- sum(startsWith(all_pos_features, "N"))
pep_len <- sum(startsWith(all_pos_features, "P"))
n_pos   <- length(all_pos_features)          # 19 (rmdup) or 29 (baseline)
theme_logo_slide <- theme_logo() +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "grey30"),
    axis.text.x = element_text(size = 11, face = "bold", angle = 0),
    panel.grid.major.y = element_line(color = "grey90")
  )

pos_dividers <- list(
  geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 0.5),
  geom_vline(xintercept = n_flank + 0.5, linetype = "dashed", color = "grey50"),
  geom_vline(xintercept = n_flank + pep_len + 0.5, linetype = "dashed", color = "grey50"),
  scale_x_continuous(breaks = 1:n_pos, labels = all_pos_features)
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
           "1M random windows scored by the XGB BLOSUM50 ensemble (lr = 0.1)",
           "Bits")
p_top <- bits_ylim(ppm_top, p_top)
ggsave(file.path(OUT, "logo_random_top10.png"), plot = p_top,
       width = 20, height = 5, dpi = 300)
cat("Saved: logo_random_top10.png\n")

p_bot <- ggseqlogo(ppm_bot, method = "bits") +
  pos_dividers + theme_logo_slide +
  lab_rand("Random windows the model calls NOT PRESENTED (bottom 10%)",
           "1M random windows scored by the XGB BLOSUM50 ensemble (lr = 0.1)",
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
# 3. DiffLogo permutation test only (letters drawn with ggseqlogo below, so
#    negative stacks stay UPRIGHT; DiffLogo's addLetter flips ht < 0 by design)
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

# Letter matrices reproducing DiffLogo's two display scales exactly (upright):
#   diff: p_top - p_bot;  js: (p_top - p_bot)/sum|..| * JS height
# (merge of 46419d5 + 5ebb687: dynamic window layout + upright ggseqlogo letters)
m_top <- as.matrix(ppm_top)
m_bot <- as.matrix(ppm_bot)
rownames(m_top) <- aa_alphabet
rownames(m_bot) <- aa_alphabet
diff_rand <- m_top - m_bot
js_h_rand <- vapply(seq_along(all_pos_features), function(j)
  shannonDivergence(m_top[, j], m_bot[, j])$height, numeric(1))
js_rand <- sapply(seq_along(all_pos_features), function(j) {
  d <- m_top[, j] - m_bot[, j]
  s <- sum(abs(d))
  if (s == 0 || js_h_rand[j] == 0) rep(0, length(d)) else d / s * js_h_rand[j]
})
rownames(js_rand) <- aa_alphabet
colnames(js_rand) <- all_pos_features

# Region labels under the axis — positions dynamic (from 46419d5)
n_cf <- n_pos - n_flank - pep_len
region_labs <- data.frame(
  x     = c(n_flank / 2, n_flank + pep_len / 2, n_flank + pep_len + n_cf / 2),
  label = c("N-Flank", "Peptide", "C-Flank")
)

render_ggdiff <- function(mat, y_label, out_png, scale_note, main_title) {
  pvals <- obj_ppm$pvals
  sig_x <- which(pvals < 0.05)
  y_lo <- min(mat)
  y_hi <- max(mat)
  span <- y_hi - y_lo
  if (span == 0) span <- 1
  y_star   <- y_lo - 0.15 * span
  y_region <- y_star - 0.10 * span   # region labels sit below the stars
  star_df <- tibble(x = sig_x, y = y_star, label = "*")
  p <- ggseqlogo(mat, method = "custom") +
    pos_dividers + theme_logo_slide +
    geom_text(data = star_df, aes(x = x, y = y, label = label),
              inherit.aes = FALSE, size = 6) +
    geom_text(data = region_labs, aes(x = x, y = y_region, label = label),
              inherit.aes = FALSE, size = 4, color = "grey30") +
    coord_cartesian(ylim = c(y_region - 0.05 * span, y_hi + 0.06 * span),
                    clip = "off") +
    lab_rand(main_title,
             paste0("Top (Positive) = enriched in model-PRESENTED random windows\n",
                    sprintf(paste0("* = DiffLogo permutation p < 0.05 (1000 shuffles; ",
                                   "n1 = %s, n2 = %s) | %s | Colours: ggseqlogo ",
                                   "chemistry scheme (Acidic=#D62839, Basic=#255C99, ",
                                   "Hydrophobic=#221E22, Polar=#5E239D, Neutral=#109648)"),
                            format(n1, big.mark = ","), format(n2, big.mark = ","),
                            scale_note)),
             y_label)
  ggsave(out_png, plot = p, width = 20, height = 5.5, dpi = 300)
  cat("Saved:", out_png, "\n")
}

render_ggdiff(diff_rand, "Difference in Probability",
              file.path(OUT, "differential_logo_random_difflogo.png"),
              "letters = difference in probability",
              "DiffLogo: top 10% vs bottom 10% (random windows)")
render_ggdiff(js_rand, "JS divergence",
              file.path(OUT, "differential_logo_random_difflogo_js.png"),
              "letters = JS divergence",
              "DiffLogo (JS): top 10% vs bottom 10% (random windows)")

# ── 3b. Split-scale differential pair — readability fix ────────────────────
# Peptide-core and flank columns differ ~10-70x in effect size, so a single
# shared y-range squashes the flank signal (C1/N1/N2) into invisible slivers.
# Two panels, independent y-scales, one PNG via grid viewports (base R).
library(grid)

pep_cols   <- which(startsWith(all_pos_features, "P"))
flank_cols <- which(!startsWith(all_pos_features, "P"))

ggdiff_panel <- function(mat, cols, panel_title, y_label, subtitle = NULL) {
  sub       <- mat[, cols, drop = FALSE]
  sub_names <- colnames(sub)
  p_sub     <- obj_ppm$pvals[cols]
  sig_x     <- which(p_sub < 0.05)
  y_lo <- min(sub); y_hi <- max(sub)
  span <- if (y_hi - y_lo == 0) 1 else y_hi - y_lo
  y_star   <- y_lo - 0.15 * span
  y_region <- y_star - 0.10 * span
  star_df <- tibble(x = sig_x, y = y_star, label = "*")
  n_in  <- length(cols)
  is_pep <- all(startsWith(sub_names, "P"))
  reg_df <- if (is_pep) {
    tibble(x = n_in / 2, y = y_region, label = "Peptide")
  } else {
    tibble(x = c(n_flank / 2, n_flank + n_cf / 2), y = y_region,
           label = c("N-Flank", "C-Flank"))
  }
  vline <- if (is_pep) geom_blank() else
    geom_vline(xintercept = n_flank + 0.5, linetype = "dashed",
               color = "grey50")
  ggseqlogo(sub, method = "custom") +
    geom_hline(yintercept = 0, linetype = "solid", color = "black",
               linewidth = 0.5) +
    vline +
    scale_x_continuous(breaks = seq_len(n_in), labels = sub_names) +
    theme_logo_slide +
    geom_text(data = star_df, aes(x = x, y = y, label = label),
              inherit.aes = FALSE, size = 6) +
    geom_text(data = reg_df, aes(x = x, y = y, label = label),
              inherit.aes = FALSE, size = 4, color = "grey30") +
    coord_cartesian(ylim = c(y_region - 0.05 * span, y_hi + 0.06 * span),
                    clip = "off") +
    labs(title = panel_title, subtitle = subtitle, y = y_label,
         x = if (is_pep) "Peptide positions" else "Flank positions")
}

split_pair <- function(mat, y_label, out_png, scale_note, main_title) {
  note <- paste0("Top (Positive) = enriched in model-PRESENTED windows\n",
                 sprintf(paste0("* = DiffLogo permutation p < 0.05 (1000 shuffles; ",
                                "n1 = %s, n2 = %s) | %s"),
                         format(n1, big.mark = ","),
                         format(n2, big.mark = ","), scale_note))
  p_pep <- ggdiff_panel(mat, pep_cols,
                        paste0(main_title, " \u2014 peptide core"), y_label, note)
  p_flank <- ggdiff_panel(mat, flank_cols,
                          paste0(main_title, " \u2014 flanks (independent scale)"),
                          y_label)
  png(out_png, width = 20, height = 5.5, units = "in", res = 300)
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(1, 2, widths = c(9, 10))))
  print(p_pep,   vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
  print(p_flank, vp = viewport(layout.pos.row = 1, layout.pos.col = 2))
  popViewport()
  dev.off()
  cat("Saved:", out_png, "\n")
}

split_pair(diff_rand, "Difference in Probability",
           file.path(OUT, "differential_logo_random_difflogo_split.png"),
           "letters = difference in probability",
           "DiffLogo: top 10% vs bottom 10%")
split_pair(js_rand, "JS divergence",
           file.path(OUT, "differential_logo_random_difflogo_js_split.png"),
           "letters = JS divergence",
           "DiffLogo (JS): top 10% vs bottom 10%")

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
