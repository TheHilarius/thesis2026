#!/usr/bin/env Rscript
# =============================================================================
# af_version_compare.R
# Compare AlphaFold v2.0 (DB) vs v2.3.18 (local colabfold) PDB structures
# for 21 proteins to assess version-dependent differences in predicted models.
# =============================================================================

library(tidyverse)

# Working directory setup (matches project convention)
setwd("/home/hilarius/projects/thesis2026")

# =============================================================================
# CONSTANTS
# =============================================================================

# 21 proteins and their batch order in batch.fasta
PROTEINS <- c(
  "Q8NCT1", "O60573", "Q8TEA8", "Q53G44", "Q12841",
  "P28908", "Q9NRC1", "Q9Y6S9", "Q9Y6H5", "P35503",
  "Q14703", "Q7Z4H2"
)

DB_DIR    <- "data/processed/structures/alphafold"
LOCAL_DIR <- "data/processed/af_version_comparison/local_output"
OUT_DIR   <- "data/processed/af_version_comparison"
RESULT_PNG <- "results/af_version_comparison_plddt.png"

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(dirname(RESULT_PNG), showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# PDB PARSING
# =============================================================================

#' Parse CA atoms from a PDB file
#' Returns a tibble with columns: resno, x, y, z, bfactor, chain
parse_ca_atoms <- function(pdb_path) {
  if (!file.exists(pdb_path)) {
    warning("File not found: ", pdb_path)
    return(NULL)
  }

  lines <- readLines(pdb_path)

  # Extract ATOM lines with CA atoms
  atom_lines <- lines[grepl("^ATOM", lines)]

  if (length(atom_lines) == 0) {
    warning("No ATOM records in: ", pdb_path)
    return(NULL)
  }

  # Filter for CA (C-alpha) atoms only
  ca_lines <- atom_lines[grepl(" CA ", atom_lines)]

  if (length(ca_lines) == 0) {
    warning("No CA atoms in: ", pdb_path)
    return(NULL)
  }

  # Parse fixed-width PDB ATOM format
  # Columns (1-indexed):
  #   1-6:   record type ("ATOM  ")
  #   7-11:  serial
  #  13-16:  atom name
  #  17:    alternate location
  #  20-22: residue name (not needed for CA)
  #  22:    not used
  #  23:    chain ID  (actually col 22, 1-indexed)
  # Actually PDB format: chain is column 22 (1-indexed), resno is 23-26
  # Let me use a more robust regex approach

  # Use fixed-width parsing matching the PDB spec:
  # Chain ID: character 22 (1-indexed)
  # Residue number: characters 23-26
  # X: characters 31-38
  # Y: characters 39-46
  # Z: characters 47-54
  # B-factor: characters 61-66

  parse_field <- function(lines, start, end) {
    substr(lines, start, end)
  }

  chains  <- trimws(parse_field(ca_lines, 22, 22))
  resnos  <- as.integer(trimws(parse_field(ca_lines, 23, 26)))
  xs      <- as.numeric(parse_field(ca_lines, 31, 38))
  ys      <- as.numeric(parse_field(ca_lines, 39, 46))
  zs      <- as.numeric(parse_field(ca_lines, 47, 54))
  bfactors <- as.numeric(parse_field(ca_lines, 61, 66))

  tibble(
    resno    = resnos,
    x        = xs,
    y        = ys,
    z        = zs,
    bfactor  = bfactors,
    chain    = chains
  )
}

# =============================================================================
# LOCAL PDB FILE DISCOVERY
# =============================================================================

#' Find the local PDB file for a given UniProt ID
#' Tries multiple naming conventions:
#'   1. batch_XXXXX_0.pdb (where XXXXX is zero-padded batch index)
#'   2. batch_XXXXX_unrelaxed_rank_1.pdb
#'   3. Colabfold naming: tr_{ID}_...pdb
#'   4. Simple {ID}.pdb
find_local_pdb <- function(uniprot_id, batch_index) {
  local_dir <- file.path(LOCAL_DIR)

  # Pattern 1: batch_XXXXX_0.pdb (5-digit zero-padded index)
  batch_name <- sprintf("batch_%05d", batch_index)
  candidate <- file.path(local_dir, paste0(batch_name, "_0.pdb"))
  if (file.exists(candidate)) return(candidate)

  # Pattern 2: batch_XXXXX with unrelaxed suffix
  candidate <- file.path(local_dir, paste0(batch_name, "_unrelaxed_rank_1.pdb"))
  if (file.exists(candidate)) return(candidate)

  # Pattern 3: batch_XXXXX with .pdb extension directly
  candidate <- file.path(local_dir, paste0(batch_name, ".pdb"))
  if (file.exists(candidate)) return(candidate)

  # Pattern 4: Search for files matching UniProt ID (colabfold naming)
  all_pdbs <- list.files(local_dir, pattern = "\\.pdb$", full.names = TRUE)
  # Try to find one containing the UniProt ID
  matching <- all_pdbs[grepl(uniprot_id, basename(all_pdbs), ignore.case = TRUE)]
  if (length(matching) > 0) return(matching[1])

  # Pattern 5: Look for any *_rank_1.pdb or *_relaxed_rank_1.pdb files
  rank1 <- all_pdbs[grepl("rank_1\\.pdb$", basename(all_pdbs))]
  if (length(rank1) >= batch_index) return(rank1[batch_index])

  return(NULL)
}

# =============================================================================
# KABSCH ALGORITHM FOR RMSD
# =============================================================================

#' Kabsch algorithm: optimal superposition of two coordinate sets
#'
#' @param coords1 Nx3 matrix of coordinates (mobile)
#' @param coords2 Nx3 matrix of coordinates (reference)
#' @return List with: rmsd, rotation_matrix, translation
kabsch <- function(coords1, coords2) {
  stopifnot(nrow(coords1) == nrow(coords2), ncol(coords1) == 3, ncol(coords2) == 3)
  n <- nrow(coords1)

  # Center both coordinate sets
  center1 <- colMeans(coords1)
  center2 <- colMeans(coords2)
  c1 <- coords1 - matrix(center1, nrow = n, ncol = 3, byrow = TRUE)
  c2 <- coords2 - matrix(center2, nrow = n, ncol = 3, byrow = TRUE)

  # Covariance matrix
  H <- t(c1) %*% c2

  # SVD
  svd_result <- svd(H)

  # Correct for reflection
  d <- det(svd_result$v %*% t(svd_result$u))
  sign_mat <- diag(c(1, 1, sign(d)))

  # Optimal rotation
  R <- svd_result$v %*% sign_mat %*% t(svd_result$u)

  # Apply rotation to centered coords1
  c1_rotated <- c1 %*% t(R)

  # Compute RMSD
  diff <- c1_rotated - c2
  rmsd <- sqrt(sum(diff^2) / n)

  return(list(
    rmsd     = rmsd,
    rotation = R,
    center1  = center1,
    center2  = center2
  ))
}

# Also provide a bio3d-based RMSD if available
kabsch_bio3d <- function(coords1, coords2) {
  if (!requireNamespace("bio3d", quietly = TRUE)) return(NULL)
  # Use bio3d's kabsch function for validation
  result <- bio3d::kabsch(coords1, coords2)
  return(result)
}

# =============================================================================
# METRICS COMPUTATION
# =============================================================================

#' Compute all comparison metrics between two parsed CA atom tables
compute_metrics <- function(db_atoms, local_atoms) {
  # Match residues by residue number (ignoring chain differences)
  # Use residue number as the primary key
  db_df <- db_atoms %>%
    distinct(resno, .keep_all = TRUE) %>%
    arrange(resno)

  local_df <- local_atoms %>%
    distinct(resno, .keep_all = TRUE) %>%
    arrange(resno)

  # Inner join on residue number
  shared <- inner_join(
    db_df %>% select(resno, x_db = x, y_db = y, z_db = z, plddt_db = bfactor),
    local_df %>% select(resno, x_local = x, y_local = y, z_local = z, plddt_local = bfactor),
    by = "resno"
  )

  n_shared <- nrow(shared)
  if (n_shared < 3) {
    warning("Too few shared residues (", n_shared, ") for comparison")
    return(NULL)
  }

  # Pearson correlation of pLDDT
  pearson_r <- cor(shared$plddt_db, shared$plddt_local, method = "pearson")

  # Spearman correlation of pLDDT
  spearman_rho <- cor(shared$plddt_db, shared$plddt_local, method = "spearman")

  # Per-residue pLDDT difference
  plddt_diff <- abs(shared$plddt_db - shared$plddt_local)
  mean_plddt_diff <- mean(plddt_diff)
  n_diff_gt10 <- sum(plddt_diff > 10)

  # C-alpha RMSD after optimal superposition (Kabsch)
  coords_db <- as.matrix(shared %>% select(x_db, y_db, z_db))
  coords_local <- as.matrix(shared %>% select(x_local, y_local, z_local))

  kabsch_result <- kabsch(coords_db, coords_local)
  rmsd <- kabsch_result$rmsd

  tibble(
    n_residues_shared     = n_shared,
    pearson_r             = pearson_r,
    spearman_rho          = spearman_rho,
    rmsd_a                = rmsd,
    mean_plddt_diff       = mean_plddt_diff,
    n_residues_diff_gt10  = n_diff_gt10,
    seq_length_db         = nrow(db_df),
    seq_length_local      = nrow(local_df)
  )
}

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

cat("============================================================\n")
cat("AlphaFold Version Comparison: v2.0 (DB) vs v2.3.18 (Local)\n")
cat("============================================================\n\n")

results <- list()

for (i in seq_along(PROTEINS)) {
  uniprot_id <- PROTEINS[i]
  cat(sprintf("[%2d/21] %s ... ", i, uniprot_id))

  # Parse DB PDB
  db_path <- file.path(DB_DIR, paste0(uniprot_id, ".pdb"))
  db_atoms <- parse_ca_atoms(db_path)

  # Find and parse local PDB
  local_path <- find_local_pdb(uniprot_id, i)
  if (is.null(local_path)) {
    cat("SKIPPED (local PDB not found)\n")
    next
  }
  local_atoms <- parse_ca_atoms(local_path)

  if (is.null(db_atoms) || is.null(local_atoms)) {
    cat("SKIPPED (parse error)\n")
    next
  }

  # Compute metrics
  metrics <- compute_metrics(db_atoms, local_atoms)

  if (is.null(metrics)) {
    cat("SKIPPED (insufficient shared residues)\n")
    next
  }

  metrics$uniprot_id <- uniprot_id
  metrics$batch_index <- i
  metrics$db_path <- db_path
  metrics$local_path <- local_path

  results[[uniprot_id]] <- metrics

  cat(sprintf("OK (r=%.4f, RMSD=%.3f Å, %d shared residues)\n",
              metrics$pearson_r, metrics$rmsd_a, metrics$n_residues_shared))
}

# =============================================================================
# RESULTS TABLE
# =============================================================================

if (length(results) == 0) {
  cat("\nNo comparisons completed. Check that local PDB files exist.\n")
  cat("Expected location: ", LOCAL_DIR, "\n")
  cat("Naming: batch_XXXXX_0.pdb or UniProt-ID-based names\n")
  quit(status = 1)
}

results_df <- bind_rows(results) %>%
  select(uniprot_id, seq_length = seq_length_db, pearson_r, spearman_rho,
         rmsd_a, mean_plddt_diff, n_residues_diff_gt10, n_residues_shared) %>%
  arrange(desc(seq_length))

# Write CSV
csv_path <- file.path(OUT_DIR, "results.csv")
write_csv(results_df, csv_path)
cat("\nResults written to:", csv_path, "\n")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

cat("\n============================================================\n")
cat("Summary Statistics\n")
cat("============================================================\n\n")

cat(sprintf("Proteins compared:    %d / 21\n", nrow(results_df)))
cat(sprintf("Seq length range:     %d - %d residues\n",
            min(results_df$seq_length), max(results_df$seq_length)))
cat(sprintf("Median seq length:    %d residues\n", median(results_df$seq_length)))
cat("\n")

cat("Pearson r (pLDDT):\n")
cat(sprintf("  Mean:   %.4f\n", mean(results_df$pearson_r)))
cat(sprintf("  Median: %.4f\n", median(results_df$pearson_r)))
cat(sprintf("  Range:  %.4f - %.4f\n", min(results_df$pearson_r), max(results_df$pearson_r)))
cat(sprintf("  Min ID: %s (r=%.4f)\n",
            results_df$uniprot_id[which.min(results_df$pearson_r)],
            min(results_df$pearson_r)))
cat("\n")

cat("Spearman rho (pLDDT):\n")
cat(sprintf("  Mean:   %.4f\n", mean(results_df$spearman_rho)))
cat(sprintf("  Median: %.4f\n", median(results_df$spearman_rho)))
cat(sprintf("  Range:  %.4f - %.4f\n", min(results_df$spearman_rho), max(results_df$spearman_rho)))
cat("\n")

cat("RMSD (Å):\n")
cat(sprintf("  Mean:   %.4f\n", mean(results_df$rmsd_a)))
cat(sprintf("  Median: %.4f\n", median(results_df$rmsd_a)))
cat(sprintf("  Range:  %.4f - %.4f\n", min(results_df$rmsd_a), max(results_df$rmsd_a)))
cat(sprintf("  Max ID: %s (RMSD=%.4f)\n",
            results_df$uniprot_id[which.max(results_df$rmsd_a)],
            max(results_df$rmsd_a)))
cat("\n")

cat("Per-residue pLDDT difference:\n")
cat(sprintf("  Mean abs diff:  %.2f points\n", mean(results_df$mean_plddt_diff)))
cat(sprintf("  Median abs diff: %.2f points\n", median(results_df$mean_plddt_diff)))
cat(sprintf("  Residues >10pt diff: %d total (mean %.1f per protein)\n",
            sum(results_df$n_residues_diff_gt10),
            mean(results_df$n_residues_diff_gt10)))
cat("\n")

# Correlation with sequence length
len_pearson <- cor(results_df$seq_length, results_df$pearson_r, use = "complete.obs")
len_rmsd <- cor(results_df$seq_length, results_df$rmsd_a, use = "complete.obs")
cat("Correlation with sequence length:\n")
cat(sprintf("  Length vs Pearson r:  r = %.3f\n", len_pearson))
cat(sprintf("  Length vs RMSD:       r = %.3f\n", len_rmsd))

# =============================================================================
# DECISION CRITERIA
# =============================================================================

cat("\n============================================================\n")
cat("Decision Criteria\n")
cat("============================================================\n\n")

median_pearson <- median(results_df$pearson_r)
median_rmsd <- median(results_df$rmsd_a)

cat(sprintf("Median Pearson r:  %.4f\n", median_pearson))
cat(sprintf("Median RMSD:       %.4f Å\n", median_rmsd))
cat("\n")

if (median_pearson > 0.98 && median_rmsd < 1.0) {
  verdict <- "Version difference negligible — mixing is safe"
} else if (median_pearson > 0.95 && median_rmsd < 2.0) {
  verdict <- "Version difference acceptable — note in methods"
} else {
  verdict <- "Version difference significant — standardize to one version"
}

cat("VERDICT:\n")
cat("  >> ", verdict, "\n")

# Print per-protein outlier flagging
cat("\nProteins with notable differences:\n")
outliers <- results_df %>%
  filter(pearson_r < 0.95 | rmsd_a > 2.0 | n_residues_diff_gt10 > 5)

if (nrow(outliers) > 0) {
  for (j in seq_len(nrow(outliers))) {
    cat(sprintf("  - %s: r=%.4f, RMSD=%.3f, >10pt diff residues=%d (len=%d)\n",
                outliers$uniprot_id[j], outliers$pearson_r[j], outliers$rmsd_a[j],
                outliers$n_residues_diff_gt10[j], outliers$seq_length[j]))
  }
} else {
  cat("  None\n")
}

# =============================================================================
# PLOTS
# =============================================================================

cat("\nGenerating plots...\n")

# Prepare data for plotting
# Gather all shared residues for scatter plot
all_residues <- list()
for (uid in results_df$uniprot_id) {
  i <- which(PROTEINS == uid)
  db_path <- file.path(DB_DIR, paste0(uid, ".pdb"))
  local_path <- find_local_pdb(uid, i)

  db_atoms <- parse_ca_atoms(db_path)
  local_atoms <- parse_ca_atoms(local_path)

  if (is.null(db_atoms) || is.null(local_atoms)) next

  db_df <- db_atoms %>% distinct(resno, .keep_all = TRUE)
  local_df <- local_atoms %>% distinct(resno, .keep_all = TRUE)

  shared <- inner_join(
    db_df %>% select(resno, plddt_db = bfactor),
    local_df %>% select(resno, plddt_local = bfactor),
    by = "resno"
  ) %>%
    mutate(uniprot_id = uid, seq_length = nrow(db_df))

  all_residues[[uid]] <- shared
}

all_residues_df <- bind_rows(all_residues)

# Sort proteins by sequence length for bar chart
proteins_by_len <- results_df %>%
  arrange(seq_length) %>%
  mutate(uniprot_id = fct_inorder(uniprot_id))

# Select 3 representative proteins: short, medium, long
n_prot <- nrow(results_df)
idx_short <- max(1, floor(n_prot * 0.1))
idx_medium <- max(1, floor(n_prot * 0.5))
idx_long <- max(1, floor(n_prot * 0.9))
representative_ids <- c(
  proteins_by_len$uniprot_id[idx_short],
  proteins_by_len$uniprot_id[idx_medium],
  proteins_by_len$uniprot_id[idx_long]
)
representative_labels <- c(
  paste0("Short: ", representative_ids[1]),
  paste0("Medium: ", representative_ids[2]),
  paste0("Long: ", representative_ids[3])
)

# Build pLDDT trace data for representative proteins
trace_data <- list()
for (k in seq_along(representative_ids)) {
  uid <- representative_ids[k]
  i <- which(PROTEINS == uid)
  db_atoms <- parse_ca_atoms(file.path(DB_DIR, paste0(uid, ".pdb")))
  local_atoms <- parse_ca_atoms(find_local_pdb(uid, i))

  if (is.null(db_atoms) || is.null(local_atoms)) next

  db_df <- db_atoms %>% distinct(resno, .keep_all = TRUE) %>% arrange(resno)
  local_df <- local_atoms %>% distinct(resno, .keep_all = TRUE) %>% arrange(resno)

  shared <- inner_join(
    db_df %>% select(resno, plddt_db = bfactor),
    local_df %>% select(resno, plddt_local = bfactor),
    by = "resno"
  ) %>%
    mutate(protein = representative_labels[k])

  trace_data[[k]] <- shared
}

trace_df <- bind_rows(trace_data)

# --- Create 3-panel figure ---
png(RESULT_PNG, width = 12, height = 4.5, units = "in", res = 300)
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 2, 1), oma = c(0, 0, 1, 0))

# Panel 1: Scatter plot of DB vs Local pLDDT
plot(all_residues_df$plddt_db, all_residues_df$plddt_local,
     pch = 16, cex = 0.3, col = rgb(0, 0, 0, 0.15),
     xlab = "DB pLDDT (v2.0)", ylab = "Local pLDDT (v2.3.18)",
     main = "pLDDT Correlation", xlim = c(0, 100), ylim = c(0, 100))
abline(0, 1, col = "red", lwd = 1.5, lty = 2)
# Add regression line
lm_fit <- lm(plddt_local ~ plddt_db, data = all_residues_df)
abline(lm_fit, col = "blue", lwd = 1.5)
legend("bottomright",
       legend = c("y = x", sprintf("r = %.4f", cor(all_residues_df$plddt_db, all_residues_df$plddt_local))),
       col = c("red", "blue"), lty = c(2, 1), lwd = 1.5, cex = 0.8, bty = "n")

# Panel 2: Bar chart of RMSD per protein, sorted by length
bar_colors <- ifelse(proteins_by_len$rmsd_a < 1.0, "steelblue",
                     ifelse(proteins_by_len$rmsd_a < 2.0, "goldenrod", "tomato"))
bp <- barplot(proteins_by_len$rmsd_a, names.arg = proteins_by_len$uniprot_id,
              col = bar_colors, border = NA,
              las = 2, cex.names = 0.65, cex.axis = 0.8,
              ylab = "RMSD (Å)", main = "Cα RMSD by Protein")
abline(h = 1.0, col = "red", lty = 2, lwd = 1)
abline(h = 2.0, col = "darkred", lty = 3, lwd = 1)
legend("topleft", legend = c("< 1.0 Å", "1-2 Å", "> 2 Å"),
       fill = c("steelblue", "goldenrod", "tomato"), cex = 0.7, bty = "n")

# Panel 3: pLDDT traces for 3 representative proteins
if (nrow(trace_df) > 0) {
  trace_proteins <- unique(trace_df$protein)
  trace_colors <- c("firebrick", "dodgerblue", "forestgreen")
  plddt_range <- range(c(trace_df$plddt_db, trace_df$plddt_local), na.rm = TRUE)
  resno_range <- range(trace_df$resno, na.rm = TRUE)

  plot(NULL, xlim = resno_range, ylim = c(0, 100),
       xlab = "Residue Number", ylab = "pLDDT",
       main = "pLDDT Traces (representative)")

  for (k in seq_along(trace_proteins)) {
    td <- trace_df %>% filter(protein == trace_proteins[k])
    col_db <- adjustcolor(trace_colors[k], alpha.f = 0.8)
    col_local <- adjustcolor(trace_colors[k], alpha.f = 0.4)
    lines(td$resno, td$plddt_db, col = col_db, lwd = 1.2)
    lines(td$resno, td$plddt_local, col = col_local, lwd = 1.2, lty = 2)
  }

  # Legend: solid = DB, dashed = Local
  legend("bottomleft",
         legend = c(trace_proteins, "", "— DB (v2.0)", "- - Local (v2.3.18)"),
         col = c(trace_colors, NA, "black", "black"),
         lty = c(1, 1, 1, NA, 1, 2), lwd = c(1.2, 1.2, 1.2, NA, 1.2, 1.2),
         cex = 0.6, bty = "n", ncol = 1)
}

title("AlphaFold v2.0 (DB) vs v2.3.18 (Local)", outer = TRUE, cex.main = 1.1)
dev.off()

cat("Plot saved to:", RESULT_PNG, "\n")

# =============================================================================
# SESSION INFO
# =============================================================================

cat("\n============================================================\n")
cat("Done.\n")
cat("============================================================\n")
cat("Output files:\n")
cat("  - Results CSV:    ", csv_path, "\n")
cat("  - Comparison plot:", RESULT_PNG, "\n")
