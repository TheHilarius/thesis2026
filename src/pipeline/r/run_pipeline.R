# ==============================================================================
# PEPTIDE FEATURE ENGINEERING PIPELINE
# Smart Runner Script
# ============================================================================== 
# ── Configuration ─────────────────────────────────────────────────────────────
FORCE_RERUN    <- TRUE
RUN_ONLY_STEPS <- NULL

source("src/pipeline/r/functions.R")   # parse_allele_arg, pp, *_FOR helpers

# ── Allele selection (default hla0201; legacy CLI compatible) ─────────────────
#   Rscript src/pipeline/r/run_pipeline.R [hla0201|hla_a0101]
#   or several at once:  PIPELINE_ALLELES=hla0201,hla_a0101 Rscript ...
alleles_env <- Sys.getenv("PIPELINE_ALLELES")
if (nzchar(alleles_env)) {
  ALLELES <- trimws(strsplit(alleles_env, ",")[[1]])
} else {
  cli <- commandArgs(trailingOnly = TRUE)
  ALLELES <- if (length(cli) == 0) "hla0201" else cli[1]
}
ALLELES <- match.arg(ALLELES, choices = ALLELE_TAGS, several.ok = TRUE)

# ── Window width (FLANK_SIZE: 5 = 19-mer default, 10 = legacy 29-mer) ────────
FLANK_SIZE <- as.integer(Sys.getenv("FLANK_SIZE", "5"))
if (!FLANK_SIZE %in% c(5L, 10L)) {
  stop("FLANK_SIZE must be 5 or 10, got: ", Sys.getenv("FLANK_SIZE"))
}
Sys.setenv(FLANK_SIZE = FLANK_SIZE)   # sourced steps read it via Sys.getenv

# ── Pipeline Steps ────────────────────────────────────────────────────────────
# All IN/OUT paths are per-allele: flat for hla0201, data/processed/hla_a0101/
# for hla_a0101 (pp()/ *_FOR helpers). Shared resources (structures/, nsp3/)
# stay flat. Width comes from FLANK_SIZE (set below).
#
# Step 02: load_iedb_data.R
#   IN:  data/raw/mhc_ligand_table_export_*.csv
#   OUT: pos_EL_all_epitopes_<allele>.csv (POS_ALL_FOR)
#
# Step 04: evaluate_netmhcpan_sensitivity.R
#   IN:  data/processed/pos_EL_all_epitopes_hla0201.csv
#   IN:  data/processed/netmhcpan/ (batch files)
#   OUT: data/processed/df_combined_pos_and_neg.csv
#
# Step 05: protein_features.R
#   IN:  data/processed/df_combined_pos_and_neg.csv
#   IN:  data/raw/fasta/combined_8_to_14mer.fasta
#   OUT: data/processed/epitopes_pos_and_neg_features.csv
#
# Step 06: netsurfp_protein_features.R
#   IN:  data/processed/epitopes_pos_and_neg_features.csv
#   IN:  NSP3 batch CSV files
#   OUT: data/processed/epitopes_pos_and_neg_features_with_nsp3.csv
#
# Step 07: alpha_fold_features.R
#   IN:  data/processed/epitopes_pos_and_neg_features_with_nsp3.csv
#   IN:  data/processed/structures/alphafold/  (SHARED, not per-allele)
#   OUT: data/processed/df_all.csv   (includes context-window dedup)
#
# Step 08: aa_encoding.R
#   IN:  data/processed/df_all.csv
#   OUT: df_<enc>.csv + df_all_<enc>.csv + compositional (per allele)
#
# ──────────────────────────────────────────────────────────────────────────────

pipeline_steps_for <- function(allele) list(
  list(
    step   = 2,
    script = "src/pipeline/r/02_load_iedb_data.R",
    output = POS_ALL_FOR(allele)
  ),
  list(
    step   = 4,
    script = "src/pipeline/r/04_evaluate_netmhcpan_sensitivity.R",
    output = pp("df_combined_pos_and_neg.csv", allele)
  ),
  list(
    step   = 5,
    script = "src/pipeline/r/05_protein_features.R",
    output = pp("epitopes_pos_and_neg_features.csv", allele)
  ),
  list(
    step   = 6,
    script = "src/pipeline/r/06_netsurfp_protein_features.R",
    output = pp("epitopes_pos_and_neg_features_with_nsp3.csv", allele)
  ),
  list(
    step   = 7,
    script = "src/pipeline/r/07_alpha_fold_features.R",
    output = pp("df_all.csv", allele)
  ),
  list(
    step   = 8,
    script = "src/pipeline/r/08_aa_encoding.R",
    output = pp("df_blosum50.csv", allele)
  )
)

run_step <- function(step_num, script_path, output_path, force_rerun, run_only) {
  if (!file.exists(script_path)) {
    stop("\u274C Script not found: ", script_path)
  }
  
  # Skip if not in the run-only list
  if (!is.null(run_only) && !(step_num %in% run_only)) {
    cat(sprintf("\u23ED\uFE0F  SKIPPING step %02d: %s (not in RUN_ONLY_STEPS)\n\n",
                step_num, basename(script_path)))
    return(invisible(TRUE))
  }
  
  # Skip if output exists and not forcing rerun
  if (file.exists(output_path) && !force_rerun && is.null(run_only)) {
    cat(sprintf("\u23ED\uFE0F  SKIPPING step %02d: %s\n    \u2514\u2500 Output exists: %s\n\n",
                step_num, basename(script_path), basename(output_path)))
    return(invisible(TRUE))
  }
  
  cat(sprintf("\u25B6\uFE0F  RUNNING step %02d: %s\n", step_num, basename(script_path)))
  t0 <- Sys.time()
  
  source(script_path)
  
  elapsed <- round(difftime(Sys.time(), t0, units = "mins"), 2)
  cat(sprintf("\u2705 FINISHED step %02d: %s (%s mins)\n\n",
              step_num, basename(script_path), elapsed))
}

# ── Execute ───────────────────────────────────────────────────────────────────
cat("\nINITIALIZING PEPTIDE PIPELINE\n")
cat("  FORCE_RERUN:    ", FORCE_RERUN, "\n")
cat("  RUN_ONLY_STEPS: ",
    if (is.null(RUN_ONLY_STEPS)) "all" else paste(RUN_ONLY_STEPS, collapse = ", "), "\n")
cat("  FLANK_SIZE:     ", FLANK_SIZE, " (", 2 * FLANK_SIZE + 9,
    "-mer window)\n", sep = "")
cat("  Alleles:        ", paste(ALLELES, collapse = ", "), "\n\n", sep = "")

total_t0 <- Sys.time()

for (allele in ALLELES) {
  Sys.setenv(ALLELE_TAG = allele)   # sourced steps resolve via parse_allele_arg()
  cat(strrep("=", 70), "\n")
  cat("  ALLELE: ", allele, " | FLANK_SIZE: ", FLANK_SIZE, " -> ",
      2 * FLANK_SIZE + 9, "-mer window\n", sep = "")
  cat(strrep("=", 70), "\n")
  for (step in pipeline_steps_for(allele)) {
    run_step(step$step, step$script, step$output, FORCE_RERUN, RUN_ONLY_STEPS)
  }
}

total_mins <- round(difftime(Sys.time(), total_t0, units = "mins"), 2)
cat(sprintf("PIPELINE COMPLETE (%s) IN %s MINS\n",
            paste(ALLELES, collapse = ","), total_mins))
