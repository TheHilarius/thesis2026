# ==============================================================================
# PEPTIDE FEATURE ENGINEERING PIPELINE
# Smart Runner Script
# ============================================================================== 
# ── Configuration ─────────────────────────────────────────────────────────────
FORCE_RERUN    <- TRUE
RUN_ONLY_STEPS <- NULL

# ── Pipeline Steps ────────────────────────────────────────────────────────────
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
#   IN:  data/raw/ (AlphaFold PDB files)
#   OUT: data/processed/df_all.csv
#
# ──────────────────────────────────────────────────────────────────────────────

pipeline_steps <- list(
  list(
    step   = 4,
    script = "src/pipeline/r/04_evaluate_netmhcpan_sensitivity.R",
    output = "data/processed/df_combined_pos_and_neg.csv"
  ),
  list(
    step   = 5,
    script = "src/pipeline/r/05_protein_features.R",
    output = "data/processed/epitopes_pos_and_neg_features.csv"
  ),
  list(
    step   = 6,
    script = "src/pipeline/r/06_netsurfp_protein_features.R",
    output = "data/processed/epitopes_pos_and_neg_features_with_nsp3.csv"
  ),
  list(
    step   = 7,
    script = "src/pipeline/r/07_alpha_fold_features.R",
    output = "data/processed/df_all.csv"
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
cat("\nNITIALIZING PEPTIDE PIPELINE\n")
cat("  FORCE_RERUN:    ", FORCE_RERUN, "\n")
cat("  RUN_ONLY_STEPS: ",
    if (is.null(RUN_ONLY_STEPS)) "all" else paste(RUN_ONLY_STEPS, collapse = ", "), "\n\n")

total_t0 <- Sys.time()

for (step in pipeline_steps) {
  run_step(step$step, step$script, step$output, FORCE_RERUN, RUN_ONLY_STEPS)
}

total_mins <- round(difftime(Sys.time(), total_t0, units = "mins"), 2)
cat(sprintf("PIPELINE COMPLETE IN %s MINS\n", total_mins))
