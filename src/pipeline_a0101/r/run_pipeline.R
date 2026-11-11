# ==============================================================================
# HLA-A*01:01 EXTERNAL-VALIDATION PIPELINE — Smart Runner
# Edition of legacy src/pipeline/r/run_pipeline.R.
#
# Runs the a0101 R chain (steps 02→08) producing the external-validation
# dataset in data/processed/hla_a0101/. The legacy pipeline and its outputs
# are untouched.
#
# PREREQUISITES (run before this script):
#   1. netMHCpan 9-mer predictions for HLA-A*01:01 (server-side):
#        bash src/pipeline_a0101/netmhcpan/run_netmhcpan_parallel.sh \
#             <fasta_batch_dir> data/processed/hla_a0101/netmhcpan hla_a0101 10
#      (batch FASTAs come from the legacy batch splitter; see README.md)
#   2. After step 02, run the a0101 FASTA fetch:
#        python src/pipeline_a0101/fetch/fetch_a0101_fastas.py
#      (steps 04+ pick the file up automatically; rerun this runner afterwards)
#
# Step 02: load IEDB a0101 positives → hla_a0101/pos_EL_all_epitopes_hla_a0101.csv
# Step 04: netMHCpan sensitivity → hla_a0101/df_combined_pos_and_neg.csv
# Step 05: protein features    → hla_a0101/epitopes_pos_and_neg_features.csv
# Step 06: NetSurfP-3 features → hla_a0101/epitopes_pos_and_neg_features_with_nsp3.csv
# Step 07: AlphaFold features  → hla_a0101/df_all.csv
# Step 08: AA encodings        → hla_a0101/df_<enc>.csv + df_all_<enc>.csv
# ==============================================================================

# ── Configuration ─────────────────────────────────────────────────────────────
FORCE_RERUN    <- TRUE
RUN_ONLY_STEPS <- NULL

# ── Pipeline Steps ────────────────────────────────────────────────────────────
pipeline_steps <- list(
  list(
    step   = 2,
    script = "src/pipeline_a0101/r/02_load_iedb_data.R",
    output = "data/processed/hla_a0101/pos_EL_all_epitopes_hla_a0101.csv"
  ),
  list(
    step   = 4,
    script = "src/pipeline_a0101/r/04_evaluate_netmhcpan_sensitivity.R",
    output = "data/processed/hla_a0101/df_combined_pos_and_neg.csv"
  ),
  list(
    step   = 5,
    script = "src/pipeline_a0101/r/05_protein_features.R",
    output = "data/processed/hla_a0101/epitopes_pos_and_neg_features.csv"
  ),
  list(
    step   = 6,
    script = "src/pipeline_a0101/r/06_netsurfp_protein_features.R",
    output = "data/processed/hla_a0101/epitopes_pos_and_neg_features_with_nsp3.csv"
  ),
  list(
    step   = 7,
    script = "src/pipeline_a0101/r/07_alpha_fold_features.R",
    output = "data/processed/hla_a0101/df_all.csv"
  ),
  list(
    step   = 8,
    script = "src/pipeline_a0101/r/08_aa_encoding.R",
    output = "data/processed/hla_a0101/df_all_sparse.csv"
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
cat("\nINITIALIZING HLA-A*01:01 EXTERNAL-VALIDATION PIPELINE\n")
cat("  FORCE_RERUN:    ", FORCE_RERUN, "\n")
cat("  RUN_ONLY_STEPS: ",
    if (is.null(RUN_ONLY_STEPS)) "all" else paste(RUN_ONLY_STEPS, collapse = ", "), "\n\n")

total_t0 <- Sys.time()

for (step in pipeline_steps) {
  run_step(step$step, step$script, step$output, FORCE_RERUN, RUN_ONLY_STEPS)
}

total_mins <- round(difftime(Sys.time(), total_t0, units = "mins"), 2)
cat(sprintf("A0101 PIPELINE COMPLETE IN %s MINS\n", total_mins))
cat("Next: encodings are in data/processed/hla_a0101/ — run\n")
cat("  python src/pipeline_a0101/python/14_predict_external.py (see README.md)\n")
