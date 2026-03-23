# ==============================================================================
# PEPTIDE FEATURE ENGINEERING PIPELINE
# Master Runner Script
# ==============================================================================

# ── SCRIPT REGISTRY ───────────────────────────────────────────────────────────
#
# SCRIPT 04: evaluate_netmhcpan_sensitivity.R
#   IN:   data/processed/pos_EL_all_epitopes_hla0201.csv
#   IN:   data/processed/netmhcpan/ (Batch files)
#   OUT:  data/processed/df_combined_pos_and_neg.csv (Labeled ML dataset)
#
# SCRIPT 05: protein_features.R
#   IN:   data/processed/df_combined_pos_and_neg.csv
#   IN:   data/raw/fasta/combined_8_to_14mer.fasta
#   OUT:  data/processed/epitopes_pos_and_neg_features.csv (Adds flanking/cleavage)
#
# SCRIPT 06: netsurfp_protein_features.R
#   IN:   data/processed/epitopes_pos_and_neg_features.csv
#   IN:   NSP3 batch CSV files
#   OUT:  data/processed/epitopes_with_nsp3.csv (Adds secondary structure/RSA)
#
# SCRIPT 07: alpha_fold_features.R
#   IN:   data/processed/epitopes_pos_and_neg_features.csv 
#   IN:   data/raw/ (AlphaFold PDB files)
#   OUT:  data/processed/alphafold_plddt_features.csv (Scalar summaries)
#   OUT:  data/processed/alphafold_plddt_features_with_vectors.rds (Full vectors)
#
# ──────────────────────────────────────────────────────────────────────────────
# ==============================================================================
# PEPTIDE FEATURE ENGINEERING PIPELINE
# Smart Runner Script (Skips steps if output already exists)
# ==============================================================================

# Toggle this to TRUE if you want to force the pipeline to overwrite existing files
FORCE_RERUN <- FALSE

# ── Define the Pipeline Steps (Script -> Expected Output) ─────────────────────
pipeline_steps <- list(
  list(
    script = "src/04_evaluate_netmhcpan_sensitivity.R",
    output = "data/processed/df_combined_pos_and_neg.csv"
  ),
  list(
    script = "src/05_protein_features.R",
    output = "data/processed/epitopes_pos_and_neg_features.csv"
  ),
  list(
    script = "src/06_netsurfp_protein_features.R",
    output = "data/processed/epitopes_pos_and_neg_features_with_nsp3.csv"
  ),
  list(
    script = "src/07_alpha_fold_features.R",
    output = "data/processed/alphafold_plddt_features.csv"
  ),
  list(
    script = "src/08_combine_all_features.R",
    output = "data/processed/final_ml_dataset.csv"
  )
)

run_step <- function(script_path, output_path, force_rerun) {
  if (!file.exists(script_path)) {
    stop("❌ Script not found: ", script_path)
  }
  
  # Check if we can skip this step
  if (file.exists(output_path) && !force_rerun) {
    cat(sprintf("⏭️  SKIPPING: %s\n    └─ Output already exists: %s\n\n", 
                basename(script_path), basename(output_path)))
    return(invisible(TRUE))
  }
  
  # Otherwise, run the script
  cat(sprintf("▶️  RUNNING: %s\n", basename(script_path)))
  t0 <- Sys.time()
  
  source(script_path)
  
  t1 <- Sys.time()
  mins <- round(difftime(t1, t0, units = "mins"), 2)
  cat(sprintf("✅ FINISHED: %s (Took %s mins)\n\n", basename(script_path), mins))
}

# ── EXECUTE ───────────────────────────────────────────────────────────────────
cat("\n🚀 INITIALIZING SMART PEPTIDE PIPELINE 🚀\n\n")
total_t0 <- Sys.time()

for (step in pipeline_steps) {
  run_step(step$script, step$output, FORCE_RERUN)
}

total_mins <- round(difftime(Sys.time(), total_t0, units = "mins"), 2)
cat(sprintf("🎉 PIPELINE COMPLETE IN %s MINS! 🎉\n", total_mins))