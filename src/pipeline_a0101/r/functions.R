# ==============================================================================
# HLA-A*01:01 (a0101) EXTERNAL-VALIDATION PIPELINE — shared helpers
#
# This edition scores a second allele with models trained on the legacy
# HLA-A*02:01 pipeline. The legacy pipeline (src/pipeline/r/) is untouched
# and documents the previous dataset; this folder is a parallel edition.
#
# All allele-specific outputs go to data/processed/hla_a0101/.
# Shared, allele-independent resources stay in their legacy locations:
#   - data/processed/structures/  (AlphaFold PDBs — per PROTEIN, not allele)
#   - data/processed/nsp3/        (NetSurfP-3 output — per PROTEIN)
#   - data/raw/fasta/combined_positives_only.fasta (legacy A0201 proteins)
#   - data/raw/fasta/hla_a0101_positives_only.fasta (a0101 proteins; produced
#     by src/pipeline_a0101/fetch/fetch_a0101_fastas.py)
# ==============================================================================

suppressPackageStartupMessages(library(tidyverse))

# Source the shared legacy helpers (read-only reuse — no edits there)
source("src/pipeline/r/functions.R")

# ── Allele constants ──────────────────────────────────────────────────────────
HLA_ALLELE_NAME <- "HLA-A01:01"      # netMHCpan-style name (as in batch output)
A0101_DIR       <- "data/processed/hla_a0101/"

# pp(f): allele-specific processed path. pp("df_all.csv")
#        → "data/processed/hla_a0101/df_all.csv"
pp <- function(f) paste0(A0101_DIR, f)

# netMHCpan batch output directory for this allele
NETMHCPAN_DIR_A0101 <- paste0(A0101_DIR, "netmhcpan/")

# ── Shared inputs ─────────────────────────────────────────────────────────────
STRUCTURES_DIR <- "data/processed/structures/alphafold/"   # shared (per protein)
STRUCTURE_LOGS <- "data/processed/structures/logs/"        # shared
NSP3_DIR       <- "data/processed/nsp3/"                   # shared (per protein)

FASTA_LEGACY <- "data/raw/fasta/combined_positives_only.fasta"
FASTA_A0101  <- "data/raw/fasta/hla_a0101_positives_only.fasta"

# ── FASTA database: union of legacy + a0101 proteins ─────────────────────────
# Mirrors the parsing in legacy 04/05 (accession extraction, isoform flags,
# distinct uniprot_id) but binds both FASTA files. Peptides from proteins in
# neither file are dropped by the downstream semi_join (same semantics as the
# legacy pipeline).
load_fasta_db <- function() {
  files <- c(FASTA_LEGACY, FASTA_A0101)
  files <- files[file.exists(files)]
  if (length(files) == 0) {
    stop("No FASTA database found. Run fetch_a0101_fastas.py first.")
  }
  if (!file.exists(FASTA_A0101)) {
    message("NOTE: ", FASTA_A0101, " not found — using legacy FASTA only. ",
            "a0101 peptides from proteins outside the legacy database will ",
            "be dropped. Run src/pipeline_a0101/fetch/fetch_a0101_fastas.py.")
  }

  purrr::map(files, function(f) {
    read_fasta_df(f) |>
      mutate(
        accession  = str_extract(header, "(?<=\\|)[^|]+(?=\\|)"),
        uniprot_id = accession,
        is_isoform = str_detect(accession, "-[0-9]+$")
      ) |>
      filter(!is.na(uniprot_id))
  }) |>
    list_rbind() |>
    mutate(seq_length = nchar(sequence)) |>
    distinct(uniprot_id, .keep_all = TRUE)
}

# Ensure allele output + figure directories exist
ensure_a0101_dirs <- function() {
  dir.create(A0101_DIR, recursive = TRUE, showWarnings = FALSE)
  dir.create(NETMHCPAN_DIR_A0101, recursive = TRUE, showWarnings = FALSE)
  dir.create("results/figures/netmhcpan_hla_a0101", recursive = TRUE,
             showWarnings = FALSE)
}
