suppressPackageStartupMessages(library(tidyverse))
source("src/pipeline/r/functions.R")
set_working_directory()

my_packages <- c(
  "tidyverse",
  "Peptides",      
  "Biostrings",    
  "stringr",       
  "progress"       
)

df_epitopes <- read_csv("data/processed/df_combined_pos_and_neg.csv", show_col_types = FALSE)

df_fasta <- read_fasta_df("data/raw/fasta/combined_positives_only.fasta") |>
  mutate(
    accession  = str_extract(header, "(?<=\\|)[^|]+(?=\\|)"),
    uniprot_id = accession,
    is_isoform = str_detect(accession, "-[0-9]+$")
  ) |>
  filter(!is.na(uniprot_id)) |>
  mutate(seq_length = nchar(sequence)) |>
  distinct(uniprot_id, .keep_all = TRUE) |>
  select(-accession, -is_isoform, -seq_length)

# ============================================================================
# STEP 1: MERGE EPITOPE DATA WITH PROTEIN SEQUENCES
# ============================================================================

# Join epitopes to their source proteins
df_merged <- df_epitopes %>%
  left_join(
    df_fasta %>% select(uniprot_id, sequence),
    by = "uniprot_id"
  )

cat("Total epitopes perfectly matched to proteins:", nrow(df_merged), "\n")

# NOTE: Script 04 (04_evaluate_netmhcpan_sensitivity.R) MUST be run first.
# All peptides here already have validated coordinates.

# ============================================================================
# STEP 2: EXTRACT FLANKING REGIONS
# ============================================================================

cat("\n=== EXTRACTING FLANKING REGIONS ===\n\n")

N_FLANK_SIZE <- 10
C_FLANK_SIZE <- 10

df_with_flanks <- df_merged %>%
  extract_flanking_regions(
    sequence_col = "sequence",
    peptide_col = "peptide",
    start_col = "start",
    end_col = "end",
    n_flank_size = N_FLANK_SIZE,
    c_flank_size = C_FLANK_SIZE
  ) |>
  mutate(
    rel_distance_from_n_terminus = distance_from_n_terminus / protein_length,
    rel_distance_from_c_terminus = distance_from_c_terminus / protein_length
  ) |>
  select(-c(distance_from_n_terminus,distance_from_c_terminus))

# ============================================================================
# STEP 2b: PROTEIN-LEVEL PADDING + PADDED SEQUENCE
# ============================================================================

cat("\n=== COMPUTING PROTEIN-LEVEL PADDING ===\n\n")

df_with_flanks <- df_with_flanks |>
  group_by(uniprot_id) |>
  mutate(
    n_pad_len = max(0L, N_FLANK_SIZE - n_flank_len_real),
    c_pad_len = max(0L, C_FLANK_SIZE - c_flank_len_real),
    sequence  = paste0(strrep("X", n_pad_len), sequence, strrep("X", c_pad_len))
  ) |>
  ungroup()

# --- Validations (zero row-dropping) ---
cat("Row count:", nrow(df_with_flanks), "\n")

stopifnot("full_context must be 29 chars" =
            all(nchar(df_with_flanks$full_context) == 29))
stopifnot("n_flank_raw must contain no X" =
            !any(grepl("X", df_with_flanks$n_flank_raw)))
stopifnot("c_flank_raw must contain no X" =
            !any(grepl("X", df_with_flanks$c_flank_raw)))
stopifnot("padded sequence length mismatch" =
            nchar(df_with_flanks$sequence) ==
            df_with_flanks$protein_length +
            df_with_flanks$n_pad_len +
            df_with_flanks$c_pad_len)

cat("  full_context length: 29 (all rows)\n")
cat("  n_flank_raw X-free: TRUE\n")
cat("  c_flank_raw X-free: TRUE\n")
cat("  padded sequence length: consistent\n")
cat("  n_pad_len range:", range(df_with_flanks$n_pad_len), "\n")
cat("  c_pad_len range:", range(df_with_flanks$c_pad_len), "\n")

# ============================================================================
# STEP 3: EXTRACT CLEAVAGE SITE POSITIONS
# ============================================================================

cat("\n=== EXTRACTING CLEAVAGE SITE POSITIONS ===\n\n")

df_with_cleavage <- df_with_flanks %>%
  extract_cleavage_positions(
    peptide_col = "peptide",
    n_flank_col = "n_flank",
    c_flank_col = "c_flank"
  )

write_csv(df_with_cleavage, "data/processed/epitopes_pos_and_neg_features.csv")
cat("\n✅ Saved final extracted features to data/processed/epitopes_pos_and_neg_features.csv\n")

# ============================================================================
# STEP 4: WRITE PADDED NSP3 INPUT FASTA
# ============================================================================

cat("\n=== WRITING PADDED NSP3 INPUT FASTA ===\n\n")

nsp3_fasta_path <- "data/processed/nsp3_input_padded.fasta"

df_fasta_out <- df_with_cleavage |>
  distinct(uniprot_id, sequence) |>
  filter(!is.na(sequence))

# Write FASTA: header is strictly >{uid} (no sp| prefix)
fasta_lines <- df_fasta_out |>
  mutate(header = paste0(">", uniprot_id)) |>
  mutate(block = paste0(header, "\n", sequence)) |>
  pull(block)

writeLines(fasta_lines, nsp3_fasta_path)

cat("  Proteins written:", nrow(df_fasta_out), "\n")
cat("  Output:", nsp3_fasta_path, "\n")
