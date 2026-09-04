library(tidyverse)
source("src/pipeline/r/functions.R")
set_working_directory()

my_packages <- c(
  "tidyverse",
  "Peptides",      
  "Biostrings",    
  "stringr",       
  "progress"       
)

df_epitopes <- read_csv("data/processed/df_combined_pos_and_neg.csv")

df_fasta <- read_fasta_df("data/raw/fasta/combined_9mer.fasta") |>
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

# ============================================================================
# STEP 2: VALIDATE AND FIX POSITIONS
# ============================================================================

cat("\n=== VALIDATING AND FIXING POSITIONS ===\n\n")

df_validated <- df_merged %>%
  validate_and_fix_positions(
    sequence_col = "sequence",
    peptide_col = "peptide",
    start_col = "start",
    end_col = "end"
  )

cat("Position validation summary:\n")
position_summary <- df_validated %>%
  count(position_status) %>%
  mutate(percentage = round(n / sum(n) * 100, 2))
print(position_summary)

# ============================================================================
# FILTER AND CLEAN DATA
# ============================================================================

df_clean <- df_validated %>%
  filter(position_status %in% c("valid_original", "fixed")) %>%
  select(
    -start_original,
    -end_original,
    -position_originally_valid,
    -position_status
  )

df_verification <- df_clean %>%
  mutate(
    verify_extract = substr(sequence, start, end),
    verify_match = (peptide == verify_extract)
  )

cat("\n✓ Verification complete - all positions correct:", 
    all(df_verification$verify_match), "\n")

df_clean <- df_verification %>%
  select(-verify_extract, -verify_match)

cat("\nFinal clean dataset:", nrow(df_clean), "epitopes\n")

# ============================================================================
# STEP 3: EXTRACT FLANKING REGIONS
# ============================================================================

cat("\n=== EXTRACTING FLANKING REGIONS ===\n\n")

N_FLANK_SIZE <- 10
C_FLANK_SIZE <- 10

df_with_flanks <- df_clean %>%
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
# STEP 4: EXTRACT CLEAVAGE SITE POSITIONS
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