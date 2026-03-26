library(tidyverse)
source("src/functions.R")
set_working_directory()


my_packages <- c(
  "tidyverse",
  "Peptides",      # For physicochemical calculations
  "Biostrings",    # For sequence handling
  "stringr",       # String manipulation
  "progress"       # Progress bars
)
#load_required_packages(my_packages)



df_epitopes <- read_csv("data/processed/df_combined_pos_and_neg.csv")

df_fasta <- read_fasta_df("data/raw/fasta/combined_8_to_14mer.fasta") |>
  mutate(uniprot_id = str_extract(header, "(?<=\\|)[A-Z0-9]+(?=\\|)"))

# ============================================================================
# STEP 1: MERGE EPITOPE DATA WITH PROTEIN SEQUENCES
# ============================================================================

# Join epitopes to their source proteins
df_merged <- df_epitopes %>%
  left_join(
    df_fasta %>% select(uniprot_id, sequence),
    by = "uniprot_id"
  )

# Check the merge success
cat("Total epitopes:", nrow(df_epitopes), "\n")
cat("Epitopes with matched protein:", sum(!is.na(df_merged$sequence)), "\n")
cat("Epitopes WITHOUT matched protein:", sum(is.na(df_merged$sequence)), "\n")

# Look at which proteins are missing. Cause by protein modifications

missing_proteins <- df_merged %>%
  filter(is.na(sequence)) %>%
  distinct(uniprot_id)

if (nrow(missing_proteins) > 0) {
  cat("\nMissing protein IDs (first 10):\n")
  print(head(missing_proteins, 10))
}
# ============================================================================
# STEP 2: VALIDATE AND FIX POSITIONS
# ============================================================================

cat("\n=== VALIDATING AND FIXING POSITIONS ===\n\n")

# Apply validation and fixing
df_validated <- df_merged %>%
  filter(!is.na(sequence)) %>%
  validate_and_fix_positions(
    sequence_col = "sequence",
    peptide_col = "peptide",
    start_col = "start",
    end_col = "end"
  )

# Show summary of what happened
cat("Position validation summary:\n")
position_summary <- df_validated %>%
  count(position_status) %>%
  mutate(percentage = round(n / sum(n) * 100, 2))
print(position_summary)

# ============================================================================
# FILTER AND CLEAN DATA
# ============================================================================

# Keep only valid_original and fixed cases
# Remove tracking columns as we've completed validation
#
# NOTE: From this point forward, the data has been:
#   1. Filtered to only include epitopes with matched protein sequences
#   2. Validated to ensure peptide positions are correct
#   3. Fixed where positions were incorrect but peptide was found in protein
#   4. Removed where peptide could not be found in protein (unfixable)
#
# The validation tracking columns (start_original, end_original, 
# position_originally_valid, position_status) are removed as they 
# have served their purpose. This information is not lost - it was
# used to ensure data quality before proceeding.

df_clean <- df_validated %>%
  filter(position_status %in% c("valid_original", "fixed")) %>%
  select(
    -start_original,
    -end_original,
    -position_originally_valid,
    -position_status
  )

# Verify all positions are now correct
df_verification <- df_clean %>%
  mutate(
    verify_extract = substr(sequence, start, end),
    verify_match = (peptide == verify_extract)
  )

cat("\n✓ Verification complete - all positions correct:", 
    all(df_verification$verify_match), "\n")

# Clean up verification dataframe (remove verification columns)
df_clean <- df_verification %>%
  select(-verify_extract, -verify_match)

cat("\nFinal clean dataset:", nrow(df_clean), "epitopes\n")

# ============================================================================
# STEP 3: EXTRACT FLANKING REGIONS
# ============================================================================

cat("\n=== EXTRACTING FLANKING REGIONS ===\n\n")

# Flanking region sizes
# 10 residues is common in literature and captures the key cleavage context
# MHCflurry 2.0 uses 5 residues; some studies use up to 15
N_FLANK_SIZE <- 10
C_FLANK_SIZE <- 10

cat("Configuration:\n")
cat("  N-terminal flanking size:", N_FLANK_SIZE, "residues\n")
cat("  C-terminal flanking size:", C_FLANK_SIZE, "residues\n")
cat("  Full context will be:", N_FLANK_SIZE, "+ peptide_length +", C_FLANK_SIZE, "residues\n\n")

# Extract flanking regions
df_with_flanks <- df_clean %>%
  extract_flanking_regions(
    sequence_col = "sequence",
    peptide_col = "peptide",
    start_col = "start",
    end_col = "end",
    n_flank_size = N_FLANK_SIZE,
    c_flank_size = C_FLANK_SIZE
  )

# ============================================================================
# FLANKING REGION SIZE VERIFICATION
# ============================================================================

cat("N-flank lengths:\n")
n_flank_dist <- df_with_flanks %>%
  count(n_flank_length = nchar(n_flank)) %>%
  mutate(percent = round(n / sum(n) * 100, 2))
print(n_flank_dist)

cat("\nC-flank lengths:\n")
c_flank_dist <- df_with_flanks %>%
  count(c_flank_length = nchar(c_flank)) %>%
  mutate(percent = round(n / sum(n) * 100, 2))
print(c_flank_dist)

# ============================================================================
# EXAMINE EXAMPLES
# ============================================================================
df_with_flanks <- df_with_flanks %>%
  as_tibble() %>%
  mutate(across(where(is.character), as.character))  # Force character columns

# ============================================================================
# STEP 4: EXTRACT CLEAVAGE SITE POSITIONS
# ============================================================================

cat("\n=== EXTRACTING CLEAVAGE SITE POSITIONS ===\n\n")

# Extract cleavage positions
df_with_cleavage <- df_with_flanks %>%
  extract_cleavage_positions(
    peptide_col = "peptide",
    n_flank_col = "n_flank",
    c_flank_col = "c_flank"
  )

# CLEAVAGE POSITION SUMMARY

cat("\n=== RESIDUE DISTRIBUTIONS ===\n\n")

# Helper to get top 3 residues from a column
top3 <- function(x) {
  tbl <- sort(table(x[!is.na(x)]), decreasing = TRUE)
  paste(names(tbl)[1:min(3, length(tbl))], collapse = ", ")
}

# All 16 position columns
position_cols <- c(
  paste0("n_cleavage_P", 4:1),
  paste0("n_cleavage_P", 1:4, "_prime"),
  paste0("c_cleavage_P", 4:1),
  paste0("c_cleavage_P", 1:4, "_prime")
)

cat("Top 3 residues at each position:\n\n")
cat("--- N-terminal cleavage site ---\n")
for (col in position_cols[1:8]) {
  cat(sprintf("  %-22s %s\n", paste0(col, ":"), top3(df_with_cleavage[[col]])))
}
cat("\n--- C-terminal cleavage site ---\n")
for (col in position_cols[9:16]) {
  cat(sprintf("  %-22s %s\n", paste0(col, ":"), top3(df_with_cleavage[[col]])))
}

# Distribution of C-terminal P1 residues (most important position)
cat("\n=== C-TERMINAL P1 DISTRIBUTION ===\n")
cat("(This is the most important residue for proteasome specificity)\n\n")

p1_distribution <- df_with_cleavage %>%
  count(c_cleavage_P1) %>%
  mutate(percentage = round(n / sum(n) * 100, 2)) %>%
  arrange(desc(n))

print(p1_distribution)

#The proteasome's specificity is largely determined by what residue is at P1. 
#Immunoproteasome strongly prefers hydrophobic (L, F, Y) and basic (K, R) residues.

write_csv(df_with_cleavage, "data/processed/epitopes_pos_and_neg_features.csv")
