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



df_epitopes <- read_csv("data/processed/pos_EL_8-to-14mers_epitopes_hla0201.csv")

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

# Choose flanking region sizes
# 10 is common in literature (captures cleavage context well)
# MHCflurry uses 5, some studies use up to 15
N_FLANK_SIZE <- 10
C_FLANK_SIZE <- 10

cat("Flanking region sizes:\n")
cat("  N-terminal:", N_FLANK_SIZE, "residues\n")
cat("  C-terminal:", C_FLANK_SIZE, "residues\n\n")

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

# Verify the extraction
cat("Verification of flanking extraction:\n")
cat("  Total rows:", nrow(df_with_flanks), "\n")
cat("  Rows with valid N-flank:", sum(!is.na(df_with_flanks$n_flank_seq)), "\n")
cat("  Rows with valid C-flank:", sum(!is.na(df_with_flanks$c_flank_seq)), "\n")

# Check lengths are consistent
n_flank_lengths <- nchar(df_with_flanks$n_flank_seq)
c_flank_lengths <- nchar(df_with_flanks$c_flank_seq)

cat("\n  N-flank length check (all should be", N_FLANK_SIZE, "):", 
    all(n_flank_lengths == N_FLANK_SIZE), "\n")
cat("  C-flank length check (all should be", C_FLANK_SIZE, "):", 
    all(c_flank_lengths == C_FLANK_SIZE), "\n")

# How many are near termini?
cat("\nTerminal epitopes:\n")
cat("  Near N-terminus (padded):", sum(df_with_flanks$near_n_terminus), "\n")
cat("  Near C-terminus (padded):", sum(df_with_flanks$near_c_terminus), "\n")


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
# ============================================================================
# CLEAVAGE POSITION SUMMARY
# ============================================================================

cat("\n=== RESIDUE DISTRIBUTIONS ===\n\n")

position_counts <- df_with_cleavage %>%
  summarise(
    c_term_P1 = paste(names(sort(table(c_term_P1), decreasing = TRUE)[1:3]), collapse = ", "),
    c_term_P1_prime = paste(names(sort(table(c_term_P1_prime), decreasing = TRUE)[1:3]), collapse = ", "),
    n_term_P1 = paste(names(sort(table(n_term_P1), decreasing = TRUE)[1:3]), collapse = ", "),
    n_term_P1_prime = paste(names(sort(table(n_term_P1_prime), decreasing = TRUE)[1:3]), collapse = ", ")
  )

cat("Top 3 residues at each position:\n")
cat("  c_term_P1:       ", position_counts$c_term_P1, "\n")
cat("  c_term_P1_prime: ", position_counts$c_term_P1_prime, "\n")
cat("  n_term_P1:       ", position_counts$n_term_P1, "\n")
cat("  n_term_P1_prime: ", position_counts$n_term_P1_prime, "\n")

# Distribution of C-terminal P1 residues (most important position)
cat("\n=== C-TERMINAL P1 DISTRIBUTION ===\n")
cat("(This is the most important residue for proteasome specificity)\n\n")

p1_distribution <- df_with_cleavage %>%
  count(c_term_P1) %>%
  mutate(percentage = round(n / sum(n) * 100, 2)) %>%
  arrange(desc(n))

print(p1_distribution)
#The proteasome's specificity is largely determined by what residue is at P1. 
#Immunoproteasome strongly prefers hydrophobic (L, F, Y) and basic (K, R) residues.

# ============================================================================
# STEP 5: CALCULATE PHYSICOCHEMICAL PROPERTIES
# ============================================================================
cat("=== CHECKING INPUT DATA ===\n\n")

# Check column exists
cat("Column names containing 'flank':\n")
print(names(df_with_cleavage)[grepl("flank", names(df_with_cleavage))])

# Check for empty strings or NAs in c_flank
cat("\nC-flank column summary:\n")
cat("  Total rows:", length(df_with_cleavage$c_flank), "\n")
cat("  NAs:", sum(is.na(df_with_cleavage$c_flank)), "\n")
cat("  Empty strings:", sum(df_with_cleavage$c_flank == "", na.rm = TRUE), "\n")
cat("  Zero length:", sum(nchar(df_with_cleavage$c_flank) == 0, na.rm = TRUE), "\n")

# Look at some c_flank values
cat("\nFirst 10 c_flank values:\n")
print(head(df_with_cleavage$c_flank, 10))

# Test the calculate_sequence_properties function directly
cat("\n=== TESTING FUNCTION DIRECTLY ===\n")

# Test with a normal sequence
test1 <- calculate_sequence_properties("GILGFVFTL", "test_")
cat("\nTest with normal sequence: ")
cat(if(nrow(test1) == 1) "OK" else "FAILED", "\n")

# Test with empty string
test2 <- calculate_sequence_properties("", "test_")
cat("Test with empty string: ")
cat(if(nrow(test2) == 1) "OK" else "FAILED", "\n")

# Test with NA
test3 <- calculate_sequence_properties(NA, "test_")
cat("Test with NA: ")
cat(if(nrow(test3) == 1) "OK" else "FAILED", "\n")



cat("\n=== CALCULATING PHYSICOCHEMICAL PROPERTIES ===\n\n")

df_with_properties <- df_with_cleavage %>%
  add_physicochemical_properties(
    peptide_col = "peptide",
    n_flank_col = "n_flank",
    c_flank_col = "c_flank"
  )

# Check new columns
property_cols <- names(df_with_properties)[grepl("hydrophobicity|charge|volume|polarity|flexibility|molecular_weight|aromaticity|hydrophobic_fraction", names(df_with_properties))]

cat("\n✓ Added", length(property_cols), "physicochemical property columns\n")
cat("\nProperty columns added:\n")
cat("  ", paste(property_cols, collapse = "\n  "), "\n")

# Summary statistics
cat("\n=== PEPTIDE PROPERTY SUMMARY ===\n\n")

df_with_properties %>%
  summarise(
    hydrophobicity_mean = mean(peptide_hydrophobicity_mean, na.rm = TRUE),
    hydrophobicity_sd = sd(peptide_hydrophobicity_mean, na.rm = TRUE),
    charge_mean = mean(peptide_charge_total, na.rm = TRUE),
    molecular_weight_mean = mean(peptide_molecular_weight, na.rm = TRUE),
    aromaticity_mean = mean(peptide_aromaticity_fraction, na.rm = TRUE)
  ) %>%
  pivot_longer(everything(), names_to = "Property", values_to = "Value") %>%
  print()

# ============================================================================
# STEP 6: ADD PROTEASOME FEATURES
# ============================================================================

cat("\n=== ADDING PROTEASOME FEATURES ===\n\n")

df_with_proteasome <- df_with_properties %>%
  add_proteasome_features()

# Summary of P1 preferences
cat("C-terminal P1 residue analysis:\n")
cat("  Hydrophobic:", sum(df_with_proteasome$c_term_P1_is_hydrophobic), 
    "(", round(mean(df_with_proteasome$c_term_P1_is_hydrophobic) * 100, 1), "%)\n")
cat("  Basic:", sum(df_with_proteasome$c_term_P1_is_basic),
    "(", round(mean(df_with_proteasome$c_term_P1_is_basic) * 100, 1), "%)\n")
cat("  Acidic:", sum(df_with_proteasome$c_term_P1_is_acidic),
    "(", round(mean(df_with_proteasome$c_term_P1_is_acidic) * 100, 1), "%)\n")

cat("\nMean proteasome scores:\n")
cat("  Immunoproteasome:", round(mean(df_with_proteasome$combined_immuno_score, na.rm = TRUE), 3), "\n")
cat("  Standard:", round(mean(df_with_proteasome$combined_standard_score, na.rm = TRUE), 3), "\n")

# ============================================================================
# STEP 7: ADD TAP TRANSPORT FEATURES
# ============================================================================

cat("\n=== ADDING TAP TRANSPORT FEATURES ===\n\n")

df_with_tap <- df_with_proteasome %>%
  add_tap_features(peptide_col = "peptide")

cat("TAP C-terminal analysis:\n")
cat("  Favorable:", sum(df_with_tap$tap_c_term_favorable),
    "(", round(mean(df_with_tap$tap_c_term_favorable) * 100, 1), "%)\n")
cat("  Unfavorable:", sum(df_with_tap$tap_c_term_unfavorable),
    "(", round(mean(df_with_tap$tap_c_term_unfavorable) * 100, 1), "%)\n")

cat("\nLength optimality for TAP:\n")
cat("  Optimal (8-12):", sum(df_with_tap$tap_length_optimal),
    "(", round(mean(df_with_tap$tap_length_optimal) * 100, 1), "%)\n")
cat("  Suboptimal (13-16):", sum(df_with_tap$tap_length_suboptimal),
    "(", round(mean(df_with_tap$tap_length_suboptimal) * 100, 1), "%)\n")

cat("\n✓ Proteasome and TAP features added\n")


write_csv(df_with_tap, "data/processed/epitopes_with_features.csv")


