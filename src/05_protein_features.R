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
load_required_packages(my_packages)



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

# Look at which proteins are missing. Missing unitprot ID in either epitope or fasta will cause this.
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
# VERIFY THE EXTRACTION
# ============================================================================

cat("Verification:\n")

# Check all flanks have correct length
n_flank_lengths <- nchar(df_with_flanks$n_flank_seq)
c_flank_lengths <- nchar(df_with_flanks$c_flank_seq)

cat("  All N-flanks are", N_FLANK_SIZE, "residues:", all(n_flank_lengths == N_FLANK_SIZE), "\n")
cat("  All C-flanks are", C_FLANK_SIZE, "residues:", all(c_flank_lengths == C_FLANK_SIZE), "\n")

# Summary of terminal epitopes
cat("\nEpitopes near protein boundaries:\n")
cat("  Near N-terminus (start of protein):", sum(df_with_flanks$near_n_terminus), 
    "(", round(mean(df_with_flanks$near_n_terminus) * 100, 2), "%)\n")
cat("  Near C-terminus (end of protein):", sum(df_with_flanks$near_c_terminus),
    "(", round(mean(df_with_flanks$near_c_terminus) * 100, 2), "%)\n")
cat("  Internal epitopes (full context):", 
    sum(!df_with_flanks$near_n_terminus & !df_with_flanks$near_c_terminus), "\n")

# ============================================================================
# EXAMINE EXAMPLES
# ============================================================================
df_with_flanks <- df_with_flanks %>%
  as_tibble() %>%
  mutate(across(where(is.character), as.character))  # Force character columns

cat("\n=== EXAMPLE RESULTS ===\n\n")

# Show a few examples
cat("Example 1 - Internal epitope (full flanking regions):\n")
example_internal <- df_with_flanks %>%
  filter(!near_n_terminus & !near_c_terminus) %>%
  head(1)

cat("  Peptide:        ", example_internal$peptide[1], "\n")
cat("  Position:       ", example_internal$start[1], "-", example_internal$end[1], 
    " (protein length:", example_internal$protein_length[1], ")\n")
cat("  N-flank:        ", example_internal$n_flank_seq[1], "\n")
cat("  C-flank:        ", example_internal$c_flank_seq[1], "\n")
cat("  Full context:   ", example_internal$full_context[1], "\n")

# ============================================================================
# STEP 4: EXTRACT CLEAVAGE SITE POSITIONS
# ============================================================================

cat("\n=== EXTRACTING CLEAVAGE SITE POSITIONS ===\n\n")

# Extract cleavage positions
df_with_cleavage <- df_with_flanks %>%
  extract_cleavage_positions(
    peptide_col = "peptide",
    n_flank_col = "n_flank_seq",
    c_flank_col = "c_flank_seq"
  )

# Verify extraction
cat("Cleavage position columns added:\n")
cleavage_cols <- names(df_with_cleavage)[grepl("cleavage", names(df_with_cleavage))]
cat(" ", paste(cleavage_cols, collapse = "\n  "), "\n")

# Show example
cat("\n=== EXAMPLE CLEAVAGE POSITIONS ===\n\n")

example <- df_with_cleavage %>% head(1)

cat("Epitope:", example$peptide[1], "\n")
cat("N-flank:", example$n_flank_seq[1], "\n")
cat("C-flank:", example$c_flank_seq[1], "\n")

cat("\nN-terminal cleavage site (where epitope begins):\n")
cat("  ...─", example$n_cleavage_P4[1], "─", example$n_cleavage_P3[1], "─", 
    example$n_cleavage_P2[1], "─", example$n_cleavage_P1[1], "─┃─", 
    example$n_cleavage_P1_prime[1], "─", example$n_cleavage_P2_prime[1], "─",
    example$n_cleavage_P3_prime[1], "─", example$n_cleavage_P4_prime[1], "─...\n")
cat("       P4   P3   P2   P1  ┃  P1'  P2'  P3'  P4'\n")
cat("                         ↑\n")
cat("                   Cut here\n")

cat("\nC-terminal cleavage site (where epitope ends):\n")
cat("  ...─", example$c_cleavage_P4[1], "─", example$c_cleavage_P3[1], "─", 
    example$c_cleavage_P2[1], "─", example$c_cleavage_P1[1], "─┃─", 
    example$c_cleavage_P1_prime[1], "─", example$c_cleavage_P2_prime[1], "─",
    example$c_cleavage_P3_prime[1], "─", example$c_cleavage_P4_prime[1], "─...\n")
cat("       P4   P3   P2   P1  ┃  P1'  P2'  P3'  P4'\n")
cat("                         ↑\n")
cat("                   Cut here\n")

cat("\n★ C-terminal P1 =", example$c_cleavage_P1[1], "(most important for proteasome specificity)\n")

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

# ============================================================================
# STEP 5: CALCULATE PHYSICOCHEMICAL PROPERTIES
# ============================================================================

cat("\n=== CALCULATING PHYSICOCHEMICAL PROPERTIES ===\n\n")

df_with_properties <- df_with_cleavage %>%
  add_physicochemical_properties(
    peptide_col = "peptide",
    n_flank_col = "n_flank_seq",
    c_flank_col = "c_flank_seq"
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
