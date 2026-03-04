library(tidyverse)
library(stringr)
source("src/functions.R")

# === User-specific working directory ===
current_user <- Sys.info()[["user"]]

if (current_user == "olive") {
  setwd("C:/Users/olive/Documents/R/special_course_spring2026")
} else if (current_user == "mj607") {
  setwd("//wsl$/Ubuntu/home/hilarius/special_course_spring2026")
} else if (current_user == "hilarius") {
  setwd("/Users/hilarius/Desktop/DTU/special_course_spring2026")
} else if (current_user == "Hilarius") {
  setwd("C:/Users/Hilarius/OneDrive - Danmarks Tekniske Universitet/Skrivebord/special_course_spring2026/special_course_spring2026")
} else {
  stop("Unknown user. Please set working directory manually.")
}

# Flanking functions
extract_flanks_safe <- function(peptide, protein_seq, flank_size = 8) {
  
  if (is.na(peptide) || is.na(protein_seq) || peptide == "" || protein_seq == "") {
    return(list(
      n_flank = NA_character_,
      c_flank = NA_character_,
      full_context = NA_character_,
      start = NA_integer_,
      end = NA_integer_,
      found = FALSE
    ))
  }
  
  # Find peptide in protein (1-indexed in R)
  match_pos <- regexpr(peptide, protein_seq, fixed = TRUE)
  
  if (match_pos == -1) {
    return(list(
      n_flank = NA_character_,
      c_flank = NA_character_,
      full_context = NA_character_,
      start = NA_integer_,
      end = NA_integer_,
      found = FALSE
    ))
  }
  
  # Calculate positions (1-indexed)
  pep_start <- as.integer(match_pos)
  pep_length <- nchar(peptide)
  pep_end <- pep_start + pep_length - 1
  prot_length <- nchar(protein_seq)
  
  # === N-terminal flank ===
  n_flank_start <- max(1, pep_start - flank_size)
  n_flank_end <- pep_start - 1
  
  if (n_flank_end >= n_flank_start) {
    n_flank <- substr(protein_seq, n_flank_start, n_flank_end)
  } else {
    n_flank <- ""
  }
  n_flank_padded <- paste0(strrep("X", flank_size - nchar(n_flank)), n_flank)
  
  # === C-terminal flank ===
  c_flank_start <- pep_end + 1
  c_flank_end <- min(prot_length, pep_end + flank_size)
  
  if (c_flank_end >= c_flank_start) {
    c_flank <- substr(protein_seq, c_flank_start, c_flank_end)
  } else {
    c_flank <- ""
  }
  c_flank_padded <- paste0(c_flank, strrep("X", flank_size - nchar(c_flank)))
  
  # === Full context ===
  full_context <- paste0(n_flank_padded, peptide, c_flank_padded)
  
  return(list(
    n_flank = n_flank_padded,
    c_flank = c_flank_padded,
    full_context = full_context,
    start = pep_start,
    end = pep_end,
    found = TRUE
  ))
}

add_flanking_regions <- function(df, df_fasta, flank_size = 8) {
  
  seq_lookup <- setNames(df_fasta$sequence, df_fasta$uniprot_id)
  
  results <- df |>
    mutate(row_id = row_number()) |>
    rowwise() |>
    mutate(
      protein_seq = seq_lookup[uniprot_id],
      flank_data = list(extract_flanks_safe(peptide, protein_seq, flank_size))
    ) |>
    ungroup() |>
    mutate(
      n_flank = map_chr(flank_data, ~ .x$n_flank),
      c_flank = map_chr(flank_data, ~ .x$c_flank),
      full_context = map_chr(flank_data, ~ .x$full_context),
      start = map_int(flank_data, ~ as.integer(.x$start)),
      end = map_int(flank_data, ~ as.integer(.x$end)),
      peptide_found = map_lgl(flank_data, ~ .x$found)
    ) |>
    select(-protein_seq, -flank_data, -row_id)
  
  return(results)
}


# === Load and clean raw data ===
df_raw <- read.csv("data/iedb_522_EL_epitopes.csv", stringsAsFactors = FALSE)

clean_names <- function(x) {
  x |>
    str_replace_all("\\.\\.\\.", "_") |>
    str_replace_all("\\.", "_") |>
    str_replace_all("-", "_") |>
    str_replace_all(" ", "_") |>
    str_to_lower()
}

colnames(df_raw) <- clean_names(colnames(df_raw))

df_clean <- df_raw |>
  select(where(~ !all(is.na(.)))) |>
  mutate(
    across(where(is.character), ~ na_if(.x, ""))
  ) |>
  mutate(
    pep_length = nchar(epitope_name),
    epitope_starting_position = as.integer(epitope_starting_position),
    epitope_ending_position = as.integer(epitope_ending_position)
  )

# === Create epitope dataframe ===
df_epitope <- df_clean |>
  select(
    epitope_name,
    epitope_starting_position,
    epitope_ending_position,
    epitope_molecule_parent,
    epitope_molecule_parent_iri,
    epitope_source_organism,
    pep_length
  ) |>
  distinct() |>
  rename(
    peptide = epitope_name,
    start = epitope_starting_position,
    end = epitope_ending_position,
    protein = epitope_molecule_parent,
    protein_iri = epitope_molecule_parent_iri,
    organism = epitope_source_organism
  ) |>
  mutate(
    uniprot_id = str_extract(protein_iri, "[^/]+$")
  )

# === Create additional info tables ===

# Protein info lookup
df_protein_info <- df_epitope |>
  select(protein, protein_iri, organism, uniprot_id) |>
  distinct(uniprot_id, .keep_all = TRUE)

# MHC restriction info (from df_clean)
df_mhc_info <- df_clean |>
  select(
    peptide = epitope_name,
    mhc_restriction_name,
    host_name,
    host_mhc_types_present
  ) |>
  distinct(peptide, .keep_all = TRUE)

# Assay info (from df_clean) - useful for quality assessment
df_assay_info <- df_clean |>
  select(
    peptide = epitope_name,
    assay_method,
    assay_qualitative_measurement,
    assay_response_frequency__
  ) |>
  distinct(peptide, .keep_all = TRUE)

# === Create positives with full info ===
df_pos <- df_epitope |>
  distinct(peptide, uniprot_id, .keep_all = TRUE) |>
  mutate(label = 1)

write_csv(df_pos, "data/pos_viral_proteins.csv")


# === Load FASTA ===
read_fasta_df <- function(filepath) {
  lines <- readLines(filepath)
  header_idx <- grep("^>", lines)
  headers <- lines[header_idx]
  sequences <- vector("character", length(header_idx))
  
  for (i in seq_along(header_idx)) {
    start <- header_idx[i] + 1
    end <- if (i < length(header_idx)) header_idx[i + 1] - 1 else length(lines)
    sequences[i] <- paste(lines[start:end], collapse = "")
  }
  
  tibble(
    header = headers,
    sequence = sequences
  )
}

df_fasta <- read_fasta_df("data/combined.fasta") |>
  mutate(
    uniprot_id = str_extract(header, "(?<=\\|)[A-Z0-9]+(?=\\|)")
  )

# Keep only proteins in positives
df_fasta <- df_fasta |>
  filter(uniprot_id %in% unique(df_pos$uniprot_id))

# ============================================================
# === GENERATE K-MERS (MATCHING POSITIVE LENGTH DISTRIBUTION) ===
# ============================================================

# Check what peptide lengths exist in positives
cat("=== Peptide Length Distribution in Positives ===\n")
print(table(df_pos$pep_length))

# Get unique lengths from positives
lengths_in_data <- sort(unique(df_pos$pep_length))
cat("Lengths to generate:", paste(lengths_in_data, collapse = ", "), "\n")

# Function to generate k-mers of specified lengths
generate_kmers <- function(seq, uid, lengths = 8:14) {
  seq_length <- nchar(seq)
  
  all_peptides <- list()
  
  for (k in lengths) {
    if (seq_length < k) next
    
    starts <- 1:(seq_length - k + 1)
    peptides <- vapply(
      starts,
      function(i) substr(seq, i, i + k - 1),
      character(1)
    )
    
    all_peptides[[as.character(k)]] <- tibble(
      peptide = peptides,
      start = starts,
      end = starts + k - 1,
      uniprot_id = uid,
      pep_length = as.integer(k)
    )
  }
  
  bind_rows(all_peptides)
}

# Generate all k-mers matching positive lengths
df_all_kmers <- df_fasta |>
  mutate(
    peptides = map2(sequence, uniprot_id, ~ generate_kmers(.x, .y, lengths = lengths_in_data))
  ) |>
  select(peptides) |>
  unnest(peptides)

cat("Total k-mers generated:", nrow(df_all_kmers), "\n")

# ============================================================
# === CLEAN POSITIVES — VERIFY IN FASTA ===
# ============================================================

df_pos_clean <- df_pos |>
  filter(
    map2_lgl(peptide, uniprot_id, function(pep, uid) {
      prot_seq <- df_fasta$sequence[df_fasta$uniprot_id == uid]
      if (length(prot_seq) == 0) return(FALSE)
      any(grepl(pep, prot_seq, fixed = TRUE))
    })
  ) |>
  distinct(peptide, uniprot_id, .keep_all = TRUE) |>
  mutate(label = 1)

cat("\nOriginal positives:", nrow(df_pos), "\n")
cat("Valid positives in FASTA:", nrow(df_pos_clean), "\n")
cat("Removed (not found in protein):", nrow(df_pos) - nrow(df_pos_clean), "\n")

cat("\nPositive length distribution after cleaning:\n")
print(table(df_pos_clean$pep_length))

# ============================================================
# === GENERATE NEGATIVES ===
# ============================================================

# Remove positives from all k-mers
df_neg_all <- df_all_kmers |>
  anti_join(
    df_pos_clean |> select(peptide, uniprot_id),
    by = c("peptide", "uniprot_id")
  ) |>
  mutate(label = 0)

cat("\nTotal potential negatives:", nrow(df_neg_all), "\n")

# Sample negatives per protein, stratified by length
set.seed(42)

df_neg_sampled <- df_neg_all |>
  group_by(uniprot_id, pep_length) |>
  group_modify(~ {
    uid <- .y$uniprot_id
    pep_len <- .y$pep_length
    
    # Count positives of this length for this protein
    n_pos <- sum(df_pos_clean$uniprot_id == uid & df_pos_clean$pep_length == pep_len)
    
    # If no positives of this exact length, use overall protein ratio
    if (n_pos == 0) {
      n_pos_protein <- sum(df_pos_clean$uniprot_id == uid)
      # Allocate proportionally
      n_pos <- max(1, round(n_pos_protein / length(lengths_in_data)))
    }
    
    n_neg <- min(10 * n_pos, nrow(.x))
    
    if (n_neg > 0 && nrow(.x) > 0) {
      slice_sample(.x, n = n_neg)
    } else {
      .x[0, ]
    }
  }) |>
  ungroup()

cat("Sampled negatives:", nrow(df_neg_sampled), "\n")

cat("\nNegative length distribution:\n")
print(table(df_neg_sampled$pep_length))

# ============================================================
# === COMBINE POSITIVES AND NEGATIVES ===
# ============================================================

df_model <- bind_rows(
  df_pos_clean |> 
    select(peptide, uniprot_id, pep_length) |>
    mutate(label = 1),
  df_neg_sampled |> 
    select(peptide, uniprot_id, pep_length, label)
)

cat("\n=== Combined Dataset ===\n")
cat("Total rows:", nrow(df_model), "\n")
cat("Positives:", sum(df_model$label == 1), "\n")
cat("Negatives:", sum(df_model$label == 0), "\n")

# ============================================================
# === ADD FLANKING REGIONS (USING CORRECT FUNCTION) ===
# ============================================================

cat("\n=== Adding Flanking Regions ===\n")

# NOW USE THE CORRECT FUNCTION!
df_model <- add_flanking_regions(df_model, df_fasta, flank_size = 8)

# Check results
cat("Peptides found in protein:", sum(df_model$peptide_found), "/", nrow(df_model), "\n")
cat("Peptides NOT found:", sum(!df_model$peptide_found), "\n")

# Remove rows where peptide wasn't found
if (sum(!df_model$peptide_found) > 0) {
  cat("Removing", sum(!df_model$peptide_found), "rows where peptide not found\n")
  df_model <- df_model |> filter(peptide_found)
}


# ============================================================
# === ADD PROTEIN AND ORGANISM INFO ===
# ============================================================

df_model <- df_model |>
  left_join(
    df_protein_info |> select(uniprot_id, protein, organism),
    by = "uniprot_id"
  )

# ============================================================
# === ADD MHC INFO ===
# ============================================================

df_model <- df_model |>
  left_join(
    df_mhc_info |> select(peptide, mhc_restriction_name),
    by = "peptide"
  )

# ============================================================
# === VALIDATION ===
# ============================================================

cat("\n=== VALIDATION CHECKS ===\n")

# Check 1: Full context length
cat("\n--- Check 1: Full Context Length ---\n")
df_model <- df_model |>
  mutate(
    expected_length = 8 + pep_length + 8,
    actual_length = nchar(full_context),
    length_ok = expected_length == actual_length
  )

cat("Correct lengths:", sum(df_model$length_ok), "/", nrow(df_model), "\n")

if (any(!df_model$length_ok)) {
  cat("WARNING: Some lengths incorrect!\n")
  df_model |>
    filter(!length_ok) |>
    select(peptide, pep_length, expected_length, actual_length) |>
    head(5) |>
    print()
}
df_model

# Check 2: Peptide in correct position
cat("\n--- Check 2: Peptide Position ---\n")
df_model <- df_model |>
  mutate(
    extracted_peptide = substr(full_context, 9, 9 + pep_length - 1),
    peptide_matches = peptide == extracted_peptide
  )

cat("Peptide at correct position:", sum(df_model$peptide_matches), "/", nrow(df_model), "\n")

if (any(!df_model$peptide_matches)) {
  cat("WARNING: Some peptides not at correct position!\n")
  df_model |>
    filter(!peptide_matches) |>
    select(peptide, extracted_peptide, full_context) |>
    head(5) |>
    print()
}

# Check 3: Sample verification
cat("\n--- Check 3: Sample Verification (5 positives) ---\n")

set.seed(123)
sample_check <- df_model |>
  filter(label == 1) |>
  slice_sample(n = min(5, sum(df_model$label == 1)))

for (i in 1:nrow(sample_check)) {
  row <- sample_check[i, ]
  cat(sprintf("\nSample %d:\n", i))
  cat(sprintf("  Peptide:      %s (length %d)\n", row$peptide, row$pep_length))
  cat(sprintf("  N-flank:      %s\n", row$n_flank))
  cat(sprintf("  C-flank:      %s\n", row$c_flank))
  cat(sprintf("  Full context: %s\n", row$full_context))
  cat(sprintf("  Position:     %d-%d\n", row$start, row$end))
  
  # Verify against protein
  prot_seq <- df_fasta$sequence[df_fasta$uniprot_id == row$uniprot_id]
  if (length(prot_seq) > 0) {
    extracted <- substr(prot_seq, row$start, row$end)
    cat(sprintf("  From protein: %s\n", extracted))
    cat(sprintf("  Match: %s\n", extracted == row$peptide))
  }
}

# Check 4: Length distribution comparison
cat("\n--- Check 4: Length Distribution ---\n")
cat("Positives:\n")
print(table(df_model$pep_length[df_model$label == 1]))
cat("\nNegatives:\n")
print(table(df_model$pep_length[df_model$label == 0]))

# ============================================================
# === FINAL SUMMARY ===
# ============================================================

cat("\n=== FINAL DATASET SUMMARY ===\n")
cat("Total rows:", nrow(df_model), "\n")
cat("Positives:", sum(df_model$label == 1), "\n")
cat("Negatives:", sum(df_model$label == 0), "\n")
cat("Ratio (neg:pos):", round(sum(df_model$label == 0) / sum(df_model$label == 1), 1), ":1\n")

cat("\nFull context length distribution:\n")
print(table(nchar(df_model$full_context)))

cat("\nColumns:\n")
print(colnames(df_model))

cat("\nMissing values:\n")
df_model |>
  summarise(
    across(c(peptide, uniprot_id, label, n_flank, c_flank, full_context),
           ~ sum(is.na(.)), .names = "na_{.col}")
  ) |>
  pivot_longer(everything(), names_to = "column", values_to = "n_missing") |>
  filter(n_missing > 0) |>
  print()

# ============================================================
# === CLEAN UP AND SAVE ===
# ============================================================

# Remove validation columns
df_model_final <- df_model |>
  select(
    peptide,
    uniprot_id,
    label,
    pep_length,
    start,
    end,
    n_flank,
    c_flank,
    full_context,
    protein,
    organism,
    mhc_restriction_name,
    peptide_found
  )

# Save
write_csv(df_model_final, "data/df_model_enriched.csv")
cat("\n=== Saved: data/df_model_enriched.csv ===\n")

# Training version (minimal columns)
df_model_training <- df_model_final |>
  select(
    peptide,
    uniprot_id,
    label,
    pep_length,
    n_flank,
    c_flank,
    full_context,
    organism,
    mhc_restriction_name
  )

write_csv(df_model_training, "data/df_model_training.csv")
cat("=== Saved: data/df_model_training.csv ===\n")