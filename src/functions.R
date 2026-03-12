library(tidyverse)

# Working directory setup
set_working_directory <- function() {
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
  
  cat("Working directory set to:", getwd(), "\n")
}

# ============================================================================
# PACKAGE MANAGEMENT
# ============================================================================

load_required_packages <- function(packages) {
  
  # Check and install CRAN packages
  # Separate Bioconductor packages
  bioc_packages <- c("Biostrings")
  cran_packages <- setdiff(packages, bioc_packages)
  
  # Install CRAN packages if missing
  for (pkg in cran_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat("Installing", pkg, "...\n")
      install.packages(pkg, dependencies = TRUE)
    }
  }
  
  # Install Bioconductor packages if missing
  for (pkg in intersect(packages, bioc_packages)) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat("Installing", pkg, "from Bioconductor...\n")
      if (!requireNamespace("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager")
      }
      BiocManager::install(pkg)
    }
  }
  
  # Load all packages
  for (pkg in packages) {
    library(pkg, character.only = TRUE)
  }
  
  cat("✓ All packages loaded\n")
}

# Clean column names
clean_names <- function(x) {
  x |>
    str_replace_all("\\.\\.\\.", "_") |>
    str_replace_all("\\.", "_") |>
    str_replace_all("-", "_") |>
    str_replace_all(" ", "_") |>
    str_to_lower()
}

# Read FASTA file into dataframe
read_fasta_df <- function(filepath) {
  if (!file.exists(filepath)) {
    stop("FASTA file not found: ", filepath)
  }
  
  lines     <- readLines(filepath)
  header_idx <- grep("^>", lines)
  headers   <- lines[header_idx]
  sequences <- vector("character", length(header_idx))
  
  for (i in seq_along(header_idx)) {
    start        <- header_idx[i] + 1
    end          <- if (i < length(header_idx)) header_idx[i + 1] - 1 else length(lines)
    sequences[i] <- paste(lines[start:end], collapse = "")
  }
  
  tibble(header = headers, sequence = sequences)
}

# Extract flanking regions around a peptide in a protein sequence
extract_flanks_safe <- function(peptide, protein_seq, flank_size = 8) {
  if (is.na(peptide) || is.na(protein_seq) || peptide == "" || protein_seq == "") {
    return(list(
      n_flank = NA_character_, c_flank = NA_character_,
      full_context = NA_character_, start = NA_integer_,
      end = NA_integer_, found = FALSE
    ))
  }
  
  match_pos <- regexpr(peptide, protein_seq, fixed = TRUE)
  
  if (match_pos == -1) {
    return(list(
      n_flank = NA_character_, c_flank = NA_character_,
      full_context = NA_character_, start = NA_integer_,
      end = NA_integer_, found = FALSE
    ))
  }
  
  pep_start   <- as.integer(match_pos)
  pep_length  <- nchar(peptide)
  pep_end     <- pep_start + pep_length - 1
  prot_length <- nchar(protein_seq)
  
  n_flank_start <- max(1, pep_start - flank_size)
  n_flank_end   <- pep_start - 1
  n_flank       <- if (n_flank_end >= n_flank_start) substr(protein_seq, n_flank_start, n_flank_end) else ""
  n_flank_padded <- paste0(strrep("X", flank_size - nchar(n_flank)), n_flank)
  
  c_flank_start <- pep_end + 1
  c_flank_end   <- min(prot_length, pep_end + flank_size)
  c_flank       <- if (c_flank_end >= c_flank_start) substr(protein_seq, c_flank_start, c_flank_end) else ""
  c_flank_padded <- paste0(c_flank, strrep("X", flank_size - nchar(c_flank)))
  
  list(
    n_flank      = n_flank_padded,
    c_flank      = c_flank_padded,
    full_context = paste0(n_flank_padded, peptide, c_flank_padded),
    start        = pep_start,
    end          = pep_end,
    found        = TRUE
  )
}

# Add flanking regions to a dataframe of peptides
add_flanking_regions <- function(df, df_fasta, flank_size = 8) {
  seq_lookup <- setNames(df_fasta$sequence, df_fasta$uniprot_id)
  
  df |>
    mutate(row_id = row_number()) |>
    rowwise() |>
    mutate(
      protein_seq = seq_lookup[uniprot_id],
      flank_data  = list(extract_flanks_safe(peptide, protein_seq, flank_size))
    ) |>
    ungroup() |>
    mutate(
      n_flank       = map_chr(flank_data, ~ .x$n_flank),
      c_flank       = map_chr(flank_data, ~ .x$c_flank),
      full_context  = map_chr(flank_data, ~ .x$full_context),
      start         = map_int(flank_data, ~ as.integer(.x$start)),
      end           = map_int(flank_data, ~ as.integer(.x$end)),
      peptide_found = map_lgl(flank_data, ~ .x$found)
    ) |>
    select(-protein_seq, -flank_data, -row_id)
}

# Generate all k-mers from a protein sequence
generate_kmers <- function(seq, uid, lengths = 8:14) {
  seq_length  <- nchar(seq)
  all_peptides <- list()
  
  for (k in lengths) {
    if (seq_length < k) next
    
    starts   <- 1:(seq_length - k + 1)
    peptides <- vapply(starts, function(i) substr(seq, i, i + k - 1), character(1))
    
    all_peptides[[as.character(k)]] <- tibble(
      peptide    = peptides,
      start      = starts,
      end        = starts + k - 1,
      uniprot_id = uid,
      pep_length = as.integer(k)
    )
  }
  
  bind_rows(all_peptides)
}

# Check if a peptide exists in a protein sequence
verify_peptide_in_protein <- function(peptide, uniprot_id, df_fasta) {
  prot_seq <- df_fasta$sequence[df_fasta$uniprot_id == uniprot_id]
  if (length(prot_seq) == 0) return(FALSE)
  any(grepl(peptide, prot_seq, fixed = TRUE))
}

# Parse NetMHCpan web text output
parse_netmhcpan_web_txt <- function(txt_file) {
  stopifnot(file.exists(txt_file))
  
  lines <- readLines(txt_file, warn = FALSE)
  
  data_lines <- lines[
    grepl("^\\s*\\d+\\s+", lines) &
      !grepl("^-{5,}", lines) &
      !grepl("^\\s*Pos\\s+MHC\\s+", lines)
  ]
  
  if (length(data_lines) == 0) {
    stop("No data lines found. Are you sure this is the NetMHCpan web .txt output?")
  }
  
  fields <- strsplit(trimws(data_lines), "\\s+")
  
  parse_row <- function(x) {
    score_idx    <- which(grepl("^\\d*\\.\\d+$", x) | grepl("^\\d+\\.\\d+$", x))
    id_candidates <- which(grepl("^(sp|tr)_.+", x) | grepl("^[A-Za-z0-9]+_[A-Za-z0-9]+", x))
    
    if (length(id_candidates) == 0) {
      if (length(score_idx) == 0) return(NULL)
      identity_idx <- score_idx[1] - 1
      score_idx    <- score_idx[1]
    } else {
      if (length(score_idx) == 0) return(NULL)
      score_first  <- score_idx[1]
      identity_idx <- max(id_candidates[id_candidates < score_first], na.rm = TRUE)
      score_idx    <- score_first
    }
    
    identity         <- x[identity_idx]
    rest             <- x[(identity_idx + 1):length(x)]
    nums             <- suppressWarnings(as.numeric(rest))
    numeric_positions <- which(!is.na(nums))
    
    score    <- if (length(numeric_positions) >= 1) nums[numeric_positions[1]] else NA_real_
    rank     <- if (length(numeric_positions) >= 2) nums[numeric_positions[2]] else NA_real_
    ba_score <- if (length(numeric_positions) >= 3) nums[numeric_positions[3]] else NA_real_
    ba_rank  <- if (length(numeric_positions) >= 4) nums[numeric_positions[4]] else NA_real_
    aff_nm   <- if (length(numeric_positions) >= 5) nums[numeric_positions[5]] else NA_real_
    
    bindlevel <- NA_character_
    if (length(numeric_positions) >= 5) {
      after_aff_idx <- numeric_positions[5] + 1
      if (after_aff_idx <= length(rest)) {
        bind_tokens <- rest[after_aff_idx:length(rest)]
        if (length(bind_tokens) > 0) bindlevel <- paste(bind_tokens, collapse = " ")
      }
    }
    
    data.frame(
      Pos = as.integer(x[1]), MHC = x[2], Peptide = x[3], Core = x[4],
      Identity = identity, Score = score, Rank = rank,
      BA_score = ba_score, BA_rank = ba_rank, Aff_nm = aff_nm,
      BindLevel = bindlevel, stringsAsFactors = FALSE
    )
  }
  
  rows <- lapply(fields, parse_row)
  rows <- rows[!vapply(rows, is.null, logical(1))]
  
  if (length(rows) == 0) {
    stop("Failed to parse any rows. Please check the input file format.")
  }
  
  df <- do.call(rbind, rows)
  df$Protein <- sub("^sp\\|([^|]+)\\|([^\\s]+)$", "sp_\\1_\\2", df$Identity)
  df$Protein <- sub("^tr\\|([^|]+)\\|([^\\s]+)$", "tr_\\1_\\2", df$Protein)
  df
}

# Export epitopes to file
export_epitopes <- function(data, file, separator = c("comma", "newline"),
                            unique_only = TRUE, sort_output = FALSE) {
  separator <- match.arg(separator)
  epitopes  <- data |> pull(epitope)
  
  if (unique_only)  epitopes <- unique(epitopes)
  if (sort_output)  epitopes <- sort(epitopes)
  
  if (separator == "comma") {
    write_lines(str_c(epitopes, collapse = ","), file)
  } else {
    write_lines(epitopes, file)
  }
  
  invisible(epitopes)
}

# Validate flanking regions
validate_flanking_regions <- function(df, df_fasta) {
  results <- list(
    total_rows         = nrow(df),
    peptides_found     = sum(df$peptide_found, na.rm = TRUE),
    peptides_not_found = sum(!df$peptide_found, na.rm = TRUE)
  )
  
  df <- df |>
    mutate(
      expected_length = 8 + pep_length + 8,
      actual_length   = nchar(full_context),
      length_ok       = expected_length == actual_length,
      extracted_peptide = substr(full_context, 9, 9 + pep_length - 1),
      peptide_matches = peptide == extracted_peptide
    )
  
  results$correct_lengths          <- sum(df$length_ok, na.rm = TRUE)
  results$incorrect_lengths        <- sum(!df$length_ok, na.rm = TRUE)
  results$peptide_position_correct <- sum(df$peptide_matches, na.rm = TRUE)
  results$peptide_position_incorrect <- sum(!df$peptide_matches, na.rm = TRUE)
  
  cat("\n=== Flanking Region Validation ===\n")
  cat("Total rows:                 ", results$total_rows, "\n")
  cat("Peptides found:             ", results$peptides_found, "\n")
  cat("Peptides not found:         ", results$peptides_not_found, "\n")
  cat("Correct context lengths:    ", results$correct_lengths, "\n")
  cat("Incorrect context lengths:  ", results$incorrect_lengths, "\n")
  cat("Peptide at correct position:", results$peptide_position_correct, "\n")
  cat("Peptide at wrong position:  ", results$peptide_position_incorrect, "\n")
  
  invisible(results)
}

# ============================================================================
# VALIDATE AND FIX PEPTIDE POSITIONS
# ============================================================================

validate_and_fix_positions <- function(data,
                                       sequence_col = "sequence",
                                       peptide_col = "peptide",
                                       start_col = "start",
                                       end_col = "end") {
  #' Validate peptide positions and attempt to fix mismatches
  #' 
  #' @description
  #' This function checks if the peptide at the given start/end position
  #' matches the reported peptide. If not, it searches for the peptide
  #' in the protein sequence and corrects the position.
  #' 
  #' @param data Data frame with merged epitope and protein data
  #' @param sequence_col Column name for protein sequence
  #' @param peptide_col Column name for peptide sequence
  #' @param start_col Column name for start position
  #' @param end_col Column name for end position
  #' 
  #' @return Data frame with validated/fixed positions and status columns
  
  data %>%
    mutate(
      # Store original positions for reference
      start_original = .data[[start_col]],
      end_original = .data[[end_col]],
      
      # Get protein length
      protein_length = nchar(.data[[sequence_col]]),
      
      # Extract peptide at reported position
      extracted_at_position = if_else(
        !is.na(.data[[sequence_col]]) & 
          .data[[start_col]] >= 1 & 
          .data[[end_col]] <= protein_length,
        substr(.data[[sequence_col]], .data[[start_col]], .data[[end_col]]),
        NA_character_
      ),
      
      # Check if position is valid
      position_originally_valid = (.data[[peptide_col]] == extracted_at_position),
      
      # Try to find the peptide in the protein sequence
      found_start = if_else(
        !is.na(.data[[sequence_col]]),
        as.integer(str_locate(.data[[sequence_col]], fixed(.data[[peptide_col]]))[, 1]),
        NA_integer_
      ),
      
      # Determine position status
      position_status = case_when(
        is.na(.data[[sequence_col]]) ~ "no_protein_sequence",
        position_originally_valid ~ "valid_original",
        !position_originally_valid & !is.na(found_start) ~ "fixed",
        !position_originally_valid & is.na(found_start) ~ "unfixable"
      ),
      
      # Set corrected positions
      start_corrected = case_when(
        position_status == "valid_original" ~ as.integer(.data[[start_col]]),
        position_status == "fixed" ~ found_start,
        TRUE ~ NA_integer_
      ),
      
      end_corrected = case_when(
        position_status %in% c("valid_original", "fixed") ~ 
          start_corrected + nchar(.data[[peptide_col]]) - 1L,
        TRUE ~ NA_integer_
      ),
      
      # Calculate how much the position shifted (for investigation)
      position_shift = if_else(
        position_status == "fixed",
        start_corrected - as.integer(.data[[start_col]]),
        NA_integer_
      )
    ) %>%
    # Clean up: use corrected positions as main positions
    mutate(
      start = start_corrected,
      end = end_corrected
    ) %>%
    # Remove temporary columns (keep tracking columns)
    select(-extracted_at_position, -found_start, -start_corrected, -end_corrected)
}

# ============================================================================
# EXTRACT FLANKING REGIONS
# ============================================================================

extract_flanking_regions <- function(data, 
                                     sequence_col = "sequence",
                                     peptide_col = "peptide",
                                     start_col = "start",
                                     end_col = "end",
                                     n_flank_size = 10,
                                     c_flank_size = 10) {
  #' Extract N-terminal and C-terminal flanking regions around epitopes
  #' 
  #' description
  #' Extracts the amino acid sequences flanking each epitope. These regions
  #' are critical for predicting proteasome cleavage and TAP transport.
  #' 
  #' If the epitope is near the protein terminus, the flanking region is
  #' padded with 'X' characters to maintain consistent length.
  #' 
  #' param data Data frame with epitopes and protein sequences
  #' param n_flank_size Number of residues to extract BEFORE epitope (default: 10)
  #' param c_flank_size Number of residues to extract AFTER epitope (default: 10)
  #' 
  #' return Data frame with added columns:
  #'   - n_flank_seq: N-terminal flanking sequence
  #'   - c_flank_seq: C-terminal flanking sequence
  #'   - full_context: n_flank + peptide + c_flank
  #'   - near_n_terminus: Boolean, TRUE if epitope is near protein start
  #'   - near_c_terminus: Boolean, TRUE if epitope is near protein end
  
  data %>%
    mutate(
      # Ensure we have protein length
      protein_length = nchar(.data[[sequence_col]]),
      
      # === N-TERMINAL FLANKING REGION ===
      # This is the sequence BEFORE the epitope (upstream)
      
      # Calculate extraction boundaries
      n_flank_actual_start = pmax(1, .data[[start_col]] - n_flank_size),
      n_flank_actual_end = .data[[start_col]] - 1,
      
      # Extract the sequence
      n_flank = if_else(
        n_flank_actual_end >= 1,
        substr(.data[[sequence_col]], n_flank_actual_start, n_flank_actual_end),
        ""
      ),
      
      # === C-TERMINAL FLANKING REGION ===
      # This is the sequence AFTER the epitope (downstream)
      
      # Calculate extraction boundaries
      c_flank_actual_start = .data[[end_col]] + 1,
      c_flank_actual_end = pmin(protein_length, .data[[end_col]] + c_flank_size),
      
      # Extract the sequence
      c_flank = if_else(
        c_flank_actual_start <= protein_length,
        substr(.data[[sequence_col]], c_flank_actual_start, c_flank_actual_end),
        ""
      ),
      
      # === COMBINED CONTEXT ===
      full_context = paste0(n_flank, .data[[peptide_col]], c_flank),
      
      # Distance from termini (useful features)
      distance_from_n_terminus = .data[[start_col]] - 1,
      distance_from_c_terminus = protein_length - .data[[end_col]]
    ) %>%
    # Remove temporary calculation columns
    select(
      -n_flank_actual_start, -n_flank_actual_end, 
      -c_flank_actual_start, -c_flank_actual_end
    )
}

# ============================================================================
# EXTRACT CLEAVAGE SITE POSITIONS
# ============================================================================

extract_cleavage_positions <- function(data,
                                       peptide_col = "peptide",
                                       n_flank_col = "n_flank",
                                       c_flank_col = "c_flank") {
  #' Extract P1 and P1' positions at cleavage sites
  #'
  #' P1 = residue immediately BEFORE cleavage (most important for proteasome)
  #' P1' = residue immediately AFTER cleavage
  #'
  #' C-terminal P1 (last residue of epitope) is the most biologically relevant
  #' as it sits in the proteasome's S1 catalytic pocket.
  
  data %>%
    mutate(
      # C-terminal cleavage site (most important)
      # P1 = last residue of epitope
      c_term_P1 = substr(.data[[peptide_col]], nchar(.data[[peptide_col]]), nchar(.data[[peptide_col]])),
      # P1' = first residue of C-flank
      c_term_P1_prime = substr(.data[[c_flank_col]], 1, 1),
      
      # N-terminal cleavage site
      # P1 = last residue of N-flank
      n_term_P1 = substr(.data[[n_flank_col]], nchar(.data[[n_flank_col]]), nchar(.data[[n_flank_col]])),
      # P1' = first residue of epitope
      n_term_P1_prime = substr(.data[[peptide_col]], 1, 1)
    )
}

# ============================================================================
# AMINO ACID PROPERTY DICTIONARIES
# ============================================================================

# These dictionaries map each amino acid to its property value
# Used for calculating features across sequences

get_aa_properties <- function() {
  #' Returns a list of amino acid property dictionaries
  #' 
  #' These are standard scales from the literature used in
  #' bioinformatics and structural biology.
  
  list(
    # Kyte-Doolittle hydrophobicity scale (most commonly used)
    # Positive = hydrophobic, Negative = hydrophilic
    hydrophobicity = c(
      A =  1.8, C =  2.5, D = -3.5, E = -3.5, F =  2.8,
      G = -0.4, H = -3.2, I =  4.5, K = -3.9, L =  3.8,
      M =  1.9, N = -3.5, P = -1.6, Q = -3.5, R = -4.5,
      S = -0.8, T = -0.7, V =  4.2, W = -0.9, Y = -1.3,
      X =  0.0  # Unknown/padding
    ),
    
    # Charge at physiological pH
    charge = c(
      A =  0, C =  0, D = -1, E = -1, F =  0,
      G =  0, H =  0, I =  0, K =  1, L =  0,
      M =  0, N =  0, P =  0, Q =  0, R =  1,
      S =  0, T =  0, V =  0, W =  0, Y =  0,
      X =  0
    ),
    
    # Molecular volume (Å³)
    volume = c(
      A =  88.6, C = 108.5, D = 111.1, E = 138.4, F = 189.9,
      G =  60.1, H = 153.2, I = 166.7, K = 168.6, L = 166.7,
      M = 162.9, N = 114.1, P = 112.7, Q = 143.8, R = 173.4,
      S =  89.0, T = 116.1, V = 140.0, W = 227.8, Y = 193.6,
      X = 120.0  # Average
    ),
    
    # Molecular weight (Da)
    molecular_weight = c(
      A =  89.1, C = 121.2, D = 133.1, E = 147.1, F = 165.2,
      G =  75.1, H = 155.2, I = 131.2, K = 146.2, L = 131.2,
      M = 149.2, N = 132.1, P = 115.1, Q = 146.2, R = 174.2,
      S = 105.1, T = 119.1, V = 117.1, W = 204.2, Y = 181.2,
      X = 120.0  # Average
    ),
    
    # Polarity (Grantham)
    polarity = c(
      A =  8.1, C =  5.5, D = 13.0, E = 12.3, F =  5.2,
      G =  9.0, H = 10.4, I =  5.2, K = 11.3, L =  4.9,
      M =  5.7, N = 11.6, P =  8.0, Q = 10.5, R = 10.5,
      S =  9.2, T =  8.6, V =  5.9, W =  5.4, Y =  6.2,
      X =  8.0  # Average
    ),
    
    # Flexibility (Bhaskaran & Ponnuswamy)
    flexibility = c(
      A = 0.360, C = 0.350, D = 0.510, E = 0.500, F = 0.310,
      G = 0.540, H = 0.320, I = 0.460, K = 0.470, L = 0.370,
      M = 0.300, N = 0.460, P = 0.510, Q = 0.490, R = 0.530,
      S = 0.510, T = 0.440, V = 0.390, W = 0.310, Y = 0.420,
      X = 0.420  # Average
    )
  )
}

# Amino acid category memberships (for boolean features)
get_aa_categories <- function() {
  list(
    hydrophobic = c("A", "I", "L", "M", "F", "V", "W", "Y"),
    polar = c("S", "T", "N", "Q", "C", "Y"),
    positive = c("K", "R", "H"),
    negative = c("D", "E"),
    aromatic = c("F", "W", "Y", "H"),
    small = c("A", "G", "S", "C", "T"),
    aliphatic = c("A", "I", "L", "V")
  )
}


# ============================================================================
# CALCULATE PHYSICOCHEMICAL PROPERTIES FOR A SEQUENCE
# ============================================================================

calculate_sequence_properties <- function(sequence, prefix = "") {
  #' Calculate physicochemical properties for an amino acid sequence
  #' 
  #' param sequence Character string of amino acids
  #' param prefix String to prepend to column names (e.g., "peptide_", "n_flank_")
  #' 
  #' return A single-row tibble with calculated properties
  
  # Handle NA or empty sequences
  if (is.na(sequence) || sequence == "") {
    return(tibble(
      !!paste0(prefix, "hydrophobicity_mean") := NA_real_,
      !!paste0(prefix, "hydrophobicity_sum") := NA_real_,
      !!paste0(prefix, "charge_total") := NA_real_,
      !!paste0(prefix, "charge_positive_count") := NA_integer_,
      !!paste0(prefix, "charge_negative_count") := NA_integer_,
      !!paste0(prefix, "volume_mean") := NA_real_,
      !!paste0(prefix, "polarity_mean") := NA_real_,
      !!paste0(prefix, "flexibility_mean") := NA_real_,
      !!paste0(prefix, "molecular_weight") := NA_real_,
      !!paste0(prefix, "aromaticity_fraction") := NA_real_,
      !!paste0(prefix, "hydrophobic_fraction") := NA_real_
    ))
  }
  
  # Get property dictionaries
  props <- get_aa_properties()
  cats <- get_aa_categories()
  
  # Split sequence into individual amino acids
  aa_list <- strsplit(sequence, "")[[1]]
  n <- length(aa_list)
  
  # Exclude X (padding) from calculations
  aa_list_no_x <- aa_list[aa_list != "X"]
  n_real <- length(aa_list_no_x)
  
  if (n_real == 0) {
    n_real <- 1  # Prevent division by zero
  }
  
  # Calculate properties
  hydro_values <- sapply(aa_list, function(x) props$hydrophobicity[x])
  charge_values <- sapply(aa_list, function(x) props$charge[x])
  volume_values <- sapply(aa_list, function(x) props$volume[x])
  polarity_values <- sapply(aa_list, function(x) props$polarity[x])
  flex_values <- sapply(aa_list, function(x) props$flexibility[x])
  mw_values <- sapply(aa_list, function(x) props$molecular_weight[x])
  
  tibble(
    # Hydrophobicity
    !!paste0(prefix, "hydrophobicity_mean") := mean(hydro_values, na.rm = TRUE),
    !!paste0(prefix, "hydrophobicity_sum") := sum(hydro_values, na.rm = TRUE),
    
    # Charge
    !!paste0(prefix, "charge_total") := sum(charge_values, na.rm = TRUE),
    !!paste0(prefix, "charge_positive_count") := sum(aa_list_no_x %in% cats$positive),
    !!paste0(prefix, "charge_negative_count") := sum(aa_list_no_x %in% cats$negative),
    
    # Size
    !!paste0(prefix, "volume_mean") := mean(volume_values, na.rm = TRUE),
    
    # Polarity and flexibility
    !!paste0(prefix, "polarity_mean") := mean(polarity_values, na.rm = TRUE),
    !!paste0(prefix, "flexibility_mean") := mean(flex_values, na.rm = TRUE),
    
    # Molecular weight (sum minus water loss from peptide bonds)
    !!paste0(prefix, "molecular_weight") := sum(mw_values, na.rm = TRUE) - (n_real - 1) * 18.015,
    
    # Categorical fractions
    !!paste0(prefix, "aromaticity_fraction") := sum(aa_list_no_x %in% cats$aromatic) / n_real,
    !!paste0(prefix, "hydrophobic_fraction") := sum(aa_list_no_x %in% cats$hydrophobic) / n_real
  )
}


# ============================================================================
# ADD PHYSICOCHEMICAL PROPERTIES TO DATA FRAME
# ============================================================================

add_physicochemical_properties <- function(data,
                                           peptide_col = "peptide",
                                           n_flank_col = "n_flank_seq",
                                           c_flank_col = "c_flank_seq") {
  #' Calculate physicochemical properties for peptide and flanking regions
  #' 
  #' description
  #' Calculates multiple physicochemical properties for:
  #' 1. The epitope/peptide itself
  #' 2. The N-terminal flanking region
  #' 3. The C-terminal flanking region
  #' 
  #' These properties are relevant for predicting:
  #' - Proteasome cleavage efficiency
  #' - TAP transport
  #' - Overall antigen processing
  #' 
  #' param data Data frame with peptide and flanking columns
  #' 
  #' return Data frame with added physicochemical property columns
  
  # Calculate for peptide
  cat("  Calculating peptide properties...\n")
  peptide_props <- map_dfr(data[[peptide_col]], ~calculate_sequence_properties(.x, "peptide_"))
  
  # Calculate for N-flank
  cat("  Calculating N-flank properties...\n")
  n_flank_props <- map_dfr(data[[n_flank_col]], ~calculate_sequence_properties(.x, "n_flank_"))
  
  # Calculate for C-flank
  cat("  Calculating C-flank properties...\n")
  c_flank_props <- map_dfr(data[[c_flank_col]], ~calculate_sequence_properties(.x, "c_flank_"))
  
  # Combine with original data
  bind_cols(data, peptide_props, n_flank_props, c_flank_props)
}

# ============================================================================
# PROTEASOME PREFERENCE SCORES
# ============================================================================

get_proteasome_preferences <- function() {
  #' P1 position preferences for standard and immunoproteasome
  #' 
  #' These scores represent how likely each amino acid is to be at the
  #' P1 position (C-terminal residue) of a proteasome cleavage product.
  #' 
  #' Scores range from 0 (disfavored) to 1 (strongly favored).
  #' Based on literature analysis of proteasome cleavage specificity.
  
  list(
    # Immunoproteasome P1 preferences
    # Enhanced after IFN-γ stimulation (viral infection context)
    # Favors: hydrophobic (L, F, Y), basic (K, R)
    immunoproteasome = c(
      A = 0.3, C = 0.2, D = 0.1, E = 0.1, F = 0.9,
      G = 0.2, H = 0.4, I = 0.6, K = 0.7, L = 0.9,
      M = 0.5, N = 0.2, P = 0.1, Q = 0.3, R = 0.7,
      S = 0.3, T = 0.3, V = 0.5, W = 0.7, Y = 0.8,
      X = 0.3
    ),
    
    # Standard (constitutive) proteasome P1 preferences
    # Normal cellular conditions
    # More tolerant of acidic residues
    standard = c(
      A = 0.4, C = 0.3, D = 0.5, E = 0.5, F = 0.7,
      G = 0.2, H = 0.3, I = 0.5, K = 0.4, L = 0.8,
      M = 0.5, N = 0.3, P = 0.1, Q = 0.4, R = 0.4,
      S = 0.3, T = 0.3, V = 0.5, W = 0.6, Y = 0.7,
      X = 0.3
    )
  )
}


# ============================================================================
# ADD PROTEASOME FEATURES
# ============================================================================

add_proteasome_features <- function(data) {
  #' Add proteasome cleavage-related features
  #' 
  #' description
  #' Calculates features relevant to proteasome cleavage prediction:
  #' 
  #' 1. P1 residue properties (C-terminal of epitope - most important!)
  #' 2. Proteasome preference scores (immunoproteasome vs standard)
  #' 3. Cleavage inhibitor flags (e.g., proline at P1')
  #' 
  #' The C-terminal P1 residue is the most critical determinant of
  #' proteasome cleavage efficiency. Different proteasome types have
  #' different preferences at this position.
  
  prefs <- get_proteasome_preferences()
  cats <- get_aa_categories()
  
  data %>%
    mutate(
      # ================================================================
      # C-TERMINAL P1 FEATURES (MOST IMPORTANT)
      # ================================================================
      # The C-terminal P1 is the last residue of the epitope
      # It sits in the proteasome's S1 catalytic pocket
      
      # Residue category flags
      c_term_P1_is_hydrophobic = c_cleavage_P1 %in% cats$hydrophobic,
      c_term_P1_is_basic = c_cleavage_P1 %in% cats$positive,
      c_term_P1_is_acidic = c_cleavage_P1 %in% cats$negative,
      c_term_P1_is_aromatic = c_cleavage_P1 %in% cats$aromatic,
      c_term_P1_is_small = c_cleavage_P1 %in% cats$small,
      
      # Proteasome preference scores
      a = sapply(c_cleavage_P1, function(x) prefs$immunoproteasome[x]),
      c_term_P1_standard_score = sapply(c_cleavage_P1, function(x) prefs$standard[x]),
      
      # ================================================================
      # C-TERMINAL P1' FEATURES
      # ================================================================
      # Proline at P1' strongly inhibits cleavage!
      c_term_P1_prime_is_proline = (c_cleavage_P1_prime == "P"),
      
      # ================================================================
      # N-TERMINAL P1 FEATURES
      # ================================================================
      n_term_P1_immuno_score = sapply(n_cleavage_P1, function(x) prefs$immunoproteasome[x]),
      n_term_P1_standard_score = sapply(n_cleavage_P1, function(x) prefs$standard[x]),
      n_term_P1_prime_is_proline = (n_cleavage_P1_prime == "P"),
      
      # ================================================================
      # COMBINED SCORES
      # ================================================================
      # Average of N-terminal and C-terminal cleavage scores
      combined_immuno_score = (c_term_P1_immuno_score + n_term_P1_immuno_score) / 2,
      combined_standard_score = (c_term_P1_standard_score + n_term_P1_standard_score) / 2,
      
      # Difference between proteasome types
      # Positive = favors immunoproteasome, Negative = favors standard
      proteasome_preference_diff = combined_immuno_score - combined_standard_score
    )
}

# ============================================================================
# TAP TRANSPORT PREFERENCE SCORES
# ============================================================================

get_tap_preferences <- function() {
  #' TAP transport preference scores by position
  #' 
  #' TAP transports peptides from cytosol to ER for MHC loading.
  #' The C-terminal residue is most important for TAP binding.
  
  list(
    # C-terminal residue preferences (most important)
    c_terminal = c(
      A = 0.3, C = 0.3, D = 0.1, E = 0.1, F = 0.9,
      G = 0.2, H = 0.4, I = 0.7, K = 0.8, L = 0.9,
      M = 0.6, N = 0.3, P = 0.1, Q = 0.3, R = 0.8,
      S = 0.3, T = 0.3, V = 0.6, W = 0.7, Y = 0.8,
      X = 0.3
    ),
    
    # Position 2 (from N-terminus) preferences
    position_2 = c(
      A = 0.4, C = 0.3, D = 0.3, E = 0.3, F = 0.5,
      G = 0.3, H = 0.4, I = 0.5, K = 0.6, L = 0.5,
      M = 0.4, N = 0.5, P = 0.2, Q = 0.6, R = 0.7,
      S = 0.4, T = 0.4, V = 0.5, W = 0.5, Y = 0.6,
      X = 0.4
    ),
    
    # Position 3 preferences
    position_3 = c(
      A = 0.4, C = 0.3, D = 0.3, E = 0.3, F = 0.6,
      G = 0.3, H = 0.4, I = 0.5, K = 0.5, L = 0.5,
      M = 0.4, N = 0.4, P = 0.3, Q = 0.4, R = 0.5,
      S = 0.4, T = 0.4, V = 0.5, W = 0.8, Y = 0.7,
      X = 0.4
    )
  )
}


# ============================================================================
# ADD TAP TRANSPORT FEATURES
# ============================================================================

add_tap_features <- function(data, peptide_col = "peptide") {
  #' Add TAP transport-related features
  #' 
  #' description
  #' TAP (Transporter Associated with Antigen Processing) transports
  #' peptides from the cytosol to the ER for MHC loading.
  #' 
  #' TAP has specific preferences that affect transport efficiency:
  #' - C-terminal residue is most important
  #' - Peptide length affects transport
  #' - Certain N-terminal positions matter
  
  tap_prefs <- get_tap_preferences()
  cats <- get_aa_categories()
  
  data %>%
    mutate(
      # ================================================================
      # C-TERMINAL FEATURES FOR TAP
      # ================================================================
      tap_c_term_residue = substr(.data[[peptide_col]], nchar(.data[[peptide_col]]), nchar(.data[[peptide_col]])),
      
      tap_c_term_score = sapply(tap_c_term_residue, function(x) tap_prefs$c_terminal[x]),
      tap_c_term_favorable = tap_c_term_residue %in% c("L", "F", "Y", "I", "M", "K", "R"),
      tap_c_term_unfavorable = tap_c_term_residue %in% c("D", "E", "P", "G"),
      
      # ================================================================
      # N-TERMINAL FEATURES FOR TAP
      # ================================================================
      tap_pos1_residue = substr(.data[[peptide_col]], 1, 1),
      tap_pos1_is_proline = (tap_pos1_residue == "P"),  # Bad for TAP
      
      tap_pos2_residue = substr(.data[[peptide_col]], 2, 2),
      tap_pos2_score = sapply(tap_pos2_residue, function(x) tap_prefs$position_2[x]),
      
      tap_pos3_residue = substr(.data[[peptide_col]], 3, 3),
      tap_pos3_score = sapply(tap_pos3_residue, function(x) tap_prefs$position_3[x]),
      
      # ================================================================
      # LENGTH FEATURES FOR TAP
      # ================================================================
      tap_length = nchar(.data[[peptide_col]]),
      tap_length_optimal = (tap_length >= 8 & tap_length <= 12),
      tap_length_suboptimal = (tap_length > 12 & tap_length <= 16),
      
      # ================================================================
      # COMBINED TAP SCORE
      # ================================================================
      tap_combined_score = (tap_c_term_score + tap_pos2_score + tap_pos3_score) / 3
    ) %>%
    # Remove temporary residue columns (keep scores and flags)
    select(-tap_c_term_residue, -tap_pos1_residue, -tap_pos2_residue, -tap_pos3_residue)
}

cat("functions.R loaded successfully.\n")