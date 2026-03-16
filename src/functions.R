library(tidyverse)

# Working directory setup
set_working_directory <- function() {
  current_user <- Sys.info()[["user"]]
  
  if (current_user == "olive") {
    setwd("C:/Users/olive/Documents/R/special_course_spring2026")
  } else if (current_user == "mj607") {
    setwd("//wsl$/Ubuntu/home/hilarius/special_course_spring2026")
  } else if (current_user == "hilarius") {
    setwd("/home/hilarius/special_course/special_course_spring2026")
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

get_aa_properties <- function() {
  list(
    hydrophobicity = c(
      A =  1.8, C =  2.5, D = -3.5, E = -3.5, F =  2.8,
      G = -0.4, H = -3.2, I =  4.5, K = -3.9, L =  3.8,
      M =  1.9, N = -3.5, P = -1.6, Q = -3.5, R = -4.5,
      S = -0.8, T = -0.7, V =  4.2, W = -0.9, Y = -1.3
    ),
    charge = c(
      A =  0, C =  0, D = -1, E = -1, F =  0,
      G =  0, H =  0, I =  0, K =  1, L =  0,
      M =  0, N =  0, P =  0, Q =  0, R =  1,
      S =  0, T =  0, V =  0, W =  0, Y =  0
    ),
    molecular_weight = c(
      A =  89.1, C = 121.2, D = 133.1, E = 147.1, F = 165.2,
      G =  75.1, H = 155.2, I = 131.2, K = 146.2, L = 131.2,
      M = 149.2, N = 132.1, P = 115.1, Q = 146.2, R = 174.2,
      S = 105.1, T = 119.1, V = 117.1, W = 204.2, Y = 181.2
    )
  )
}

get_aa_categories <- function() {
  list(
    hydrophobic = c("A", "I", "L", "M", "F", "V", "W", "Y"),
    aromatic = c("F", "W", "Y", "H"),
    positive = c("K", "R", "H"),
    negative = c("D", "E"),
    polar = c("S", "T", "N", "Q", "C", "Y"),
    small = c("A", "G", "S", "C", "T")
  )
}
# ============================================================================
# CALCULATE PHYSICOCHEMICAL PROPERTIES FOR A SINGLE SEQUENCE
# ============================================================================

calculate_sequence_properties <- function(sequence, prefix = "") {
  #' Calculate physicochemical properties for an amino acid sequence
  #'
  #' Always returns exactly 1 row with consistent column names
  
  props <- get_aa_properties()
  cats <- get_aa_categories()
  
  # Define column names (must be consistent for all cases)
  col_hydro <- paste0(prefix, "_hydrophobicity_mean")
  col_charge <- paste0(prefix, "_charge_total")
  col_mw <- paste0(prefix, "_molecular_weight")
  col_arom <- paste0(prefix, "_aromaticity_frac")
  col_hydrophobic <- paste0(prefix, "_hydrophobic_frac")
  
  # Handle NA, NULL, or empty sequences
  if (is.null(sequence) || length(sequence) == 0 || is.na(sequence) || 
      !is.character(sequence) || sequence == "" || nchar(sequence) == 0) {
    result <- tibble(
      placeholder1 = NA_real_,
      placeholder2 = NA_real_,
      placeholder3 = NA_real_,
      placeholder4 = NA_real_,
      placeholder5 = NA_real_
    )
    names(result) <- c(col_hydro, col_charge, col_mw, col_arom, col_hydrophobic)
    return(result)
  }
  
  # Split sequence into amino acids
  aa_list <- strsplit(as.character(sequence), "")[[1]]
  n <- length(aa_list)
  
  # Handle zero-length after split
  if (n == 0) {
    result <- tibble(
      placeholder1 = NA_real_,
      placeholder2 = NA_real_,
      placeholder3 = NA_real_,
      placeholder4 = NA_real_,
      placeholder5 = NA_real_
    )
    names(result) <- c(col_hydro, col_charge, col_mw, col_arom, col_hydrophobic)
    return(result)
  }
  
  # Calculate properties
  hydro_values <- sapply(aa_list, function(x) {
    if (x %in% names(props$hydrophobicity)) props$hydrophobicity[x] else NA_real_
  })
  
  charge_values <- sapply(aa_list, function(x) {
    if (x %in% names(props$charge)) props$charge[x] else 0
  })
  
  mw_values <- sapply(aa_list, function(x) {
    if (x %in% names(props$molecular_weight)) props$molecular_weight[x] else NA_real_
  })
  
  # Build result with consistent column names
  result <- tibble(
    placeholder1 = mean(hydro_values, na.rm = TRUE),
    placeholder2 = sum(charge_values, na.rm = TRUE),
    placeholder3 = sum(mw_values, na.rm = TRUE) - max(0, (n - 1)) * 18.015,
    placeholder4 = sum(aa_list %in% cats$aromatic) / n,
    placeholder5 = sum(aa_list %in% cats$hydrophobic) / n
  )
  names(result) <- c(col_hydro, col_charge, col_mw, col_arom, col_hydrophobic)
  
  return(result)
}
# ============================================================================
# ADD PHYSICOCHEMICAL PROPERTIES TO DATA FRAME
# ============================================================================

add_physicochemical_properties <- function(data,
                                           peptide_col = "peptide",
                                           n_flank_col = "n_flank",
                                           c_flank_col = "c_flank") {
  #' Add physicochemical properties for peptide and flanking regions
  #'
  #' Calculates hydrophobicity, charge, molecular weight, and composition
  #' for the peptide and both flanking regions.
  
  # Verify columns exist
  for (col in c(peptide_col, n_flank_col, c_flank_col)) {
    if (!col %in% names(data)) {
      stop("Column '", col, "' not found in data")
    }
  }
  
  n_rows <- nrow(data)
  cat("  Processing", n_rows, "rows...\n")
  
  # Calculate for peptide
  cat("  Calculating peptide properties...")
  peptide_props <- map_dfr(seq_len(n_rows), function(i) {
    calculate_sequence_properties(data[[peptide_col]][i], "peptide")
  })
  cat(" done (", nrow(peptide_props), " rows)\n", sep = "")
  
  # Calculate for N-flank  
  cat("  Calculating N-flank properties...")
  n_flank_props <- map_dfr(seq_len(n_rows), function(i) {
    calculate_sequence_properties(data[[n_flank_col]][i], "n_flank")
  })
  cat(" done (", nrow(n_flank_props), " rows)\n", sep = "")
  
  # Calculate for C-flank
  cat("  Calculating C-flank properties...")
  c_flank_props <- map_dfr(seq_len(n_rows), function(i) {
    calculate_sequence_properties(data[[c_flank_col]][i], "c_flank")
  })
  cat(" done (", nrow(c_flank_props), " rows)\n", sep = "")
  
  # Verify row counts
  if (nrow(peptide_props) != n_rows || 
      nrow(n_flank_props) != n_rows || 
      nrow(c_flank_props) != n_rows) {
    stop("Row count mismatch: peptide=", nrow(peptide_props),
         ", n_flank=", nrow(n_flank_props),
         ", c_flank=", nrow(c_flank_props),
         ", expected=", n_rows)
  }
  
  # Combine all
  bind_cols(data, peptide_props, n_flank_props, c_flank_props)
}
# ============================================================================
# PROTEASOME PREFERENCE SCORES (Literature-Based)
# ============================================================================

get_proteasome_preferences <- function() {
  #' P1 position cleavage preferences for standard and immunoproteasome
  #'
  #' Based on: Toes et al., 2001, J Exp Med; Tenzer et al., 2005, Nature Immunology
  #'
  #' Values are relative cleavage frequencies normalized to average = 1.0
  #' Higher = more likely to be cleaved AFTER this residue
  
  list(
    # Immunoproteasome (IFN-γ induced, viral infection context)
    # Enhanced: hydrophobic (L, F, Y), basic (K, R)
    # Reduced: acidic (D, E)
    immunoproteasome = c(
      L = 2.3, F = 2.2, Y = 2.0, W = 1.5, I = 1.1,
      V = 1.1, M = 1.1, K = 1.0, R = 0.9, A = 0.6,
      H = 0.6, T = 0.6, E = 0.6, Q = 0.7, S = 0.5,
      D = 0.5, N = 0.5, C = 0.4, G = 0.3, P = 0.2
    ),
    
    # Constitutive (standard) proteasome
    # More balanced, tolerates acidic residues better
    constitutive = c(
      L = 1.9, F = 1.8, Y = 1.5, E = 1.3, D = 1.2,
      W = 1.2, I = 1.0, V = 1.0, M = 1.0, A = 0.8,
      Q = 0.8, H = 0.7, S = 0.7, T = 0.7, N = 0.6,
      K = 0.6, R = 0.5, C = 0.5, G = 0.4, P = 0.2
    )
  )
}


# ============================================================================
# ADD PROTEASOME FEATURES
# ============================================================================

add_proteasome_features <- function(data, c_term_p1_col = "c_term_P1") {
  #' Add proteasome cleavage-related features
  #'
  #' Focuses on C-terminal P1 position (most biologically relevant).
  #' The C-terminal P1 is the last residue of the epitope and sits
  #' in the proteasome's S1 catalytic pocket.
  #'
  #' @param data Data frame with cleavage position column
  #' @param c_term_p1_col Name of column containing C-terminal P1 residue
  
  # Verify column exists
  if (!c_term_p1_col %in% names(data)) {
    stop("Column '", c_term_p1_col, "' not found. Available columns: ",
         paste(names(data)[grepl("P1|term", names(data))], collapse = ", "))
  }
  
  prefs <- get_proteasome_preferences()
  cats <- get_aa_categories()
  
  data %>%
    mutate(
      # Get the P1 residue for easier reference
      .p1_residue = .data[[c_term_p1_col]],
      
      # ================================================================
      # C-TERMINAL P1 RESIDUE CATEGORIES
      # ================================================================
      c_term_P1_is_hydrophobic = .p1_residue %in% cats$hydrophobic,
      c_term_P1_is_basic = .p1_residue %in% cats$positive,
      c_term_P1_is_acidic = .p1_residue %in% cats$negative,
      c_term_P1_is_aromatic = .p1_residue %in% cats$aromatic,
      c_term_P1_is_small = .p1_residue %in% cats$small,
      
      # ================================================================
      # PROTEASOME PREFERENCE SCORES
      # ================================================================
      c_term_P1_immuno_score = sapply(.p1_residue, function(x) {
        if (is.na(x) || x == "" || !x %in% names(prefs$immunoproteasome)) {
          return(NA_real_)
        }
        prefs$immunoproteasome[x]
      }),
      
      c_term_P1_standard_score = sapply(.p1_residue, function(x) {
        if (is.na(x) || x == "" || !x %in% names(prefs$constitutive)) {
          return(NA_real_)
        }
        prefs$constitutive[x]
      }),
      
      # ================================================================
      # PROTEASOME TYPE PREFERENCE
      # ================================================================
      # Positive = favors immunoproteasome, Negative = favors standard
      proteasome_preference_diff = c_term_P1_immuno_score - c_term_P1_standard_score
      
    ) %>%
    # Remove temporary column
    select(-.p1_residue)
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


# =========================================
# NetSurfP - 3.0 functions
# =========================================
# Build lookup table: UniProt ID → NSP3 csv path
build_nsp3_path_lookup <- function(nsp3_root = "data/processed/nsp3/") {
  
  csv_files <- list.files(
    nsp3_root,
    pattern   = "\\.csv$",
    recursive = TRUE,
    full.names = TRUE
  )
  
  # Folder name looks like "4999_Q16678" → extract "Q16678"
  uniprot_from_path <- function(path) {
    folder <- basename(dirname(path))
    sub("^[0-9]+_", "", folder)
  }
  
  tibble(
    path       = csv_files,
    uniprot_id = uniprot_from_path(csv_files)
  )
}

# Read a single NSP3 output csv → clean residue table
read_nsp3_csv <- function(path, uniprot_id) {
  df <- data.table::fread(path, header = TRUE) |>
    as_tibble()
  
  names(df) <- trimws(names(df))
  
  # Rename bracket columns to valid R names
  # p[q3_H] → p_q3_H
  names(df) <- gsub("[", "_", names(df), fixed = TRUE)
  names(df) <- gsub("]", "",  names(df), fixed = TRUE)
  
  df |>
    select(
      n, seq, rsa,
      q3, p_q3_H, p_q3_E, p_q3_C,
      q8, p_q8_G, p_q8_H, p_q8_I, p_q8_B, p_q8_E, p_q8_S, p_q8_T, p_q8_C,
      disorder
    ) |>
    mutate(
      n        = as.integer(n),
      rsa      = as.numeric(rsa),
      disorder = as.numeric(disorder),
      q3       = as.character(q3),
      q8       = as.character(q8),
      across(starts_with("p_q"), as.numeric),
      uniprot  = uniprot_id
    )
}

# Aggregate NSP3 features over a slice of residues
aggregate_nsp3_window <- function(res_df) {
  if (nrow(res_df) == 0) {
    return(tibble(
      mean_rsa        = NA_real_,
      mean_disorder   = NA_real_,
      # q3
      frac_helix      = NA_real_,
      frac_sheet      = NA_real_,
      frac_coil       = NA_real_,
      mean_p_q3_H     = NA_real_,
      mean_p_q3_E     = NA_real_,
      mean_p_q3_C     = NA_real_,
      # q8
      frac_q8_G       = NA_real_,
      frac_q8_H       = NA_real_,
      frac_q8_I       = NA_real_,
      frac_q8_B       = NA_real_,
      frac_q8_E       = NA_real_,
      frac_q8_S       = NA_real_,
      frac_q8_T       = NA_real_,
      frac_q8_C       = NA_real_,
      mean_p_q8_G     = NA_real_,
      mean_p_q8_H     = NA_real_,
      mean_p_q8_I     = NA_real_,
      mean_p_q8_B     = NA_real_,
      mean_p_q8_E     = NA_real_,
      mean_p_q8_S     = NA_real_,
      mean_p_q8_T     = NA_real_,
      mean_p_q8_C     = NA_real_
    ))
  }
  
  tibble(
    mean_rsa        = mean(res_df$rsa,           na.rm = TRUE),
    mean_disorder   = mean(res_df$disorder,      na.rm = TRUE),
    # q3 hard assignments
    frac_helix      = mean(res_df$q3 == "H",     na.rm = TRUE),
    frac_sheet      = mean(res_df$q3 == "E",     na.rm = TRUE),
    frac_coil       = mean(res_df$q3 == "C",     na.rm = TRUE),
    # q3 probabilities
    mean_p_q3_H     = mean(res_df$p_q3_H,        na.rm = TRUE),
    mean_p_q3_E     = mean(res_df$p_q3_E,        na.rm = TRUE),
    mean_p_q3_C     = mean(res_df$p_q3_C,        na.rm = TRUE),
    # q8 hard assignments
    frac_q8_G       = mean(res_df$q8 == "G",     na.rm = TRUE),
    frac_q8_H       = mean(res_df$q8 == "H",     na.rm = TRUE),
    frac_q8_I       = mean(res_df$q8 == "I",     na.rm = TRUE),
    frac_q8_B       = mean(res_df$q8 == "B",     na.rm = TRUE),
    frac_q8_E       = mean(res_df$q8 == "E",     na.rm = TRUE),
    frac_q8_S       = mean(res_df$q8 == "S",     na.rm = TRUE),
    frac_q8_T       = mean(res_df$q8 == "T",     na.rm = TRUE),
    frac_q8_C       = mean(res_df$q8 == "C",     na.rm = TRUE),
    # q8 probabilities
    mean_p_q8_G     = mean(res_df$p_q8_G,        na.rm = TRUE),
    mean_p_q8_H     = mean(res_df$p_q8_H,        na.rm = TRUE),
    mean_p_q8_I     = mean(res_df$p_q8_I,        na.rm = TRUE),
    mean_p_q8_B     = mean(res_df$p_q8_B,        na.rm = TRUE),
    mean_p_q8_E     = mean(res_df$p_q8_E,        na.rm = TRUE),
    mean_p_q8_S     = mean(res_df$p_q8_S,        na.rm = TRUE),
    mean_p_q8_T     = mean(res_df$p_q8_T,        na.rm = TRUE),
    mean_p_q8_C     = mean(res_df$p_q8_C,        na.rm = TRUE)
  )
}

cat("functions.R loaded successfully.\n")