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
# =========================================
# NetSurfP - 3.0 functions
# =========================================
# Build lookup table: UniProt ID → NSP3 csv path
build_nsp3_path_lookup <- function(nsp3_root = "data/processed/nsp3/") {
  
  csv_files <- list.files(
    nsp3_root,
    pattern    = "\\.csv$",
    recursive  = TRUE,
    full.names = TRUE
  )
  
  # Path looks like: .../batch_1/01/4999_Q16678/4999_Q16678.csv
  # basename(dirname(path)) = "4999_Q16678"
  # sub("^[0-9]+_", "", ...) = "Q16678"
  uniprot_from_path <- function(path) {
    folder <- basename(dirname(path))
    sub("^[0-9]+_", "", folder)
  }
  
  tibble(
    path       = csv_files,
    uniprot_id = uniprot_from_path(csv_files)
  )
}

# ─────────────────────────────────────────
# Read a single NSP3 output csv
# Returns clean per-residue tibble
# ─────────────────────────────────────────
read_nsp3_csv <- function(path, uniprot_id) {
  
  df <- data.table::fread(path, header = TRUE) |>
    as_tibble()
  
  # Trim whitespace from column names
  names(df) <- trimws(names(df))
  
  # p[q3_H] → p_q3_H  (brackets not valid in R)
  names(df) <- gsub("[", "_", names(df), fixed = TRUE)
  names(df) <- gsub("]", "",  names(df), fixed = TRUE)
  
  # The id column has format ">Q16678" — strip the >
  # n is the residue number (1-indexed)
  df |>
    select(
      n, seq, rsa,
      q3, p_q3_H, p_q3_E, p_q3_C,
      q8, p_q8_G, p_q8_H, p_q8_I, p_q8_B,
      p_q8_E, p_q8_S, p_q8_T, p_q8_C,
      disorder
    ) |>
    mutate(
      n          = as.integer(n),
      rsa        = as.numeric(rsa),
      disorder   = as.numeric(disorder),
      q3         = as.character(q3),
      q8         = as.character(q8),
      across(starts_with("p_q"), as.numeric),
      uniprot    = uniprot_id
    )
}

# ─────────────────────────────────────────
# Aggregate NSP3 features over a window
# of residues (peptide, nflank, cflank etc)
# ─────────────────────────────────────────
aggregate_nsp3_window <- function(res_df) {
  
  # Return NAs if window is empty
  # (e.g. peptide at protein terminus with no flank)
  if (nrow(res_df) == 0) {
    return(tibble(
      mean_rsa      = NA_real_,
      mean_disorder = NA_real_,
      # q3 hard calls
      frac_helix    = NA_real_,
      frac_sheet    = NA_real_,
      frac_coil     = NA_real_,
      # q3 probabilities
      mean_p_q3_H   = NA_real_,
      mean_p_q3_E   = NA_real_,
      mean_p_q3_C   = NA_real_,
      # q8 hard calls
      frac_q8_G     = NA_real_,
      frac_q8_H     = NA_real_,
      frac_q8_I     = NA_real_,
      frac_q8_B     = NA_real_,
      frac_q8_E     = NA_real_,
      frac_q8_S     = NA_real_,
      frac_q8_T     = NA_real_,
      frac_q8_C     = NA_real_,
      # q8 probabilities
      mean_p_q8_G   = NA_real_,
      mean_p_q8_H   = NA_real_,
      mean_p_q8_I   = NA_real_,
      mean_p_q8_B   = NA_real_,
      mean_p_q8_E   = NA_real_,
      mean_p_q8_S   = NA_real_,
      mean_p_q8_T   = NA_real_,
      mean_p_q8_C   = NA_real_
    ))
  }
  
  tibble(
    mean_rsa      = mean(res_df$rsa,      na.rm = TRUE),
    mean_disorder = mean(res_df$disorder, na.rm = TRUE),
    # q3 hard calls — fraction of residues assigned to each class
    frac_helix    = mean(res_df$q3 == "H", na.rm = TRUE),
    frac_sheet    = mean(res_df$q3 == "E", na.rm = TRUE),
    frac_coil     = mean(res_df$q3 == "C", na.rm = TRUE),
    # q3 mean probabilities
    mean_p_q3_H   = mean(res_df$p_q3_H,  na.rm = TRUE),
    mean_p_q3_E   = mean(res_df$p_q3_E,  na.rm = TRUE),
    mean_p_q3_C   = mean(res_df$p_q3_C,  na.rm = TRUE),
    # q8 hard calls
    frac_q8_G     = mean(res_df$q8 == "G", na.rm = TRUE),
    frac_q8_H     = mean(res_df$q8 == "H", na.rm = TRUE),
    frac_q8_I     = mean(res_df$q8 == "I", na.rm = TRUE),
    frac_q8_B     = mean(res_df$q8 == "B", na.rm = TRUE),
    frac_q8_E     = mean(res_df$q8 == "E", na.rm = TRUE),
    frac_q8_S     = mean(res_df$q8 == "S", na.rm = TRUE),
    frac_q8_T     = mean(res_df$q8 == "T", na.rm = TRUE),
    frac_q8_C     = mean(res_df$q8 == "C", na.rm = TRUE),
    # q8 mean probabilities
    mean_p_q8_G   = mean(res_df$p_q8_G,  na.rm = TRUE),
    mean_p_q8_H   = mean(res_df$p_q8_H,  na.rm = TRUE),
    mean_p_q8_I   = mean(res_df$p_q8_I,  na.rm = TRUE),
    mean_p_q8_B   = mean(res_df$p_q8_B,  na.rm = TRUE),
    mean_p_q8_E   = mean(res_df$p_q8_E,  na.rm = TRUE),
    mean_p_q8_S   = mean(res_df$p_q8_S,  na.rm = TRUE),
    mean_p_q8_T   = mean(res_df$p_q8_T,  na.rm = TRUE),
    mean_p_q8_C   = mean(res_df$p_q8_C,  na.rm = TRUE)
  )
}

# ─────────────────────────────────────────
# Extract NSP3 features for one peptide
# across all four windows:
#   peptide / nflank / cflank / full_context
# ─────────────────────────────────────────
extract_nsp3_windows <- function(uniprot_id,
                                 nflank_start, nflank_end,
                                 pep_start,    pep_end,
                                 cflank_start, cflank_end,
                                 window_start, window_end,
                                 nsp3_split) {
  
  # Helper to build empty named output when protein is missing
  empty_window <- function(suffix) {
    empty_df <- tibble(
      n        = integer(),
      seq      = character(),
      rsa      = numeric(),
      q3       = character(),
      p_q3_H   = numeric(),
      p_q3_E   = numeric(),
      p_q3_C   = numeric(),
      q8       = character(),
      p_q8_G   = numeric(),
      p_q8_H   = numeric(),
      p_q8_I   = numeric(),
      p_q8_B   = numeric(),
      p_q8_E   = numeric(),
      p_q8_S   = numeric(),
      p_q8_T   = numeric(),
      p_q8_C   = numeric(),
      disorder = numeric(),
      uniprot  = character()
    )
    feats <- aggregate_nsp3_window(empty_df)
    names(feats) <- paste0(names(feats), suffix)
    feats
  }
  
  # ── Protein not in NSP3 output ──────────────────────────────────────────────
  res <- nsp3_split[[uniprot_id]]
  
  if (is.null(res)) {
    return(bind_cols(
      empty_window("_peptide"),
      empty_window("_nflank"),
      empty_window("_cflank"),
      empty_window("_full_context")
    ))
  }
  
  # ── Slice each window by residue number ─────────────────────────────────────
  # n is 1-indexed residue number matching UniProt coordinates
  pep_rows    <- res |> filter(n >= pep_start    & n <= pep_end)
  nflank_rows <- res |> filter(n >= nflank_start & n <= nflank_end)
  cflank_rows <- res |> filter(n >= cflank_start & n <= cflank_end)
  window_rows <- res |> filter(n >= window_start & n <= window_end)
  
  # ── Aggregate each window ────────────────────────────────────────────────────
  pep_feats    <- aggregate_nsp3_window(pep_rows)
  nflank_feats <- aggregate_nsp3_window(nflank_rows)
  cflank_feats <- aggregate_nsp3_window(cflank_rows)
  window_feats <- aggregate_nsp3_window(window_rows)
  
  # ── Add suffix so columns don't clash ───────────────────────────────────────
  names(pep_feats)    <- paste0(names(pep_feats),    "_peptide")
  names(nflank_feats) <- paste0(names(nflank_feats), "_nflank")
  names(cflank_feats) <- paste0(names(cflank_feats), "_cflank")
  names(window_feats) <- paste0(names(window_feats), "_full_context")
  
  bind_cols(pep_feats, nflank_feats, cflank_feats, window_feats)
}
#
# Load netmhcpan batches
#
# Read and combine all NetMHCpan batch CSV files
load_netmhcpan_batches <- function(netmhcpan_dir = "data/processed/netmhcpan/",
                                   binders_only = TRUE) {
  
  csv_files <- list.files(netmhcpan_dir, pattern = "\\.csv$", full.names = TRUE)
  cat("NetMHCpan batch files found:", length(csv_files), "\n")
  
  real_colnames <- c("Pos", "Peptide", "ID", "core", "icore", "Score", "Rank", "Ave", "NB")
  
  map(csv_files, function(f) {
    df <- read_table(f, skip = 2, col_names = real_colnames, show_col_types = FALSE)
    if (binders_only) df <- df |> filter(NB == 1)
    df
  }, .progress = TRUE) |>
    list_rbind()
}

cat("functions.R loaded successfully.\n")