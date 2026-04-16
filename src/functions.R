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
  #' Extract specific amino acid positions at N-terminal and C-terminal cleavage sites
  #'
  #' Uses Schechter-Berger nomenclature to extract the amino acids at key
  #' positions around each cleavage site.
  #'
  #' NOMENCLATURE:
  #'   ...N4 N3 N2 N1 | P1 P2 P3 P4 P5 P6 P7 P8 P9 | C1 C2 C3 C4...
  #'                  ^cut                         ^cut
  #'
  #'   P1 = residue immediately BEFORE the cut (most important!)
  #'   P1' = residue immediately AFTER the cut
  #'
  #' For each epitope, we extract positions for BOTH cleavage events:
  #'   1. N-terminal cleavage: where the epitope's N-terminus is created
  #'   2. C-terminal cleavage: where the epitope's C-terminus is created
  
  data |>
    mutate(
      # Store lengths for indexing
      pep_len = nchar(.data[[peptide_col]]),
      n_flank_len = nchar(.data[[n_flank_col]]),
      c_flank_len = nchar(.data[[c_flank_col]])
    ) |>
    # N-flank positions: N4, N3, N2, N1 (reading left to right)
    mutate(
      N4 = if_else(n_flank_len >= 4, substr(.data[[n_flank_col]], n_flank_len - 3, n_flank_len - 3), NA_character_),
      N3 = if_else(n_flank_len >= 3, substr(.data[[n_flank_col]], n_flank_len - 2, n_flank_len - 2), NA_character_),
      N2 = if_else(n_flank_len >= 2, substr(.data[[n_flank_col]], n_flank_len - 1, n_flank_len - 1), NA_character_),
      N1 = if_else(n_flank_len >= 1, substr(.data[[n_flank_col]], n_flank_len, n_flank_len), NA_character_)
    ) |>
    # Peptide positions: P1 through P9
    mutate(
      P1 = substr(.data[[peptide_col]], 1, 1),
      P2 = substr(.data[[peptide_col]], 2, 2),
      P3 = substr(.data[[peptide_col]], 3, 3),
      P4 = substr(.data[[peptide_col]], 4, 4),
      P5 = substr(.data[[peptide_col]], 5, 5),
      P6 = substr(.data[[peptide_col]], 6, 6),
      P7 = substr(.data[[peptide_col]], 7, 7),
      P8 = substr(.data[[peptide_col]], 8, 8),
      P9 = substr(.data[[peptide_col]], 9, 9)
    ) |>
    # C-flank positions: C1, C2, C3, C4
    mutate(
      C1 = if_else(c_flank_len >= 1, substr(.data[[c_flank_col]], 1, 1), NA_character_),
      C2 = if_else(c_flank_len >= 2, substr(.data[[c_flank_col]], 2, 2), NA_character_),
      C3 = if_else(c_flank_len >= 3, substr(.data[[c_flank_col]], 3, 3), NA_character_),
      C4 = if_else(c_flank_len >= 4, substr(.data[[c_flank_col]], 4, 4), NA_character_)
    ) |>
    # Remove temporary length columns
    select(-pep_len, -n_flank_len, -c_flank_len)
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
  
  names(df) <- trimws(names(df))
  names(df) <- gsub("[", "_", names(df), fixed = TRUE)
  names(df) <- gsub("]", "",  names(df), fixed = TRUE)
  
  df |>
    select(
      n, seq, rsa,
      q8, p_q8_G, p_q8_H, p_q8_I, p_q8_B,
      p_q8_E, p_q8_S, p_q8_T, p_q8_C,
      disorder
    ) |>
    mutate(
      n          = as.integer(n),
      rsa        = as.numeric(rsa),
      disorder   = as.numeric(disorder),
      q8         = as.character(q8),
      across(starts_with("p_q"), as.numeric),
      uniprot    = uniprot_id
    )
}

# ─────────────────────────────────────────
# Aggregate NSP3 features over a window
# of residues (peptide, nflank, cflank etc)
# Now only computes RSA and disorder (Q8 handled by point system)
# ─────────────────────────────────────────
aggregate_nsp3_window <- function(res_df) {
  
  if (nrow(res_df) == 0) {
    return(tibble(
      mean_rsa      = NA_real_,
      mean_disorder = NA_real_
    ))
  }
  
  tibble(
    mean_rsa      = mean(res_df$rsa,      na.rm = TRUE),
    mean_disorder = mean(res_df$disorder,  na.rm = TRUE)
  )
}

# ─────────────────────────────────────────
# Extract NSP3 features for one peptide
# across three windows:
#   peptide / nflank / cflank
# Returns RSA and disorder only (Q8 handled by extract_nsp3_q8_features)
# ─────────────────────────────────────────
extract_nsp3_windows <- function(uniprot_id,
                                 nflank_start, nflank_end,
                                 pep_start,    pep_end,
                                 cflank_start, cflank_end,
                                 nsp3_split) {
  
  na_row <- tibble(
    mean_rsa_peptide = NA_real_, mean_disorder_peptide = NA_real_,
    mean_rsa_nflank  = NA_real_, mean_disorder_nflank  = NA_real_,
    mean_rsa_cflank  = NA_real_, mean_disorder_cflank  = NA_real_
  )
  
  res <- nsp3_split[[uniprot_id]]
  if (is.null(res)) return(na_row)
  
  # Slice each window by residue number
  pep_rows    <- res[res$n >= pep_start    & res$n <= pep_end, ]
  nflank_rows <- if (!is.na(nflank_start) && !is.na(nflank_end) && nflank_end >= nflank_start) {
    res[res$n >= nflank_start & res$n <= nflank_end, ]
  } else {
    res[0, ]
  }
  cflank_rows <- if (!is.na(cflank_start) && !is.na(cflank_end) && cflank_end >= cflank_start) {
    res[res$n >= cflank_start & res$n <= cflank_end, ]
  } else {
    res[0, ]
  }
  
  # Aggregate each window
  pep_feats    <- aggregate_nsp3_window(pep_rows)
  nflank_feats <- aggregate_nsp3_window(nflank_rows)
  cflank_feats <- aggregate_nsp3_window(cflank_rows)
  
  # Add suffix so columns don't clash
  names(pep_feats)    <- paste0(names(pep_feats),    "_peptide")
  names(nflank_feats) <- paste0(names(nflank_feats), "_nflank")
  names(cflank_feats) <- paste0(names(cflank_feats), "_cflank")
  
  bind_cols(pep_feats, nflank_feats, cflank_feats)
}


# Load netmhcpan batches
#
# Read and combine all NetMHCpan batch CSV files (only 9 mers NOW)
load_netmhcpan_batches <- function(netmhcpan_dir = "data/processed/netmhcpan/",
                                   binders_only = TRUE,
                                   peptide_length = NULL) {
  
  csv_files <- list.files(netmhcpan_dir, pattern = "\\.csv$", full.names = TRUE)
  cat("NetMHCpan batch files found:", length(csv_files), "\n")
  
  real_colnames <- c("Pos", "Peptide", "ID", "core", "icore", "Score", "Rank", "Ave", "NB")
  
  map(csv_files, function(f) {
    df <- read_table(f, skip = 2, col_names = real_colnames, show_col_types = FALSE)
    if (!is.null(peptide_length)) df <- df |> filter(nchar(Peptide) == peptide_length)
    if (binders_only) df <- df |> filter(NB == 1)
    df
  }, .progress = TRUE) |>
    list_rbind()
}

get_netmhcpan_total <- function(netmhcpan_dir = "data/processed/netmhcpan/",
                                peptide_length = NULL,
                                valid_ids = NULL) {
  csv_files <- list.files(netmhcpan_dir, pattern = "\\.csv$", full.names = TRUE)
  cat("Counting total peptides across", length(csv_files), "NetMHCpan files...\n")
  
  real_colnames <- c("Pos", "Peptide", "ID", "core", "icore", "Score", "Rank", "Ave", "NB")
  
  total_peptides <- map_dbl(csv_files, function(f) {
    df <- read_table(f, skip = 2, col_names = real_colnames, show_col_types = FALSE)
    if (!is.null(peptide_length)) df <- df |> filter(nchar(Peptide) == peptide_length)
    if (!is.null(valid_ids)) {
      df <- df |> 
        mutate(uniprot_id = str_split_i(ID, "_", 2)) |>
        filter(uniprot_id %in% valid_ids)
    }
    nrow(df)
  }, .progress = TRUE) |> sum()
  
  return(total_peptides)
}

### Alpha fold features
# ── AlphaFold pLDDT helpers ───────────────────────────────────────────────────

#' Read one AlphaFold PDB and return a per-residue pLDDT lookup table.
#'
#' AlphaFold stores pLDDT in the B-factor field.
#' We keep only CA atoms so we get exactly one row per residue.
#'
#' @param uniprot_id  Character. UniProt accession (= PDB file stem).
#' @param dir         Character. Directory that holds the .pdb files.
#' @return A tibble with columns: uniprot_id, residue_num, aa, plddt
#'         Returns NULL (with a warning) if the file cannot be read.

extract_plddt_lookup <- function(uniprot_id, dir) {
  pdb_path <- file.path(dir, paste0(uniprot_id, ".pdb"))
  
  tryCatch({
    pdb <- bio3d::read.pdb(pdb_path, verbose = FALSE)
    
    # One row per residue — CA atoms carry the residue number and pLDDT
    ca <- pdb$atom[pdb$atom$elety == "CA", ]
    
    tibble::tibble(
      uniprot_id  = uniprot_id,
      residue_num = as.integer(ca$resno),
      aa          = ca$resid,          # three-letter amino acid code
      plddt       = as.numeric(ca$b)   # B-factor = pLDDT in AlphaFold PDBs
    )
  }, error = function(e) {
    warning(sprintf("[extract_plddt_lookup] Failed for %s: %s",
                    uniprot_id, e$message))
    NULL
  })
}


#' Build a full pLDDT lookup list from all PDB files in a directory.
#'
#' @param proteins_needed  Character vector of UniProt IDs to load.
#' @param dir              Directory containing .pdb files.
#' @return A named list (keyed by uniprot_id).
#'         Each element is a tibble with residue_num, aa, plddt.

build_plddt_lookup <- function(proteins_needed, dir) {
  # Discover which proteins actually have a PDB file
  af_files        <- list.files(dir, pattern = "\\.pdb$", full.names = FALSE)
  af_uniprot_all  <- stringr::str_remove(af_files, "\\.pdb$")
  af_uniprot_filt <- af_uniprot_all[af_uniprot_all %in% proteins_needed]
  
  n_found   <- length(af_uniprot_filt)
  n_needed  <- length(proteins_needed)
  n_missing <- n_needed - n_found
  
  message(sprintf(
    "AlphaFold PDBs found: %d / %d needed  (%d missing)",
    n_found, n_needed, n_missing
  ))
  
  if (n_missing > 0 & n_missing <= 10) {
    # Only show missing IDs if there are few of them
    missing_ids <- setdiff(proteins_needed, af_uniprot_filt)
    message("Missing UniProt IDs: ",
            paste(missing_ids, collapse = ", "))
  } else if (n_missing > 10) {
    message(sprintf("(%d missing proteins not listed)", n_missing))
  }
  
  # Load all files with progress indication
  message("Loading pLDDT data...")
  lookup_flat <- purrr::map(af_uniprot_filt, extract_plddt_lookup,
                            dir = dir, .progress = TRUE) |>
    purrr::list_rbind()
  
  message(sprintf(
    "Loaded pLDDT for %d proteins, %d residues total.",
    dplyr::n_distinct(lookup_flat$uniprot_id),
    nrow(lookup_flat)
  ))
  
  # Return as a named list (fast access by protein)
  split(lookup_flat, lookup_flat$uniprot_id)
}


#' Extract per-residue pLDDT scores for one genomic window.
#'
#' @param uid          UniProt ID (character).
#' @param start        Integer. First residue of the window.
#' @param end          Integer. Last residue of the window.
#' @param lookup_split Named list produced by build_plddt_lookup().
#' @return A numeric vector of pLDDT scores (one per residue in order).
#'         Returns NA_real_ if the protein is missing or the window is empty.

extract_plddt_vector <- function(uid, start, end, lookup_split) {
  lkp <- lookup_split[[uid]]
  if (is.null(lkp) || nrow(lkp) == 0) return(NA_real_)
  
  vals <- lkp$plddt[lkp$residue_num >= start & lkp$residue_num <= end]
  if (length(vals) == 0) return(NA_real_)
  vals   # full vector, in residue order
}


#' Summarise a pLDDT vector to a scalar mean (safe version).
#' Useful when you just want one number per region.

mean_plddt_region <- function(uid, start, end, lookup_split) {
  v <- extract_plddt_vector(uid, start, end, lookup_split)
  if (all(is.na(v))) return(NA_real_)
  mean(v, na.rm = TRUE)
}

# ============================================================================
# RE-ENGINEERED NETSURFP Q8 FEATURES — BIOLOGICAL POINT SYSTEM
# ============================================================================

# Constants used by compute_q8_full_context and make_na_result
Q8_MIN_RUNS <- c(
  H = 4L, G = 3L, I = 5L, E = 2L,
  B = 1L, T = 2L, S = 1L, C = 1L
)
Q8_STATES <- names(Q8_MIN_RUNS)
RESCUE_THRESHOLD <- 0.4


#' Find contiguous runs in a character vector
#' @return data.frame: value, start, end, len
find_all_runs <- function(x) {
  r <- rle(x)
  ends   <- cumsum(r$lengths)
  starts <- ends - r$lengths + 1L
  data.frame(
    value = r$values,
    start = starts,
    end   = ends,
    len   = r$lengths,
    stringsAsFactors = FALSE
  )
}


#' Determine which window(s) a run overlaps
#' @param run_start Integer start of run (in full-context coords)
#' @param run_end Integer end of run
#' @param nfl_end Last position of nflank in full context
#' @param pep_end Last position of peptide in full context
#' @return Character vector: subset of c("nflank", "peptide", "cflank")
run_overlaps_windows <- function(run_start, run_end, nfl_end, pep_end) {
  windows <- character(0)
  if (run_start <= nfl_end)                      windows <- c(windows, "nflank")
  if (run_end > nfl_end & run_start <= pep_end)  windows <- c(windows, "peptide")
  if (run_end > pep_end)                         windows <- c(windows, "cflank")
  windows
}


#' Extract Q8 probability matrix from NSP3 data for a given window
#'
#' @param nsp3_protein data.frame of NSP3 output for one protein
#' @param start Start position (1-indexed, protein coordinates)
#' @param end End position (1-indexed, protein coordinates)
#' @return Matrix with nrow = (end - start + 1), ncol = 8
extract_q8_matrix <- function(nsp3_protein, start, end) {
  if (is.null(nsp3_protein) || start < 1 || end < start) {
    return(matrix(NA_real_, nrow = 0, ncol = 8,
                  dimnames = list(NULL, Q8_STATES)))
  }
  
  # Filter by residue number (safe — doesn't assume row order)
  rows <- nsp3_protein[nsp3_protein$n >= start & nsp3_protein$n <= end, ]
  
  if (nrow(rows) == 0) {
    return(matrix(NA_real_, nrow = 0, ncol = 8,
                  dimnames = list(NULL, Q8_STATES)))
  }
  
  # These match the column names from read_nsp3_csv
  q8_cols <- c("p_q8_H", "p_q8_G", "p_q8_I", "p_q8_E",
               "p_q8_B", "p_q8_T", "p_q8_S", "p_q8_C")
  mat <- as.matrix(rows[, q8_cols])
  colnames(mat) <- Q8_STATES
  mat
}


#' Core: compute Q8 points and transitions on a full-context probability matrix
#'
#' @param prob_matrix Matrix [n_residues x 8], colnames = Q8_STATES.
#'   Rows ordered: nflank residues, then peptide, then cflank
#' @param n_nflank Integer length of nflank (can be 0 if at protein start)
#' @param n_peptide Integer length of peptide (9)
#' @param n_cflank Integer length of cflank (can be 0 if at protein end)
#' @param rescue_threshold Numeric (default 0.4)
#' @return Single-row tibble with point and transition features
compute_q8_full_context <- function(prob_matrix,
                                    n_nflank,
                                    n_peptide,
                                    n_cflank,
                                    rescue_threshold = RESCUE_THRESHOLD) {
  
  n_total <- nrow(prob_matrix)
  
  if (n_total == 0 || is.null(prob_matrix)) {
    return(make_na_result())
  }
  
  stopifnot(all(Q8_STATES %in% colnames(prob_matrix)))
  prob_matrix <- prob_matrix[, Q8_STATES, drop = FALSE]
  
  # Window boundaries in full-context coordinates (1-indexed)
  nfl_end <- n_nflank
  pep_end <- n_nflank + n_peptide
  
  # --- Step 1: Argmax assignment ---
  argmax_idx <- apply(prob_matrix, 1, which.max)
  primary    <- Q8_STATES[argmax_idx]
  
  # --- Step 2: Find all runs from argmax ---
  all_runs <- find_all_runs(primary)
  
  # --- Step 3: Identify qualifying runs (meet minimum) ---
  qualifying <- all_runs |>
    mutate(min_needed = Q8_MIN_RUNS[value]) |>
    filter(len >= min_needed)
  
  # --- Step 4: Rescue for runs that are exactly 1 short ---
  shortfall <- all_runs |>
    mutate(min_needed = Q8_MIN_RUNS[value]) |>
    filter(len == min_needed - 1L, min_needed > 1L)
  
  already_qualified <- unique(qualifying$value)
  
  rescue_list <- list()
  
  for (i in seq_len(nrow(shortfall))) {
    state     <- shortfall$value[i]
    run_start <- shortfall$start[i]
    run_end   <- shortfall$end[i]
    
    if (state %in% already_qualified) next
    
    # Check left neighbor
    left_pos <- run_start - 1L
    if (left_pos >= 1 && prob_matrix[left_pos, state] >= rescue_threshold) {
      rescue_list[[length(rescue_list) + 1]] <- data.frame(
        state         = state,
        position      = left_pos,
        prob          = prob_matrix[left_pos, state],
        min_run       = Q8_MIN_RUNS[state],
        new_run_start = left_pos,
        new_run_end   = run_end,
        stringsAsFactors = FALSE
      )
    }
    
    # Check right neighbor
    right_pos <- run_end + 1L
    if (right_pos <= n_total && prob_matrix[right_pos, state] >= rescue_threshold) {
      rescue_list[[length(rescue_list) + 1]] <- data.frame(
        state         = state,
        position      = right_pos,
        prob          = prob_matrix[right_pos, state],
        min_run       = Q8_MIN_RUNS[state],
        new_run_start = run_start,
        new_run_end   = right_pos,
        stringsAsFactors = FALSE
      )
    }
  }
  
  # Resolve rescue conflicts
  rescued_states    <- character(0)
  claimed_positions <- integer(0)
  rescued_runs <- data.frame(
    value = character(0), start = integer(0), end = integer(0),
    stringsAsFactors = FALSE
  )
  
  if (length(rescue_list) > 0) {
    rc_df <- bind_rows(rescue_list) |>
      arrange(desc(min_run), desc(prob))
    
    for (i in seq_len(nrow(rc_df))) {
      s   <- rc_df$state[i]
      pos <- rc_df$position[i]
      
      if (pos %in% claimed_positions) next
      if (s %in% rescued_states) next
      if (s %in% already_qualified) next
      
      claimed_positions <- c(claimed_positions, pos)
      rescued_states    <- c(rescued_states, s)
      rescued_runs <- bind_rows(rescued_runs, data.frame(
        value = s,
        start = rc_df$new_run_start[i],
        end   = rc_df$new_run_end[i],
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # --- Step 5: Combine qualifying + rescued runs ---
  all_valid_runs <- bind_rows(
    qualifying |> select(value, start, end),
    rescued_runs
  )
  
  # --- Step 6: Assign points per window + detect transitions ---
  points <- list(
    nflank  = setNames(rep(0L, 8), Q8_STATES),
    peptide = setNames(rep(0L, 8), Q8_STATES),
    cflank  = setNames(rep(0L, 8), Q8_STATES)
  )
  
  trans_n_pep <- setNames(rep(0L, 8), Q8_STATES)
  trans_pep_c <- setNames(rep(0L, 8), Q8_STATES)
  
  for (i in seq_len(nrow(all_valid_runs))) {
    state <- all_valid_runs$value[i]
    rs    <- all_valid_runs$start[i]
    re    <- all_valid_runs$end[i]
    
    windows <- run_overlaps_windows(rs, re, nfl_end, pep_end)
    
    for (w in windows) {
      points[[w]][state] <- 1L
    }
    
    if ("nflank" %in% windows && "peptide" %in% windows) {
      trans_n_pep[state] <- 1L
    }
    if ("peptide" %in% windows && "cflank" %in% windows) {
      trans_pep_c[state] <- 1L
    }
  }
  
  # --- Step 7: Assemble output ---
  result <- c(
    setNames(points$peptide, paste0("q8point_", Q8_STATES, "_peptide")),
    setNames(points$nflank,  paste0("q8point_", Q8_STATES, "_nflank")),
    setNames(points$cflank,  paste0("q8point_", Q8_STATES, "_cflank")),
    setNames(trans_n_pep,    paste0("q8trans_", Q8_STATES, "_n_pep")),
    setNames(trans_pep_c,    paste0("q8trans_", Q8_STATES, "_pep_c"))
  )
  
  as_tibble_row(result)
}


#' Helper: NA result for missing data
make_na_result <- function() {
  nms <- c(
    paste0("q8point_", Q8_STATES, "_peptide"),
    paste0("q8point_", Q8_STATES, "_nflank"),
    paste0("q8point_", Q8_STATES, "_cflank"),
    paste0("q8trans_", Q8_STATES, "_n_pep"),
    paste0("q8trans_", Q8_STATES, "_pep_c")
  )
  as_tibble_row(setNames(rep(NA_integer_, length(nms)), nms))
}


#' Full extraction for one peptide: builds full context, computes Q8 points
#'
#' @param uniprot_id UniProt ID
#' @param nflank_start Start of N-flank (protein coordinates, 1-indexed)
#' @param nflank_end End of N-flank
#' @param pep_start Start of peptide
#' @param pep_end End of peptide
#' @param cflank_start Start of C-flank
#' @param cflank_end End of C-flank
#' @param nsp3_split Named list of NSP3 data.frames by protein
#' @return Single-row tibble with q8point and q8trans features
extract_nsp3_q8_features <- function(uniprot_id,
                                     nflank_start, nflank_end,
                                     pep_start, pep_end,
                                     cflank_start, cflank_end,
                                     nsp3_split) {
  
  nsp3_prot <- nsp3_split[[uniprot_id]]
  if (is.null(nsp3_prot)) return(make_na_result())
  
  # Guard against NA peptide coordinates
  if (is.na(pep_start) || is.na(pep_end)) return(make_na_result())
  
  prot_len <- max(nsp3_prot$n, na.rm = TRUE)
  
  # --- Determine actual valid ranges within the protein ---
  # Peptide: always valid (9-mer, must exist in protein)
  pep_start_c <- as.integer(pep_start)
  pep_end_c   <- as.integer(pep_end)
  actual_pep_len <- pep_end_c - pep_start_c + 1L
  
  # N-flank: may be missing (peptide at N-terminus) or shorter
  if (is.na(nflank_start) || is.na(nflank_end) || nflank_end < 1L || nflank_start > nflank_end) {
    actual_nfl_len <- 0L
    nfl_start_c    <- pep_start_c  # placeholder, won't be used
  } else {
    nfl_start_c    <- max(1L, as.integer(nflank_start))
    nfl_end_c      <- min(as.integer(nflank_end), prot_len)
    actual_nfl_len <- if (nfl_end_c >= nfl_start_c) nfl_end_c - nfl_start_c + 1L else 0L
  }
  
  # C-flank: may be missing (peptide at C-terminus) or shorter
  if (is.na(cflank_start) || is.na(cflank_end) || cflank_start > prot_len || cflank_start > cflank_end) {
    actual_cfl_len <- 0L
    cfl_end_c      <- pep_end_c  # placeholder, won't be used
  } else {
    cfl_start_c    <- as.integer(cflank_start)
    cfl_end_c      <- min(as.integer(cflank_end), prot_len)
    actual_cfl_len <- if (cfl_end_c >= cfl_start_c) cfl_end_c - cfl_start_c + 1L else 0L
  }
  
  # --- Build full context range ---
  full_start <- if (actual_nfl_len > 0) nfl_start_c else pep_start_c
  full_end   <- if (actual_cfl_len > 0) cfl_end_c   else pep_end_c
  
  mat_full <- extract_q8_matrix(nsp3_prot, full_start, full_end)
  
  if (nrow(mat_full) == 0) return(make_na_result())
  
  # Verify dimensions match expectations
  expected_len <- actual_nfl_len + actual_pep_len + actual_cfl_len
  if (nrow(mat_full) != expected_len) {
    warning(sprintf(
      "Matrix size mismatch for %s pos %d-%d: expected %d rows, got %d",
      uniprot_id, pep_start, pep_end, expected_len, nrow(mat_full)
    ))
    return(make_na_result())
  }
  
  compute_q8_full_context(
    prob_matrix = mat_full,
    n_nflank    = actual_nfl_len,
    n_peptide   = actual_pep_len,
    n_cflank    = actual_cfl_len
  )
}

#  Cliff's Delta (overflow-safe, chunked computation) 
cliff_delta_sampled <- function(x, y, n_sample = 50000, seed = 42) {
  set.seed(seed)
  if (length(x) > n_sample) x <- sample(x, n_sample)
  if (length(y) > n_sample) y <- sample(y, n_sample)
  
  x_chunks <- split(x, ceiling(seq_along(x) / 5000))
  total_greater <- 0; total_less <- 0; total_pairs <- 0
  
  for (chunk in x_chunks) {
    comparisons <- outer(chunk, y, FUN = function(a, b) sign(a - b))
    total_greater <- total_greater + sum(comparisons > 0)
    total_less    <- total_less    + sum(comparisons < 0)
    total_pairs   <- total_pairs   + length(comparisons)
  }
  
  delta <- (total_greater - total_less) / total_pairs
  magnitude <- case_when(
    abs(delta) < 0.147 ~ "negligible",
    abs(delta) < 0.33  ~ "small",
    abs(delta) < 0.474 ~ "medium",
    TRUE               ~ "large"
  )
  list(estimate = delta, magnitude = magnitude)
}

sample_bin_matched <- function(df_pos, df_neg, n_bins, seed = 42) {
  breaks <- seq(0, 2, length.out = n_bins + 1)
  
  pos_binned <- df_pos |>
    mutate(rank_bin = cut(rank, breaks = breaks, include.lowest = TRUE))
  neg_binned <- df_neg |>
    mutate(rank_bin = cut(rank, breaks = breaks, include.lowest = TRUE))
  
  props <- pos_binned |> 
    count(rank_bin, name = "n_pos") |> 
    mutate(prop = n_pos / sum(n_pos))
  avail <- neg_binned |> 
    count(rank_bin, name = "n_available")
  
  plan <- props |>
    left_join(avail, by = "rank_bin") |>
    mutate(n_available = replace_na(n_available, 0),
           max_total = ifelse(prop > 0, n_available / prop, Inf))
  
  max_n <- floor(min(plan$max_total))
  plan <- plan |> mutate(n_sample = floor(max_n * prop))
  
  set.seed(seed)
  plan |>
    select(rank_bin, n_sample) |>
    pmap_dfr(function(rank_bin, n_sample) {
      neg_binned |>
        filter(rank_bin == !!rank_bin) |>
        slice_sample(n = n_sample) # random within each bin
    }) |>
    mutate(label = 0)
}

cat("functions.R loaded successfully.\n")