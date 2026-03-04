library(tidyverse)
library(stringr)

# working directory setup for git
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

# cleaning of colnames
clean_names <- function(x) {
  x |>
    str_replace_all("\\.\\.\\.", "_") |>
    str_replace_all("\\.", "_") |>
    str_replace_all("-", "_") |>
    str_replace_all(" ", "_") |>
    str_to_lower()
}

# read fasta file into df
read_fasta_df <- function(filepath) {
  
  if (!file.exists(filepath)) {
    stop("FASTA file not found: ", filepath)
  }
  
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

# Flanking region extraction
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
  
  pep_start <- as.integer(match_pos)
  pep_length <- nchar(peptide)
  pep_end <- pep_start + pep_length - 1
  prot_length <- nchar(protein_seq)
  
  n_flank_start <- max(1, pep_start - flank_size)
  n_flank_end <- pep_start - 1
  
  if (n_flank_end >= n_flank_start) {
    n_flank <- substr(protein_seq, n_flank_start, n_flank_end)
  } else {
    n_flank <- ""
  }
  n_flank_padded <- paste0(strrep("X", flank_size - nchar(n_flank)), n_flank)
  
  c_flank_start <- pep_end + 1
  c_flank_end <- min(prot_length, pep_end + flank_size)
  
  if (c_flank_end >= c_flank_start) {
    c_flank <- substr(protein_seq, c_flank_start, c_flank_end)
  } else {
    c_flank <- ""
  }
  c_flank_padded <- paste0(c_flank, strrep("X", flank_size - nchar(c_flank)))
  
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

# K-mer generation
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

verify_peptide_in_protein <- function(peptide, uniprot_id, df_fasta) {
  
  prot_seq <- df_fasta$sequence[df_fasta$uniprot_id == uniprot_id]
  
  if (length(prot_seq) == 0) return(FALSE)
  
  any(grepl(peptide, prot_seq, fixed = TRUE))
}

# NetMHCpan web output parsing
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
    score_idx <- which(grepl("^\\d*\\.\\d+$", x) | grepl("^\\d+\\.\\d+$", x))
    id_candidates <- which(grepl("^(sp|tr)_.+", x) | grepl("^[A-Za-z0-9]+_[A-Za-z0-9]+", x))
    
    if (length(id_candidates) == 0) {
      if (length(score_idx) == 0) return(NULL)
      identity_idx <- score_idx[1] - 1
      score_idx <- score_idx[1]
    } else {
      if (length(score_idx) == 0) return(NULL)
      score_first <- score_idx[1]
      identity_idx <- max(id_candidates[id_candidates < score_first], na.rm = TRUE)
      score_idx <- score_first
    }
    
    pos <- as.integer(x[1])
    mhc <- x[2]
    peptide <- x[3]
    core <- x[4]
    identity <- x[identity_idx]
    rest <- x[(identity_idx + 1):length(x)]
    nums <- suppressWarnings(as.numeric(rest))
    numeric_positions <- which(!is.na(nums))
    
    if (length(numeric_positions) < 5) {
      score <- if (length(numeric_positions) >= 1) nums[numeric_positions[1]] else NA_real_
      rank <- if (length(numeric_positions) >= 2) nums[numeric_positions[2]] else NA_real_
      ba_score <- if (length(numeric_positions) >= 3) nums[numeric_positions[3]] else NA_real_
      ba_rank <- if (length(numeric_positions) >= 4) nums[numeric_positions[4]] else NA_real_
      aff_nm <- if (length(numeric_positions) >= 5) nums[numeric_positions[5]] else NA_real_
    } else {
      score   <- nums[numeric_positions[1]]
      rank    <- nums[numeric_positions[2]]
      ba_score <- nums[numeric_positions[3]]
      ba_rank  <- nums[numeric_positions[4]]
      aff_nm   <- nums[numeric_positions[5]]
    }
    
    bindlevel <- NA_character_
    if (length(numeric_positions) >= 5) {
      after_aff_idx <- numeric_positions[5] + 1
      if (after_aff_idx <= length(rest)) {
        bind_tokens <- rest[after_aff_idx:length(rest)]
        if (length(bind_tokens) > 0) bindlevel <- paste(bind_tokens, collapse = " ")
      }
    }
    
    data.frame(
      Pos = pos,
      MHC = mhc,
      Peptide = peptide,
      Core = core,
      Identity = identity,
      Score = score,
      Rank = rank,
      BA_score = ba_score,
      BA_rank = ba_rank,
      Aff_nm = aff_nm,
      BindLevel = bindlevel,
      stringsAsFactors = FALSE
    )
  }
  
  rows <- lapply(fields, parse_row)
  rows <- rows[!vapply(rows, is.null, logical(1))]
  
  if (length(rows) == 0) {
    stop("Failed to parse any rows. Please check the input file format.")
  }
  
  df <- do.call(rbind, rows)
  
  df$Protein <- df$Identity
  df$Protein <- sub("^sp\\|([^|]+)\\|([^\\s]+)$", "sp_\\1_\\2", df$Protein)
  df$Protein <- sub("^tr\\|([^|]+)\\|([^\\s]+)$", "tr_\\1_\\2", df$Protein)
  
  df
}

# epitope export
export_epitopes <- function(data,
                            file,
                            separator = c("comma", "newline"),
                            unique_only = TRUE,
                            sort_output = FALSE) {
  
  separator <- match.arg(separator)
  
  epitopes <- data %>%
    pull(epitope)
  
  if (unique_only) {
    epitopes <- unique(epitopes)
  }
  
  if (sort_output) {
    epitopes <- sort(epitopes)
  }
  
  if (separator == "comma") {
    output <- str_c(epitopes, collapse = ",")
    write_lines(output, file)
  }
  
  if (separator == "newline") {
    write_lines(epitopes, file)
  }
  
  invisible(epitopes)
}

validate_flanking_regions <- function(df, df_fasta) {
  
  results <- list(
    total_rows = nrow(df),
    peptides_found = sum(df$peptide_found, na.rm = TRUE),
    peptides_not_found = sum(!df$peptide_found, na.rm = TRUE)
  )
  
  df <- df |>
    mutate(
      expected_length = 8 + pep_length + 8,
      actual_length = nchar(full_context),
      length_ok = expected_length == actual_length
    )
  
  results$correct_lengths <- sum(df$length_ok, na.rm = TRUE)
  results$incorrect_lengths <- sum(!df$length_ok, na.rm = TRUE)
  
  df <- df |>
    mutate(
      extracted_peptide = substr(full_context, 9, 9 + pep_length - 1),
      peptide_matches = peptide == extracted_peptide
    )
  
  results$peptide_position_correct <- sum(df$peptide_matches, na.rm = TRUE)
  results$peptide_position_incorrect <- sum(!df$peptide_matches, na.rm = TRUE)
  
  cat("\n=== Flanking Region Validation ===\n")
  cat("Total rows:", results$total_rows, "\n")
  cat("Peptides found:", results$peptides_found, "\n")
  cat("Peptides not found:", results$peptides_not_found, "\n")
  cat("Correct context lengths:", results$correct_lengths, "\n")
  cat("Incorrect context lengths:", results$incorrect_lengths, "\n")
  cat("Peptide at correct position:", results$peptide_position_correct, "\n")
  cat("Peptide at wrong position:", results$peptide_position_incorrect, "\n")
  
  invisible(results)
}

cat("functions.R loaded successfully.\n")