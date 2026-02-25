current_user <- Sys.info()[["user"]]

if (current_user == "olive") {
  setwd("C:/Users/olive/Documents/R/special_course_spring2026")
} else if (current_user == "mj607") {
  setwd("//wsl$/Ubuntu/home/hilarius/special_course_spring2026")
} else if (current_user == "hilarius") {
  setwd("/Users/hilarius/Desktop/DTU/special_course_spring2026")
} else {
  stop("Unknown user. Please set working directory manually.")
}


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
    stop("Failed to parse any rows. Please share the first ~40 lines of the .txt so the parser can be adjusted.")
  }
  
  df <- do.call(rbind, rows)
  
  df$Protein <- df$Identity
  df$Protein <- sub("^sp\\|([^|]+)\\|([^\\s]+)$", "sp_\\1_\\2", df$Protein)
  df$Protein <- sub("^tr\\|([^|]+)\\|([^\\s]+)$", "tr_\\1_\\2", df$Protein)
  
  df
}


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
