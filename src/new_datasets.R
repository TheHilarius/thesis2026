library(tidyverse)
library(stringr)
source("src/functions.R")

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
  select(where(~ !all(is.na(.))))

df_clean <- df_clean |>
  mutate(
    across(where(is.character), ~ na_if(.x, ""))
  )

df_clean <- df_clean |>
  mutate(
    pep_length = nchar(epitope_name)
  )

df_clean <- df_clean |>
  mutate(
    pep_length = nchar(epitope_name),
    epitope_starting_position = as.integer(epitope_starting_position),
    epitope_ending_position   = as.integer(epitope_ending_position)
  )


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
    peptide   = epitope_name,
    start     = epitope_starting_position,
    end       = epitope_ending_position,
    protein   = epitope_molecule_parent,
    protein_iri = epitope_molecule_parent_iri,
    organism  = epitope_source_organism
  ) |>
  mutate(
    uniprot_id = str_extract(protein_iri, "[^/]+$")
  )

df_pos_9 <- df_epitope |>
  filter(pep_length == 9) |>
  distinct(peptide, uniprot_id, .keep_all = TRUE) |>
  mutate(label = 1)

df_protein_info <- df_epitope |>
  select(protein, protein_iri, organism, uniprot_id) |>
  distinct(uniprot_id, .keep_all = TRUE)

#####################
# Generate negatives
#####################
read_fasta_df <- function(filepath) {
  
  lines <- readLines(filepath)
  
  header_idx <- grep("^>", lines)
  
  headers <- lines[header_idx]
  
  sequences <- vector("character", length(header_idx))
  
  for (i in seq_along(header_idx)) {
    
    start <- header_idx[i] + 1
    end   <- if (i < length(header_idx)) header_idx[i + 1] - 1 else length(lines)
    
    sequences[i] <- paste(lines[start:end], collapse = "")
  }
  
  tibble(
    header   = headers,
    sequence = sequences
  )
}

###
# ------------------------------------------------------------
# Load FASTA
# ------------------------------------------------------------

df_fasta <- read_fasta_df("data/combined.fasta") |>
  mutate(
    uniprot_id = str_extract(header, "(?<=\\|)[A-Z0-9]+(?=\\|)")
  )

# Keep only proteins that exist in both datasets
df_fasta <- df_fasta |>
  filter(uniprot_id %in% unique(df_pos_9$uniprot_id))

# ------------------------------------------------------------
# Generate all 9-mers
# ------------------------------------------------------------

generate_9mers <- function(seq, uid) {
  
  seq_length <- nchar(seq)
  
  if (seq_length < 9) return(NULL)
  
  starts <- 1:(seq_length - 8)
  
  peptides <- vapply(
    starts,
    function(i) substr(seq, i, i + 8),
    character(1)
  )
  
  tibble(
    peptide = peptides,
    start = starts,
    end = starts + 8,
    uniprot_id = uid
  )
}

df_all_9mers <- df_fasta |>
  mutate(
    peptides = map2(sequence, uniprot_id, generate_9mers)
  ) |>
  select(peptides) |>
  unnest(peptides)

# ------------------------------------------------------------
# CLEAN POSITIVES — keep only peptides actually found in protein
# ------------------------------------------------------------

df_pos_9_clean <- df_pos_9 |>
  semi_join(
    df_all_9mers |> select(peptide, uniprot_id),
    by = c("peptide", "uniprot_id")
  ) |>
  distinct(peptide, uniprot_id, .keep_all = TRUE)

# Diagnostics
cat("Original positives:", nrow(df_pos_9), "\n")
cat("Valid positives in FASTA:", nrow(df_pos_9_clean), "\n")
cat("Removed (not found in protein):", 
    nrow(df_pos_9) - nrow(df_pos_9_clean), "\n")

# ------------------------------------------------------------
# Generate negatives
# ------------------------------------------------------------

df_neg_9 <- df_all_9mers |>
  anti_join(
    df_pos_9_clean |> select(peptide, uniprot_id),
    by = c("peptide", "uniprot_id")
  ) |>
  mutate(label = 0)

# Add labels to positives
df_pos_9_clean <- df_pos_9_clean |>
  mutate(label = 1)


set.seed(42)

df_neg_sampled <- df_neg_9 |>
  group_by(uniprot_id) |>
  group_modify(~ {
    uid   <- .y$uniprot_id
    n_pos <- sum(df_pos_9_clean$uniprot_id == uid)
    n_neg <- min(10 * n_pos, nrow(.x))
    slice_sample(.x, n = n_neg)
  }) |>
  ungroup()

df_model <- bind_rows(
  df_pos_9_clean |> select(peptide, start, end, uniprot_id, label),
  df_neg_sampled  |> select(peptide, start, end, uniprot_id, label)
)

cat("=== Dataset Summary ===\n")
cat("Positives (df_pos_9_clean):", nrow(df_pos_9_clean), "×", ncol(df_pos_9_clean), "\n")
cat("Negatives (df_neg_sampled):", nrow(df_neg_sampled), "×", ncol(df_neg_sampled), "\n")
cat("Final model data (df_model):", nrow(df_model), "×", ncol(df_model), "\n")