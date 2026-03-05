library(tidyverse)
source("src/functions.R")
set_working_directory()

# Load raw data
df_table4_raw    <- read_csv("data/raw/assarsson_2009_initial.csv")
df_assarsson_raw <- read_csv("data/raw/epitope_table_export_1770817956.csv")

# Clean IEDB export
df_assarsson_raw_clean <- df_assarsson_raw |>
  select(where(~ !all(is.na(.)))) |>
  select(-c(
    "Epitope ID - IEDB IRI",
    "Epitope - Source Molecule IRI",
    "Epitope - Source Organism IRI",
    "Epitope - Species IRI",
    "Epitope - Source Organism",
    "Epitope - Species",
    "Epitope - Object Type"
  ))

colnames(df_assarsson_raw_clean) <- sub("^Epitope - ", "", colnames(df_assarsson_raw_clean))

# Merge and clean
df_merged <- df_table4_raw |>
  left_join(df_assarsson_raw_clean, by = c("Sequence" = "Name")) |>
  mutate(across(everything(), as.character)) |>
  mutate(
    notes = case_when(
      str_detect(Infection,   "[ef]$") ~ str_extract(Infection,   "[ef]$"),
      str_detect(Immunogenic, "[ef]$") ~ str_extract(Immunogenic, "[ef]$"),
      TRUE ~ NA_character_
    )
  ) |>
  mutate(
    Infection   = str_remove(Infection,   "[ef]$"),
    Immunogenic = str_remove(Immunogenic, "[ef]$")
  ) |>
  mutate(across(everything(), ~ na_if(.x, "?"))) |>
  mutate(
    Infection           = as.numeric(Infection),
    Immunogenic         = as.numeric(Immunogenic),
    `Starting Position` = as.integer(`Starting Position`),
    `Ending Position`   = as.integer(`Ending Position`),
    Classification      = as.factor(Classification)
  ) |>
  mutate(
    Affinity = if_else(Affinity == "<1", "0.1", Affinity),
    Affinity = as.numeric(Affinity)
  ) |>
  rename_with(tolower) |>
  rename(
    start_pos      = `starting position`,
    end_pos        = `ending position`,
    mol_source     = `source molecule`,
    mol_parent     = `molecule parent`,
    mol_parent_iri = `molecule parent iri`,
    orf            = `orf name`
  ) |>
  mutate(
    processing_disclosed = if_else(processed == "ND", "ND", "Disclosed"),
    processed            = na_if(processed, "ND"),
    processed            = as.numeric(processed),
    pep_length           = nchar(sequence)
  ) |>
  relocate(processing_disclosed, .after = processed) |>
  mutate(
    bio_meaning = case_when(
      classification == "Dominant"    ~ "Passes ALL filters → recognized during infection",
      classification == "Subdominant" ~ "Immunogenic but NOT recognized during infection",
      classification == "Cryptic"     ~ "Binds MHC but fails immunogenicity/processing",
      classification == "Negative"    ~ "Fails at binding or immunogenicity"
    ),
    naturally_processed = case_when(
      classification == "Dominant"    ~ "Yes",
      classification == "Subdominant" ~ "Partially",
      classification == "Cryptic"     ~ "No",
      classification == "Negative"    ~ "No"
    ),
    dc_processed = if_else(
      classification %in% c("Dominant", "Subdominant"),
      "Processed by DC", "Not processed"
    )
  )

# Extract protein info lookup table
colnames(df_assarsson_raw) <- sub("^Epitope - ", "", colnames(df_assarsson_raw))

df_protein_info <- df_assarsson_raw |>
  rename_with(tolower) |>
  rename(
    mol_source     = `source molecule`,
    mol_source_iri = `source molecule iri`,
    mol_parent     = `molecule parent`,
    mol_parent_iri = `molecule parent iri`
  ) |>
  select(mol_source, mol_source_iri, mol_parent, mol_parent_iri) |>
  mutate(protein_id = str_extract(mol_parent_iri, "[^/]+$")) |>
  distinct(protein_id, .keep_all = TRUE)

# Split into mapped (n=72) and unmapped subsets
df_merged_assarsson72 <- df_merged |>
  filter(!is.na(start_pos) & !is.na(end_pos))

df_merged_leftover <- df_merged |>
  filter(is.na(start_pos) & is.na(end_pos))

# Export
write_csv(df_protein_info, "data/processed/assarsson_protein_info.csv")
write_csv(df_merged,       "data/processed/merged_assarsson_data.csv")

df_merged_assarsson72 |>
  rename(epitope = sequence) |>
  export_epitopes(
    file      = "data/processed/assarsson72_aff100_epitopes.txt",
    separator = "comma"
  )

df_merged_assarsson72 |>
  rename(epitope = sequence) |>
  export_epitopes(
    file      = "data/processed/iedb_assarsson72_aff100_epitopes.txt",
    separator = "newline"
  )