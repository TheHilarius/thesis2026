library(tidyverse)
library(stringr)
source("src/functions.R")
set_working_directory()

# load and clean data
df_raw <- read.csv("data/iedb_553_EL_epitopes.csv", stringsAsFactors = FALSE)
colnames(df_raw) <- clean_names(colnames(df_raw))

df_clean <- df_raw |>
  select(where(~ !all(is.na(.)))) |>
  mutate(across(where(is.character), ~ na_if(.x, ""))) |>
  mutate(pep_length = nchar(epitope_name))

# create epitope dataframe
df_epitope <- df_clean |>
  select(epitope_name, epitope_molecule_parent, 
         epitope_molecule_parent_iri, epitope_source_organism, pep_length) |>
  distinct() |>
  rename(peptide = epitope_name, protein = epitope_molecule_parent,
         protein_iri = epitope_molecule_parent_iri, organism = epitope_source_organism) |>
  mutate(uniprot_id = str_extract(protein_iri, "[^/]+$"))

# lookup tables
df_protein_info <- df_epitope |>
  select(protein, protein_iri, organism, uniprot_id) |>
  distinct(uniprot_id, .keep_all = TRUE)

df_mhc_info <- df_clean |>
  select(peptide = epitope_name, mhc_restriction_name) |>
  distinct(peptide, .keep_all = TRUE)

# positives
df_pos <- df_epitope |>
  distinct(peptide, uniprot_id, .keep_all = TRUE) |>
  mutate(label = 1)

cat("\nPositive peptide lengths:\n")
print(table(df_pos |> pull(pep_length)))

# load fasta
df_fasta <- read_fasta_df("data/combined.fasta") |>
  mutate(uniprot_id = str_extract(header, "(?<=\\|)[A-Z0-9]+(?=\\|)")) |>
  filter(uniprot_id %in% (df_pos |> pull(uniprot_id) |> unique()))

# generate 9-mers
df_all_9mers <- df_fasta |>
  mutate(peptides = map2(sequence, uniprot_id, ~ generate_kmers(.x, .y, lengths = 9))) |>
  select(peptides) |>
  unnest(peptides)

# clean positives
df_pos_clean <- df_pos |>
  filter(map2_lgl(peptide, uniprot_id, ~ verify_peptide_in_protein(.x, .y, df_fasta))) |>
  distinct(peptide, uniprot_id, .keep_all = TRUE) |>
  mutate(label = 1)

# generate negatives (10:1 ratio)
df_neg_all <- df_all_9mers |>
  anti_join(df_pos_clean |> select(peptide, uniprot_id), by = c("peptide", "uniprot_id")) |>
  mutate(label = 0)

set.seed(42)

df_neg_sampled <- df_neg_all |>
  group_by(uniprot_id) |>
  group_modify(~ {
    n_pos <- df_pos_clean |> filter(uniprot_id == .y$uniprot_id) |> nrow()
    n_neg <- min(10 * n_pos, nrow(.x))
    if (n_neg > 0) slice_sample(.x, n = n_neg) else .x[0, ]
  }) |>
  ungroup()

# combine
df_model <- bind_rows(
  df_pos_clean |> select(peptide, uniprot_id, pep_length) |> mutate(label = 1),
  df_neg_sampled |> select(peptide, uniprot_id, pep_length, label)
)

# add flanking regions
df_model <- df_model |>
  add_flanking_regions(df_fasta, flank_size = 8) |>
  filter(peptide_found)

# add metadata
df_model <- df_model |>
  left_join(df_protein_info |> select(uniprot_id, protein, organism), by = "uniprot_id") |>
  left_join(df_mhc_info, by = "peptide")

# summary
cat("\n--- Dataset Summary ---\n")
cat("Total:", nrow(df_model), "\n")
cat("Positives:", df_model |> filter(label == 1) |> nrow(), "\n")
cat("Negatives:", df_model |> filter(label == 0) |> nrow(), "\n")
cat("Ratio:", (df_model |> filter(label == 0) |> nrow()) / (df_model |> filter(label == 1) |> nrow()) |> round(1), ":1\n")

# save
df_model_final <- df_model |>
  select(peptide, uniprot_id, label, pep_length, n_flank, c_flank, 
         full_context, start, end, protein, organism, mhc_restriction_name)

write_csv(df_model_final, "data/df_model_training.csv")