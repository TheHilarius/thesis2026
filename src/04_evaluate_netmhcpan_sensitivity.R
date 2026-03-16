library(tidyverse)
source("src/functions.R")
set_working_directory()

df_raw <- read_csv("data/processed/pos_EL_all_epitopes_hla0201.csv")
# Load IEDB 200K EL epitopes, filter to 8-14mers
df_iedb_comp <- df_raw |>
  filter(pep_length >= 8, pep_length <= 14) |>
  # remove O60361 because it was deleted from SwissProt in 2026_01 and fasta missing
  filter(uniprot_id != "O60361")

write_csv(df_iedb_comp, "data/processed/pos_EL_8-to-14mers_epitopes_hla0201.csv")

  df_protein_lookup <- df_iedb_comp |>
  select(uniprot_id, source_molecule, molecule_parent) |>
  slice_head(n = 1, by = uniprot_id)

write_csv(df_protein_lookup, "data/processed/protein_lookup.csv")
# Next step: run src/export_fasta_files_from_iedb.py on data/processed/pos_EL_all_epitopes_hla0201.csv
# Results will be in data/raw/fasta/fasta_all_hla0201

# Load and parse NetMHCpan predictions (Assarsson proteins)
df_netmhcpan_raw <- read_csv("data/raw/NetMHCpan_predicted_results.csv",
                             col_names = FALSE,
                             show_col_types = FALSE)

hla_name    <- df_netmhcpan_raw |> slice(1) |> pull(X4)
real_colnames <- c("Pos", "Peptide", "ID", "core", "icore", "Score", "Rank", "Ave", "NB")

df_netmhcpan_binders <- df_netmhcpan_raw |>
  slice(-(1:2)) |>
  set_names(real_colnames) |>
  mutate(HLA = hla_name, .before = 1) |>
  filter(NB == 1) |>
  mutate(binder = case_when(
    Rank < 0.5 ~ "SB",
    Rank < 2   ~ "WB",
    TRUE       ~ "NB"
  )) |>
  select(-NB) |>
  rename_with(tolower) |>
  mutate(
    pep_length = nchar(peptide),
    pos        = as.numeric(pos),
    end        = pos + pep_length - 1,
    uniprot_id = str_split_i(id, "_", 2)
  ) |>
  rename(start = pos) |>
  relocate(pep_length, .after = peptide) |>
  relocate(end, .after = start) |>
  relocate(id, .after = binder) |>
  select(-c(id, core, icore, score, ave)) |>
  left_join(df_protein_lookup, by = "uniprot_id")

# Compare NetMHCpan predictions vs IEDB confirmed epitopes
df_overlap        <- df_netmhcpan_binders |> semi_join(df_iedb_comp, by = c("peptide", "uniprot_id"))
df_netmhcpan_only <- df_netmhcpan_binders |> anti_join(df_iedb_comp, by = c("peptide", "uniprot_id"))
df_iedb_only      <- df_iedb_comp         |> anti_join(df_netmhcpan_binders, by = c("peptide", "uniprot_id"))

cat("=== Comparison Summary ===\n")
cat("Experimentally confirmed (IEDB):          ", nrow(df_iedb_comp), "\n")
cat("Predicted binders (netMHCpan):            ", nrow(df_netmhcpan_binders), "\n")
cat("Overlap (both):                           ", nrow(df_overlap), "\n")
cat("netMHCpan only (predicted, not confirmed):", nrow(df_netmhcpan_only), "\n")
cat("IEDB only (confirmed, not predicted):     ", nrow(df_iedb_only), "\n")
cat("Sensitivity (IEDB peptides recovered):    ",
    round(nrow(df_overlap) / nrow(df_iedb_comp) * 100, 1), "%\n")