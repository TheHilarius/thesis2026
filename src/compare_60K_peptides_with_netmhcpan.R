library(tidyverse)
library(stringr)
source("src/functions.R")
set_working_directory()

df_netmhc_comp <- read.csv("data/pos_EL_all_epitopes_hla0201.csv", 
                   sep = ",", stringsAsFactors = FALSE) |>
  filter(pep_length >= 8, pep_length <= 14)

write_csv(df_netmhc_comp, "data/pos_EL_8-to-14mers_epitopes_hla0201.csv")

df_protein_lookup <- df_netmhc_comp |>
  select(uniprot_id, source_molecule, molecule_parent) |>
  slice_head(n = 1, by = uniprot_id)

# Use src/export_fasta_files_from_iedb.py on data/pos_EL_all_epitopes_hla0201.csv
# results are in data/fasta_all_hla0201 and data/combined_all_hla0201.csv

df_nehmhcpan_raw <- read_csv("data/NetMHCpan_assarsson.csv", 
                             col_names = FALSE, 
                             show_col_types = FALSE)
hla_name <- df_nehmhcpan_raw %>% slice(1) %>% pull(X4)
real_colnames <- c("Pos", "Peptide", "ID", "core", "icore", "Score", "Rank", "Ave", "NB")

df_nehmhcpan_raw <- df_nehmhcpan_raw %>%
  slice(-(1:2)) %>%
  set_names(real_colnames) %>%
  mutate(HLA = hla_name, .before = 1)

df_netmhcpan_binders <- df_nehmhcpan_raw |>
  filter(NB == 1) |>
  mutate(binder = case_when(
    Rank < 0.5 ~ "SB",
    Rank < 2   ~ "WB",
    TRUE       ~ "NB"
  )) |>
  select(-NB) |>
  rename_with(tolower) |>
  mutate(pep_length = nchar(peptide)) |>
  mutate(pos = as.numeric(pos)) |>
  relocate(pep_length, .after = peptide) |> 
  mutate(end = pos + pep_length - 1) |> 
  rename(start = pos) |>
  relocate(end, .after = start) |>
  mutate(
    uniprot_id = str_split_i(id, "_", 2)
  ) |>
  relocate(id, .after = binder) |>
  select(-c(id,core,icore,score,ave)) |>
  left_join(df_protein_lookup, by = "uniprot_id")


# -----------
# Comparison 
# -----------
df_overlap <- df_netmhcpan_binders |>
  semi_join(df_netmhc_comp, by = c("peptide", "uniprot_id"))

df_netmhcpan_only <- df_netmhcpan_binders |>
  anti_join(df_netmhc_comp, by = c("peptide", "uniprot_id"))


df_iedb_only <- df_netmhc_comp |>
  anti_join(df_netmhcpan_binders, by = c("peptide", "uniprot_id"))

# ---------
# Summary
# ---------
cat("=== Comparison Summary ===\n")
cat("Experimentally confirmed (IEDB):", nrow(df_netmhc_comp), "\n")
cat("Predicted binders (netMHCpan):  ", nrow(df_netmhcpan_binders), "\n")
cat("\n")
cat("Overlap (both):                 ", nrow(df_overlap), "\n")
cat("netMHCpan only (predicted, not confirmed):", nrow(df_netmhcpan_only), "\n")
cat("IEDB only (confirmed, not predicted):     ", nrow(df_iedb_only), "\n")
cat("\n")
cat("Sensitivity (IEDB peptides recovered by netMHCpan):",
    round(nrow(df_overlap) / nrow(df_netmhc_comp) * 100, 1), "%\n")
