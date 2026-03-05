# Strong binder threshold: %Rank < 0.5
# Weak binder threshold:   %Rank < 2.0

library(tidyverse)
source("src/functions.R")
set_working_directory()

# Load protein info
df_protein_info <- read_csv("data/processed/assarsson_protein_info.csv")

# Parse NetMHCpan CSV output (EL mode)
parse_netmhcpan_csv <- function(filepath, ba_mode = FALSE) {
  df_raw   <- read_csv(filepath, col_names = FALSE, show_col_types = FALSE)
  hla_name <- df_raw |> slice(1) |> pull(X4)
  
  col_names <- if (ba_mode) {
    c("Pos", "Peptide", "ID", "core", "icore", "Score", "Rank", "BA_score", "BA_Rank", "Ave", "NB")
  } else {
    c("Pos", "Peptide", "ID", "core", "icore", "Score", "Rank", "Ave", "NB")
  }
  
  df_raw |>
    slice(-(1:2)) |>
    set_names(col_names) |>
    mutate(HLA = hla_name, .before = 1)
}

# Clean parsed NetMHCpan output into binder dataframe
clean_netmhcpan <- function(df, df_protein_info) {
  df |>
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
      end_pos    = pos + pep_length - 1
    ) |>
    rename(start_pos = pos) |>
    relocate(pep_length, .after = peptide) |>
    relocate(end_pos,    .after = start_pos) |>
    mutate(
      protein_id     = str_split_i(id, "_", 2),
      mol_parent_iri = paste0("http://www.uniprot.org/uniprot/", protein_id)
    ) |>
    relocate(id, .after = mol_parent_iri) |>
    left_join(df_protein_info, by = "protein_id") |>
    select(-c(id, rank, core, icore, mol_source_iri))
}

# EL mode
df_binders <- parse_netmhcpan_csv("data/raw/NetMHCpan_assarsson.csv") |>
  clean_netmhcpan(df_protein_info) |>
  select(start_pos, end_pos, peptide, pep_length, protein_id, binder, everything()) |>
  rename(epitope = peptide)

# BA mode — parse affinity (nM) from web txt output
df_aff_nM <- parse_netmhcpan_web_txt("data/raw/NetMHCpan_BAmode_assarsson_alltext.txt") |>
  rename_with(tolower) |>
  select(peptide, aff_nm)

df_binders_ba <- parse_netmhcpan_csv("data/raw/NetMHCpan_BAmode_assarsson.csv", ba_mode = TRUE) |>
  clean_netmhcpan(df_protein_info) |>
  left_join(df_aff_nM, by = "peptide") |>
  rename(affinity = aff_nm, epitope = peptide) |>
  select(start_pos, end_pos, epitope, pep_length, protein_id, affinity, binder, everything())

df_binders_aff100nm <- df_binders_ba |>
  filter(affinity < 100)

# Export
export_epitopes(df_binders_ba,      file = "data/processed/netmhcpan_epitopes.txt",         separator = "comma")
export_epitopes(df_binders_ba,      file = "data/processed/iedb_netmhcpan_epitopes.txt",     separator = "newline")
export_epitopes(df_binders_aff100nm, file = "data/processed/netmhcpan_aff100_epitopes.txt",  separator = "comma")
export_epitopes(df_binders_aff100nm, file = "data/processed/iedb_netmhcpan_aff100_epitopes.txt", separator = "newline")