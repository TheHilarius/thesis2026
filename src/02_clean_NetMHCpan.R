#Threshold for strong prediction: % Rank   0.5
#Threshold for weak prediction: % Rank  2
library(tidyverse)
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

library(tidyverse)

df_protein_info <- read.csv("data/assarsson_protein_info.csv")

# Without BA mode
df_raw <- read_csv("data/NetMHCpan_assarsson.csv", col_names = FALSE, show_col_types = FALSE)
hla_name <- df_raw %>% slice(1) %>% pull(X4)
real_colnames <- c("Pos", "Peptide", "ID", "core", "icore", "Score", "Rank", "Ave", "NB")

df_netmhcpan <- df_raw %>%
  slice(-(1:2)) %>%
  set_names(real_colnames) %>%
  mutate(HLA = hla_name, .before = 1)

df_binders <- df_netmhcpan |>
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
  mutate(end_pos = pos + pep_length - 1) |> # MINUS 1 OR NOT?????
  rename(start_pos = pos) |>
  relocate(end_pos, .after = start_pos) |>
  mutate(
    protein_id = str_split_i(id, "_", 2),
    mol_parent_iri = paste0("http://www.uniprot.org/uniprot/", protein_id)
    ) |>
  relocate(id, .after = mol_parent_iri) |>
  left_join(df_protein_info, by = "protein_id") |>
  select(-c(id,rank,core,icore,mol_source_iri)) |>
  select(
    start_pos, end_pos, peptide, pep_length, protein_id, binder,
    everything()
  ) |>
  rename(epitope = peptide)
  
# With BA mode
df_aff_nM_raw <- parse_netmhcpan_web_txt("data/NetMHCpan_BAmode_assarsson_alltext.txt")
df_aff_nM <- df_aff_nM_raw |>
  rename_with(tolower) |>
  mutate(protein_id = str_split_i(identity, "_", 2)) |>
  select(c("peptide", "aff_nm","protein_id")) |>
  select(-protein_id)

df_ba_raw <- read_csv("data/NetMHCpan_BAmode_assarsson.csv", col_names = FALSE, show_col_types = FALSE)
hla_name_ba <- df_ba_raw %>% slice(1) %>% pull(X4)
real_colnames_ba <- c("Pos", "Peptide", "ID", "core", "icore",
                      "Score", "Rank",
                      "BA_score", "BA_Rank",
                      "Ave", "NB")

df_netmhcpan_ba <- df_ba_raw %>%
  slice(-(1:2)) %>%
  set_names(real_colnames_ba) %>%
  mutate(HLA = hla_name_ba, .before = 1)

df_binders_ba <- df_netmhcpan_ba |>
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
  mutate(end_pos = pos + pep_length - 1) |> # MINUS 1 OR NOT?????
  rename(start_pos = pos) |>
  relocate(end_pos, .after = start_pos) |>
  mutate(
    protein_id = str_split_i(id, "_", 2),
    mol_parent_iri = paste0("http://www.uniprot.org/uniprot/", protein_id)
  ) |>
  relocate(id, .after = mol_parent_iri) |>
  left_join(df_protein_info, by = "protein_id") |>
  select(-c(id,rank,core,icore,mol_source_iri)) |>
  left_join(df_aff_nM, by =c("peptide" = "peptide")) |>
  rename(
    affinity = aff_nm,
    epitope = peptide) |>
  select(
    start_pos, end_pos, epitope, pep_length, protein_id, affinity, binder,
    everything()
  )

df_binders_aff100nm <- df_binders_ba |> 
  filter(affinity < 100)

export_epitopes(
  df_binders_ba,
  file = "data/netmhcpan_epitopes.txt",
  separator = "comma"
)
export_epitopes(
  df_binders_ba,
  file = "data/iedb_netmhcpan_epitopes.txt",
  separator = "newline"
)
export_epitopes(
  df_binders_aff100nm,
  file = "data/netmhcpan_aff100_epitopes.txt",
  separator = "comma"
)
export_epitopes(
  df_binders_aff100nm,
  file = "data/iedb_netmhcpan_aff100_epitopes.txt",
  separator = "newline"
)


