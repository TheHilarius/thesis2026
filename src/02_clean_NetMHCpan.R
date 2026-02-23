#Threshold for strong prediction: % Rank   0.5
#Threshold for weak prediction: % Rank  2

library(tidyverse)

current_user <- Sys.info()[["user"]]

if (current_user == "olive") {
  setwd("C:/Users/olive/Documents/R/special_course_spring2026")
} else if (current_user == "mj607") {
  setwd("//wsl$/Ubuntu/home/hilarius/special_course_spring2026")
} else {
  stop("Unknown user. Please set working directory manually.")
}


library(tidyverse)

df_protein_info <- read.csv("data/assarsson_protein_info.csv")

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
  

