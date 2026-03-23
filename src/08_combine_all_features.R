library(tidyverse)
source("src/functions.R")
set_working_directory()

cat("\n=== Combining NetSurfP and AlphaFold Features ===\n")

# 1. Load the two parallel datasets
df_nsp3 <- read_csv("data/processed/epitopes_pos_and_neg_features_with_nsp3.csv", show_col_types = FALSE)
df_af   <- read_csv("data/processed/alphafold_plddt_features.csv", show_col_types = FALSE)

# 2. Drop redundant coordinate columns from AF so we don't get duplicates (.x / .y)
df_af_clean <- df_af |> 
  select(-pep_start, -pep_end)

# 3. Join them together using the peptide and protein ID
df_final_ml <- df_nsp3 |> 
  left_join(df_af_clean, by = c("peptide", "uniprot_id"))

# 4. Save the ultimate ML dataset!
write_csv(df_final_ml, "data/processed/final_ml_dataset.csv")

cat("✅ Successfully joined features!\n")
cat("Final dataset has", nrow(df_final_ml), "rows and", ncol(df_final_ml), "columns.\n")
cat("Saved to: data/processed/final_ml_dataset.csv\n")