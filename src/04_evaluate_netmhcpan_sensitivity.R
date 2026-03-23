library(tidyverse)
source("src/functions.R")
set_working_directory()
library(ggplot2)

# 1. Load IEDB positives 
df_raw <- read_csv("data/processed/pos_EL_all_epitopes_hla0201.csv")

df_iedb_pos <- df_raw |>
  filter(pep_length >= 8, pep_length <= 14) |>
  filter(uniprot_id != "O60361")  # deleted from SwissProt in 2026_01

write_csv(df_iedb_pos, "data/processed/pos_EL_8-to-14mers_epitopes_hla0201.csv")
cat("IEDB positives:", nrow(df_iedb_pos), "\n")


# 2. Protein lookup table 
df_protein_lookup <- df_iedb_pos |>
  select(uniprot_id, source_molecule, molecule_parent) |>
  slice_head(n = 1, by = uniprot_id)

write_csv(df_protein_lookup, "data/processed/protein_lookup.csv")
cat("Unique proteins:", nrow(df_protein_lookup), "\n")


# 3. Load NetMHCpan predictions (binders only) 
cat("Loading NetMHCpan binders...\n")
df_netmhcpan_raw <- load_netmhcpan_batches(binders_only = TRUE)
cat("Total binders loaded:", nrow(df_netmhcpan_raw), "\n")


# 4. Parse and clean binders 
df_netmhcpan_binders <- df_netmhcpan_raw |>
  mutate(HLA = "HLA-A02:01") |>
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
  relocate(end,        .after = start)   |>
  relocate(id,         .after = binder)  |>
  select(-c(id, core, icore, score, ave)) |>
  left_join(df_protein_lookup, by = "uniprot_id")

cat("Parsed binders:", nrow(df_netmhcpan_binders), "\n")


# 5. Compare NetMHCpan vs IEDB 
df_overlap        <- df_netmhcpan_binders |> semi_join(df_iedb_pos, by = c("peptide", "uniprot_id"))
df_netmhcpan_only <- df_netmhcpan_binders |> anti_join(df_iedb_pos, by = c("peptide", "uniprot_id"))
df_iedb_only      <- df_iedb_pos          |> anti_join(df_netmhcpan_binders, by = c("peptide", "uniprot_id"))

cat("\n=== Comparison Summary ===\n")
cat("Experimentally confirmed (IEDB):          ", nrow(df_iedb_pos), "\n")
cat("Predicted binders (NetMHCpan):            ", nrow(df_netmhcpan_binders), "\n")
cat("Overlap (both):                           ", nrow(df_overlap), "\n")
cat("NetMHCpan only (predicted, not confirmed):", nrow(df_netmhcpan_only), "\n")
cat("IEDB only (confirmed, not predicted):     ", nrow(df_iedb_only), "\n")
cat("Sensitivity (IEDB peptides recovered):    ",
    round(nrow(df_overlap) / nrow(df_iedb_pos) * 100, 1), "%\n")


# 6. Save split outputs 
write_csv(df_netmhcpan_binders, "data/processed/netmhcpan_binders.csv")
write_csv(df_netmhcpan_only,    "data/processed/netmhcpan_only.csv")
write_csv(df_iedb_only,         "data/processed/iedb_only.csv")

cat("\n✅ Saved split data to data/processed/\n")


# 7. Confusion Matrix Calculation & Plot 
TP <- nrow(df_overlap)         
FP <- nrow(df_netmhcpan_only)  
FN <- nrow(df_iedb_only)       

# Calculate true negatives directly from source files to prevent memory crashes
total_peptides <- get_netmhcpan_total()
TN <- total_peptides - nrow(df_netmhcpan_binders) - FN

# Derived metrics
sensitivity  <- TP / (TP + FN)          
specificity  <- TN / (TN + FP)
precision    <- TP / (TP + FP)
f1           <- 2 * (precision * sensitivity) / (precision + sensitivity)

df_cm <- tibble(
  Predicted  = factor(
    c("Binder", "Binder", "Non-Binder", "Non-Binder"),
    levels = c("Binder", "Non-Binder")
  ),
  Actual = factor(
    c("IEDB Positive", "IEDB Negative", "IEDB Positive", "IEDB Negative"),
    levels = c("IEDB Positive", "IEDB Negative")
  ),
  Count  = c(TP, FP, FN, TN),
  quad   = c("TP", "FP", "FN", "TN")
)

df_cm <- df_cm |>
  mutate(display = paste0(quad, "\n", scales::comma(Count)))

quad_colours <- c(
  "TP" = "#2ecc71",
  "TN" = "#3498db",
  "FP" = "#e74c3c",
  "FN" = "#e67e22"
)

sum_binder     <- TP + FP
sum_nonbinder  <- FN + TN
sum_iedb_pos   <- TP + FN
sum_iedb_neg   <- FP + TN

col_sums <- c(
  "Binder"     = paste0("Total: ", scales::comma(sum_binder)),
  "Non-Binder" = paste0("Total: ", scales::comma(sum_nonbinder))
)

row_sums <- c(
  "IEDB Positive" = paste0("Total:\n", scales::comma(sum_iedb_pos)),
  "IEDB Negative" = paste0("Total:\n", scales::comma(sum_iedb_neg))
)

p1 <- ggplot(df_cm, aes(x = Predicted, y = Actual, fill = quad)) +
  geom_tile(colour = "white", linewidth = 2) +
  geom_text(aes(label = display), size = 5, fontface = "bold", colour = "white") +
  scale_fill_manual(
    values = quad_colours,
    labels = c("TP" = "True Pos.", "TN" = "True Neg.", 
               "FP" = "False Pos.", "FN" = "False Neg.")
  ) +
  scale_x_discrete(position = "top", sec.axis = dup_axis(name = NULL, labels = col_sums)) +
  scale_y_discrete(sec.axis = dup_axis(name = NULL, labels = row_sums)) +
  labs(
    title    = "Peptide Binders: NetMHCpan vs IEDB",
    subtitle = paste0(
      "HLA-A*02:01  |  Binder threshold: Rank < 2%\n",
      "Sensitivity: ", round(sensitivity * 100, 1), "%  |  ",
      "Precision: ",   round(precision  * 100, 1), "%  |  ",
      "F1: ",          round(f1, 3)
    ),
    x    = "NetMHCpan Prediction",
    y    = "",
    fill = NULL
  ) +
  theme_bw(base_size = 13) +
  theme(
    legend.position  = "bottom",
    legend.box.margin = margin(r = 80),
    plot.title       = element_text(margin = margin(l = 20, b = 5)),
    plot.subtitle    = element_text(size = 10, colour = "grey40", margin = margin(l = 20, b = 10)),
    axis.text        = element_text(size = 12, face = "bold"),
    axis.text.x.bottom = element_text(size = 11, face = "italic", colour = "grey30"),
    axis.text.y.right  = element_text(size = 11, face = "italic", colour = "grey30", hjust = 0),
    axis.title.x.top   = element_text(face = "bold", margin = margin(b = 10)),
    panel.grid       = element_blank(),
    plot.title.position = "plot",
    plot.margin = margin(t = 10, r = 10, b = 10, l = 5) 
  )

print(p1)


# 8. Build Labelled Machine Learning Dataset 
df_positives <- df_overlap |> 
  mutate(label = 1)

# Downsample negatives to 1:5 ratio based on the overlapping positives
n_pos <- nrow(df_positives)
n_neg_target <- n_pos * 5

set.seed(42)
df_negatives <- df_netmhcpan_only |>
  slice_sample(n = n_neg_target) |>
  mutate(label = 0)

cat("\nPositives:", nrow(df_positives), "\n")
cat("Negatives:", nrow(df_negatives), "\n")
cat("Ratio neg/pos:", round(nrow(df_negatives) / nrow(df_positives), 1), "\n")

df_combined <- bind_rows(df_positives, df_negatives) |>
  select(-c(rank, hla, binder))

write_csv(df_combined, "data/processed/df_combined_pos_and_neg.csv")


# 9. Peptide Length Distribution Plot 
df_lengths <- bind_rows(
  df_positives      |> mutate(set = "Positives (Overlap)"),
  df_netmhcpan_only |> mutate(set = "Negatives (NetMHCpan only)")
)

df_lengths_prop <- df_lengths |>
  count(set, pep_length) |>         
  group_by(set) |>                  
  mutate(pct = n / sum(n)) |>       
  ungroup()

p2 <- ggplot(df_lengths_prop, aes(x = pep_length, y = pct, fill = set)) +
  geom_col(position = "dodge", colour = "black", linewidth = 0.2) + 
  scale_fill_manual(values = c("Positives (Overlap)" = "#2ecc71",
                               "Negatives (NetMHCpan only)" = "#e74c3c")) +
  scale_x_continuous(breaks = 8:14) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) + 
  labs(
    title    = "Peptide Length Distribution (Normalized)",
    subtitle = "Comparing the proportion of lengths within each class",
    x        = "Peptide Length",
    y        = "Percentage of Class",
    fill     = NULL
  ) +
  theme_bw(base_size = 13) +
  theme(
    legend.position  = "top",
    panel.grid.minor = element_blank(),
    plot.title.position = "plot"
  )

print(p2)


# 10. Save Plots 
ggsave("results/confusion_matrix.png",    p1, width = 7, height = 5, dpi = 150)
ggsave("results/peptide_length_dist.png", p2, width = 7, height = 5, dpi = 150)
cat("\n✅ Saved all plots to results/ \n")