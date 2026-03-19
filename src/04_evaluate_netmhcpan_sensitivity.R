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
# df_positives should be both positives by netmhcpan and iedb
# Think confusion matrix

# and possibly make 

cat("\n=== Comparison Summary ===\n")
cat("Experimentally confirmed (IEDB):          ", nrow(df_iedb_pos), "\n")
cat("Predicted binders (NetMHCpan):            ", nrow(df_netmhcpan_binders), "\n")
cat("Overlap (both):                           ", nrow(df_overlap), "\n")
cat("NetMHCpan only (predicted, not confirmed):", nrow(df_netmhcpan_only), "\n")
cat("IEDB only (confirmed, not predicted):     ", nrow(df_iedb_only), "\n")
cat("Sensitivity (IEDB peptides recovered):    ",
    round(nrow(df_overlap) / nrow(df_iedb_pos) * 100, 1), "%\n")



# ── Confusion Matrix ──────────────────────────────────────────────────────────

# Define the four quadrants
# TP: predicted binder AND confirmed by IEDB
# FP: predicted binder but NOT confirmed by IEDB
# FN: NOT predicted as binder but confirmed by IEDB
# TN: NOT predicted as binder AND NOT in IEDB
#     ≈ total raw predictions − all binders − IEDB-only peptides

TP <- nrow(df_overlap)
FP <- nrow(df_netmhcpan_only)
FN <- nrow(df_iedb_only)
TN <- nrow(df_netmhcpan_raw) - nrow(df_netmhcpan_binders) - nrow(df_iedb_only)

cat("\n=== Confusion Matrix Counts ===\n")
cat("TP:", scales::comma(TP), "\n")
cat("FP:", scales::comma(FP), "\n")
cat("FN:", scales::comma(FN), "\n")
cat("TN:", scales::comma(TN), "(approximated from raw predictions)\n")

# Derived metrics
sensitivity  <- TP / (TP + FN)          # Recall
specificity  <- TN / (TN + FP)
precision    <- TP / (TP + FP)
f1           <- 2 * (precision * sensitivity) / (precision + sensitivity)

cat("\n=== Performance Metrics ===\n")
cat("Sensitivity (Recall):  ", round(sensitivity * 100, 2), "%\n")
cat("Specificity:           ", round(specificity * 100, 2), "%\n")
cat("Precision (PPV):       ", round(precision  * 100, 2), "%\n")
cat("F1 Score:              ", round(f1,              4), "\n")

# Build a tidy data frame for ggplot
df_cm <- tibble(
  Predicted  = factor(
    c("Binder",        "Binder",
      "Non-Binder",    "Non-Binder"),
    levels = c("Binder", "Non-Binder")
  ),
  Actual = factor(
    c("IEDB Positive", "IEDB Negative",
      "IEDB Positive", "IEDB Negative"),
    levels = c("IEDB Positive", "IEDB Negative")   # top-left = TP
  ),
  Count  = c(TP, FP, FN, TN),
  quad   = c("TP", "FP", "FN", "TN")
)

# Annotation: show both the quadrant label and the formatted count
df_cm <- df_cm |>
  mutate(
    display = paste0(quad, "\n", scales::comma(Count))
  )

# Colour the tiles by quadrant type, not by raw count,
# because TN is ~1000× larger and would wash out the other cells.
quad_colours <- c(
  "TP" = "#2ecc71",   # green  – correct positive
  "TN" = "#3498db",   # blue   – correct negative
  "FP" = "#e74c3c",   # red    – false alarm
  "FN" = "#e67e22"    # orange – missed positive
)

p4 <- ggplot(df_cm, aes(x = Predicted, y = Actual, fill = quad)) +
  geom_tile(colour = "white", linewidth = 2) +
  geom_text(
    aes(label = display),
    size      = 5,
    fontface  = "bold",
    colour    = "white"
  ) +
  scale_fill_manual(
    values = quad_colours,
    labels = c(
      "TP" = "True Positive",
      "TN" = "True Negative",
      "FP" = "False Positive",
      "FN" = "False Negative"
    )
  ) +
  labs(
    title    = "Confusion Matrix: NetMHCpan vs IEDB",
    subtitle = paste0(
      "HLA-A*02:01  |  Binder threshold: Rank < 2%\n",
      "Sensitivity: ", round(sensitivity * 100, 1), "%  |  ",
      "Precision: ",   round(precision  * 100, 1), "%  |  ",
      "F1: ",          round(f1, 3)
    ),
    x    = "NetMHCpan Prediction",
    y    = "IEDB Ground Truth",
    fill = NULL
  ) +
  theme_bw(base_size = 13) +
  theme(
    legend.position  = "top",
    plot.subtitle    = element_text(size = 10, colour = "grey40"),
    axis.text        = element_text(size = 12, face = "bold"),
    panel.grid       = element_blank()
  )

p4

ggsave("results/confusion_matrix.png", p4, width = 7, height = 5, dpi = 150)
cat("  results/confusion_matrix.png\n")


# 6. Save outputs 
write_csv(df_netmhcpan_binders, "data/processed/netmhcpan_binders.csv")
write_csv(df_netmhcpan_only,    "data/processed/netmhcpan_only.csv")
write_csv(df_iedb_only,         "data/processed/iedb_only.csv")

cat("\n✅ Saved:\n")
cat("  data/processed/netmhcpan_binders.csv\n")
cat("  data/processed/netmhcpan_only.csv\n")
cat("  data/processed/iedb_only.csv\n")

# df_netmhcpan_raw     56,614,068  ← all predictions (intentional)
# df_netmhcpan_binders    680,723  ← NB == 1 (binders)
# df_netmhcpan_only       646,367  ← predicted but not in IEDB
# df_overlap               34,356  ← predicted AND in IEDB
# df_iedb_only             18,918  ← in IEDB but not predicted
# df_iedb_pos              53,129  ← all IEDB positives

# 7. Build labelled dataset 
df_positives <- df_iedb_overlap |> ## Change the positive list to include only the overlapping regions. 
  mutate(label = 1)

# Downsample negatives to 1:3 ratio
n_pos <- nrow(df_iedb_pos)
n_neg_target <- n_pos * 3

set.seed(42)
df_negatives <- df_netmhcpan_only |>
  slice_sample(n = n_neg_target) |>
  mutate(label = 0)

cat("\nPositives:", nrow(df_positives), "\n")
cat("Negatives:", nrow(df_negatives), "\n")
cat("Ratio neg/pos:", round(nrow(df_negatives) / nrow(df_positives), 1), "\n")


df_combined <- bind_rows(
  df_positives          |> mutate(label = 1),
  df_negatives |> mutate(label = 0)
) |>
  select(-c(rank,hla,binder))

write_csv(df_combined, "data/processed/df_combined_pos_and_neg.csv")


# Dataset composition plot 
df_composition <- tibble(
  category = c(
    "IEDB confirmed\n(Positives)",
    "NetMHCpan only\n(Negatives)",
    "IEDB only\n(not predicted)"
  ),
  n = c(nrow(df_iedb_pos), nrow(df_netmhcpan_only), nrow(df_iedb_only)),
  type = c("Positive", "Negative", "Unresolved")
)

p1 <- ggplot(df_composition, aes(x = reorder(category, -n), y = n, fill = type)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = scales::comma(n)), vjust = -0.5, size = 4) +
  scale_fill_manual(values = c("Positive" = "#2ecc71",
                               "Negative" = "#e74c3c",
                               "Unresolved" = "#95a5a6")) +
  scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.1))) +
  labs(
    title    = "Dataset Composition",
    subtitle = "NetMHCpan predictions vs IEDB confirmed epitopes",
    x        = NULL,
    y        = "Number of peptides",
    fill     = NULL
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "top")

p1

# Ratio comparison plot 
df_ratios <- tibble(
  ratio    = c("1:1", "1:3", "1:5", "1:10", "Full (1:12)"),
  n_neg    = c(1, 3, 5, 10, 12) * nrow(df_iedb_pos),
  n_pos    = nrow(df_iedb_pos),
  feasible = c(TRUE, TRUE, TRUE, TRUE, FALSE)
) |>
  mutate(
    n_neg    = pmin(n_neg, nrow(df_netmhcpan_only)),
    n_total  = n_pos + n_neg
  )

p2 <- ggplot(df_ratios, aes(x = ratio, y = n_total, fill = feasible)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = scales::comma(n_total)), vjust = -0.5, size = 4) +
  geom_hline(yintercept = nrow(df_iedb_pos), linetype = "dashed",
             color = "#2ecc71", linewidth = 0.8) +
  annotate("text", x = 0.6, y = nrow(df_iedb_pos) * 1.05,
           label = "# positives", color = "#2ecc71", size = 3.5, hjust = 0) +
  scale_fill_manual(values = c("TRUE" = "#3498db", "FALSE" = "#bdc3c7")) +
  scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.1))) +
  labs(
    title    = "Training Set Size by Pos:Neg Ratio",
    subtitle = paste0(scales::comma(nrow(df_netmhcpan_only)),
                      " negatives available from NetMHCpan"),
    x        = "Pos : Neg ratio",
    y        = "Total training samples",
    fill     = "Feasible"
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "none")

p2
# Peptide length distribution 
df_lengths <- bind_rows(
  df_iedb_pos       |> mutate(set = "Positives (IEDB)"),
  df_netmhcpan_only |> mutate(set = "Negatives (NetMHCpan only)")
)

p3 <- ggplot(df_lengths, aes(x = pep_length, fill = set)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("Positives (IEDB)" = "#2ecc71",
                               "Negatives (NetMHCpan only)" = "#e74c3c")) +
  scale_x_continuous(breaks = 8:14) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "Peptide Length Distribution",
    x     = "Peptide length",
    y     = "Count",
    fill  = NULL
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "top")

p3
# Save all plots 
ggsave("results/dataset_composition.png",    p1, width = 7, height = 5, dpi = 150)
ggsave("results/training_ratio_options.png", p2, width = 7, height = 5, dpi = 150)
ggsave("results/peptide_length_dist.png",    p3, width = 7, height = 5, dpi = 150)




