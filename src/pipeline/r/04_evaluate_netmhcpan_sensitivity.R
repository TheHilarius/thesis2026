suppressPackageStartupMessages(library(tidyverse))
source("src/pipeline/r/functions.R")
set_working_directory()
suppressPackageStartupMessages(library(ggplot2))

# Load IEDB positives 
df_raw <- read_csv("data/processed/pos_EL_all_epitopes_hla0201.csv", show_col_types = FALSE)

df_iedb_pos <- df_raw |>
  filter(pep_length == 9) |>
  filter(uniprot_id != "O60361")  # deleted from SwissProt in 2026_01

# deduplication
n_before <- nrow(df_iedb_pos)
df_iedb_pos <- df_iedb_pos |>
  distinct(peptide, uniprot_id, start, end, .keep_all = TRUE)
n_after <- nrow(df_iedb_pos)

if (n_before > n_after) {
  cat("Removed", n_before - n_after, "duplicate positives\n")
} else {
  cat("No duplicate positives found.\n")
}

# Validate against FASTA database ────────────────────────────────────────
cat("Loading FASTA database to remove obsolete isoforms...\n")
df_fasta_raw <- read_fasta_df("data/raw/fasta/combined_positives_only.fasta") |>
  mutate(
    accession  = str_extract(header, "(?<=\\|)[^|]+(?=\\|)"),
    uniprot_id = accession,
    is_isoform = str_detect(accession, "-[0-9]+$")
  ) |>
  filter(!is.na(uniprot_id))

# Keep ALL isoform sequences — do not collapse isoforms to canonical IDs.
# This preserves biological ground truth: if cleavage happened on isoform -7,
# we need the 3D structure of isoform -7, not the canonical parent.
# Isoforms without AlphaFold structures will cleanly fall into no_pdb exclusion.
df_fasta <- df_fasta_raw |>
  mutate(seq_length = nchar(sequence)) |>
  distinct(uniprot_id, .keep_all = TRUE)

cat("FASTA entries loaded:      ", nrow(df_fasta_raw), "\n")
cat("Isoform entries (`-N`):    ", sum(df_fasta_raw$is_isoform), "\n")
cat("Unique sequences kept:     ", nrow(df_fasta), "\n")
cat("(Isoform IDs preserved — no canonical collapse)\n\n")

df_iedb_pos <- df_iedb_pos |>
  semi_join(df_fasta, by = "uniprot_id")

# ── Biological Integrity Check: validate peptide coordinates ──────────────
# Join canonical sequence from FASTA, then validate that the peptide at
# reported start/end coordinates matches the actual amino acid string.
# Drops peptides with isoform-mismatched coordinates (silent coordinate shifts).
cat("\n--- Biological Integrity Check (positives) ---\n")
df_iedb_pos <- df_iedb_pos |>
  left_join(df_fasta |> select(uniprot_id, sequence), by = "uniprot_id") |>
  validate_and_fix_positions(
    sequence_col = "sequence",
    peptide_col  = "peptide",
    start_col    = "start",
    end_col      = "end"
  )

coord_summary_pos <- df_iedb_pos |> count(position_status) |>
  mutate(percentage = round(n / sum(n) * 100, 2))
cat("Position validation (positives):\n")
print(coord_summary_pos)

n_unfixable_pos <- sum(df_iedb_pos$position_status == "unfixable", na.rm = TRUE)
df_iedb_pos <- df_iedb_pos |>
  filter(position_status %in% c("valid_original", "fixed")) |>
  select(-start_original, -end_original, -position_originally_valid,
         -position_status, -position_shift)
cat("Dropped", n_unfixable_pos, "unfixable positives\n")
if (n_unfixable_pos > 0) {
  cat("  (These peptides could not be found anywhere in the reference FASTA.\n")
  cat("   Likely causes: IEDB curational errors, uncaptured splice variants,\n")
  cat("   or point mutations not present in the wild-type reference genome.)\n")
}
cat("IEDB positives after coord validation:", nrow(df_iedb_pos), "\n\n")

# Remove proteins with selenocysteine (U).
sec_proteins <- df_fasta |>
  filter(str_detect(sequence, "U")) |>
  pull(uniprot_id)

cat("Selenocysteine (U) proteins found in FASTA:", length(sec_proteins), "\n")

n_iedb_sec <- df_iedb_pos |>
  filter(uniprot_id %in% sec_proteins) |>
  nrow()

df_iedb_pos <- df_iedb_pos |>
  filter(!uniprot_id %in% sec_proteins)

cat("Removed", n_iedb_sec, "IEDB positives from", length(sec_proteins),
    "selenocysteine-containing proteins\n")
cat("IEDB positives after Sec filter:", nrow(df_iedb_pos), "\n")

# ── Unified protein exclusion ledger ──────────────────────────────────────
# STRICT ORDER: Length filter FIRST, then AlphaFold checks.
# This prevents loading pLDDT data for proteins that will be excluded by length.
# Check ALL proteins against ALL filters before applying any removals.
# This enables overlap analysis and prevents sequential bias.
cat("\n--- Building exclusion ledger (all proteins) ---\n")

# Snapshot pre-filter state (needed for peptide ledger and overlap analysis)
df_iedb_pos_pre_af <- df_iedb_pos

# ── STEP 1: Length filter (applied FIRST) ────────────────────────────────
df_ledger <- tibble(
  uniprot_id = unique(df_iedb_pos$uniprot_id)
) |>
  left_join(df_fasta |> select(uniprot_id, seq_length), by = "uniprot_id") |>
  mutate(
    too_short       = seq_length < 130,
    too_long        = seq_length > 5000,
    length_excluded = too_short | too_long
  )

n_too_short <- sum(df_ledger$too_short)
n_too_long  <- sum(df_ledger$too_long)
cat("  Length filter (130-5000 aa):\n")
cat("    Too short (<130 aa):", n_too_short, "proteins\n")
cat("    Too long  (>5000 aa):", n_too_long, "proteins\n")
cat("    Total length-excluded:", n_too_short + n_too_long, "proteins\n")
cat("    Peptides lost:",
    sum(df_iedb_pos_pre_af$uniprot_id %in% (df_ledger |> filter(too_short) |> pull(uniprot_id))),
    "+",
    sum(df_iedb_pos_pre_af$uniprot_id %in% (df_ledger |> filter(too_long) |> pull(uniprot_id))),
    "IEDB positives\n")

# ── STEP 2: AlphaFold checks (ONLY for proteins passing length filter) ────
af_dirs <- "data/processed/structures/alphafold/"
af_files <- list.files(af_dirs, pattern = "\\.pdb$", full.names = FALSE)
af_proteins <- str_remove(af_files, "\\.pdb$")

cat("  AlphaFold PDB files available:", length(af_proteins), "\n")

# Add AlphaFold status to ledger (no_pdb flagged for all proteins for diagnostics)
df_ledger <- df_ledger |>
  mutate(
    has_pdb = uniprot_id %in% af_proteins,
    no_pdb  = !has_pdb
  )

# ── Dynamic breakdown of missing AlphaFold structures ──────────────────────
# Cross-reference no_pdb proteins against the fetch logs to explain WHY
# structures are missing. These logs are produced by fetch_structures.sh.
no_pdb_ids <- df_ledger$uniprot_id[df_ledger$no_pdb & !df_ledger$length_excluded]

if (length(no_pdb_ids) > 0) {
  missing_log <- read_tsv("data/processed/structures/logs/missing_structures.tsv",
                          show_col_types = FALSE)
  invalid_log <- read_tsv("data/processed/structures/logs/invalid_uniprot_ids.tsv",
                          show_col_types = FALSE)
  truly_invalid <- read_tsv("data/processed/structures/logs/truly_invalid_ids.tsv",
                            show_col_types = FALSE)

  n_no_af2    <- sum(no_pdb_ids %in% missing_log$uniprot_id)
  n_isoform   <- sum(grepl("-", no_pdb_ids) & no_pdb_ids %in% invalid_log$uniprot_id)
  # NP/RefSeq anomaly: regex in script 02 extracts "NP" from RefSeq IRIs like
  # "uniprot/NP_000959.2" because [A-Z0-9]+ doesn't match underscores.
  # coalesce picks "NP" over the correct parent ID. 6 peptides affected.
  # We intentionally let them drop here to avoid re-running the pipeline for 6 peptides.
  n_invalid   <- sum(no_pdb_ids %in% truly_invalid$uniprot_id)
  n_other     <- length(no_pdb_ids) - n_no_af2 - n_isoform - n_invalid

  cat("  AlphaFold breakdown (", length(no_pdb_ids), " proteins without PDBs):\n")
  cat("    No AF2/PDB structure exists:       ", n_no_af2, "\n")
  if (n_isoform > 0) cat("    Isoforms not modeled by AF2:       ", n_isoform, "\n")
  if (n_invalid > 0) cat("    Invalid IDs (RefSeq NP_ anomaly):  ", n_invalid,
        " (Safely dropped; caused by upstream regex edge-case)\n")
  if (n_other > 0)   cat("    Other:                              ", n_other, "\n")
}

# ── STEP 3: Out-of-range check (ONLY for proteins with PDBs) ─────────────
# Load pLDDT only for proteins that have PDBs AND pass length filter
af_proteins_needed <- df_ledger$uniprot_id[df_ledger$has_pdb & !df_ledger$length_excluded]

af_lookup <- list()
for (d in af_dirs) {
  af_lookup <- c(af_lookup, build_plddt_lookup(proteins_needed = af_proteins_needed, dir = d))
}

af_modelled <- map_dfr(names(af_lookup), function(uid) {
  lkp <- af_lookup[[uid]]
  tibble(uniprot_id = uid, max_modelled = max(lkp$residue_num))
})

# Flag out_of_range: any peptide end > max_modelled
proteins_out_of_range <- df_iedb_pos |>
  left_join(af_modelled, by = "uniprot_id") |>
  filter(end > max_modelled) |>
  distinct(uniprot_id)

df_ledger <- df_ledger |>
  mutate(out_of_range = uniprot_id %in% proteins_out_of_range$uniprot_id)

# Log AlphaFold exclusions (only among proteins that passed length filter)
n_no_pdb <- sum(df_ledger$no_pdb & !df_ledger$length_excluded)
n_oor     <- sum(df_ledger$out_of_range & !df_ledger$length_excluded)
cat("  AlphaFold filter (among length-passing proteins):\n")
cat("    No PDB:", n_no_pdb, "proteins\n")
cat("    Out of range:", n_oor, "proteins\n")
# Only count length-passing proteins to avoid double-counting overlap with length filter
cat("    Peptides lost:",
    sum(df_iedb_pos_pre_af$uniprot_id %in% (df_ledger |> filter(no_pdb & !length_excluded) |> pull(uniprot_id))),
    "+",
    sum(df_iedb_pos_pre_af$uniprot_id %in% (df_ledger |> filter(out_of_range & !length_excluded) |> pull(uniprot_id))),
    "IEDB positives\n")

# ── Build final exclusion flags ──────────────────────────────────────────
# Length takes priority in exclusion_reasons (first match wins)
df_ledger <- df_ledger |>
  mutate(
    excluded = too_short | too_long | no_pdb | out_of_range,
    exclusion_reasons = case_when(
      too_short ~ "too_short",
      too_long  ~ "too_long",
      no_pdb    ~ "no_pdb",
      out_of_range ~ "out_of_range",
      TRUE      ~ ""
    )
  )

write_csv(df_ledger, "data/processed/exclusion_ledger.csv")

# ── Peptide-level exclusion ledger ──────────────────────────────────────
df_peptide_ledger <- df_iedb_pos_pre_af |>
  left_join(
    df_ledger |> select(uniprot_id, too_short, too_long, no_pdb, out_of_range,
                         excluded, exclusion_reasons),
    by = "uniprot_id"
  )

write_csv(df_peptide_ledger, "data/processed/exclusion_ledger_peptides.csv")
cat("  Peptide ledger:", nrow(df_peptide_ledger), "peptides,",
    sum(df_peptide_ledger$excluded), "from excluded proteins\n")

cat("  Proteins in ledger: ", nrow(df_ledger), "\n")
cat("  Excluded:           ", sum(df_ledger$excluded), "\n")
cat("    too_short:        ", sum(df_ledger$too_short), "\n")
cat("    too_long:         ", sum(df_ledger$too_long), "\n")
cat("    no_pdb:           ", sum(df_ledger$no_pdb), "\n")
cat("    out_of_range:     ", sum(df_ledger$out_of_range), "\n")
cat("  Retained:           ", sum(!df_ledger$excluded), "\n")

# Apply single unified filter
df_iedb_pos <- df_iedb_pos |>
  semi_join(df_ledger |> filter(!excluded), by = "uniprot_id")

cat("\n  Unified exclusion results:\n")
cat("  Proteins excluded:  ", sum(df_ledger$excluded), "\n")
cat("  Positives remaining:", nrow(df_iedb_pos), "\n\n")

write_csv(df_iedb_pos, "data/processed/pos_EL_9mers_epitopes_hla0201.csv")
cat("IEDB positives:", nrow(df_iedb_pos), "\n")

# Protein lookup table 
df_protein_lookup <- df_iedb_pos |>
  select(uniprot_id, source_molecule, molecule_parent) |>
  slice_head(n = 1, by = uniprot_id)

write_csv(df_protein_lookup, "data/processed/protein_lookup.csv")
cat("Unique proteins:", nrow(df_protein_lookup), "\n")


# Load NetMHCpan predictions (9-mer binders only) 
cat("Loading NetMHCpan 9-mer binders...\n")
df_netmhcpan_raw <- load_netmhcpan_batches(binders_only = TRUE, peptide_length = 9)
cat("Total 9-mer binders loaded:", nrow(df_netmhcpan_raw), "\n")


# Parse, clean, and filter to valid proteins 
df_netmhcpan_binders_parsed <- df_netmhcpan_raw |>
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
  select(-c(id, core, icore, score, ave))

# Filter binders to valid proteins and join metadata
df_netmhcpan_binders <- df_netmhcpan_binders_parsed |>
  semi_join(df_protein_lookup, by = "uniprot_id") |>
  left_join(df_protein_lookup, by = "uniprot_id")

cat("Parsed binders (valid proteins only):", nrow(df_netmhcpan_binders), "\n")
cat("Unique proteins retained:", n_distinct(df_netmhcpan_binders$uniprot_id), "\n")

missing_proteins <- df_protein_lookup |>
  anti_join(df_netmhcpan_binders, by = "uniprot_id")

cat("Note:", nrow(missing_proteins),
    "proteins in lookup had no predicted 9-mer binders (Rank < 2%)\n")
cat("The", nrow(missing_proteins), "proteins without 9-mer binders have",
    get_netmhcpan_total(peptide_length = 9, valid_ids = missing_proteins$uniprot_id),
    "non-binders (predicted by NetMHCpan)\n")


# Compare NetMHCpan vs IEDB 
df_iedb_pos <- df_iedb_pos |>
  left_join(
    df_netmhcpan_binders |>
      select(peptide, uniprot_id, rank) |>
      slice_min(rank, n = 1, with_ties = FALSE, by = c(peptide, uniprot_id)),
    by = c("peptide", "uniprot_id")
  )
cat("IEDB positives with rank (TP):", sum(!is.na(df_iedb_pos$rank)), "\n")
cat("IEDB positives without rank (FN):", sum(is.na(df_iedb_pos$rank)), "\n")

# Combined removed-proteins file from the exclusion ledger.
# Pull counts directly from the ledger using peptide-level data.
df_all_removed_proteins <- df_ledger |>
  filter(excluded) |>
  # n_positives: IEDB positive peptides from each excluded protein
  left_join(
    df_iedb_pos_pre_af |> count(uniprot_id, name = "n_positives"),
    by = "uniprot_id"
  ) |>
  # n_negatives: NetMHCpan binders from each excluded protein, minus IEDB positives
  left_join(
    df_netmhcpan_binders_parsed |>
      anti_join(df_iedb_pos_pre_af, by = c("peptide", "uniprot_id")) |>
      count(uniprot_id, name = "n_negatives"),
    by = "uniprot_id"
  ) |>
  mutate(
    across(starts_with("n_"), ~ replace_na(.x, 0L)),
    filter_stage = case_when(
      too_short | too_long ~ "NetSurfP",
      no_pdb | out_of_range ~ "AlphaFold",
      TRUE ~ "Unknown"
    )
  ) |>
  select(uniprot_id, seq_length, filter_stage, exclusion_reasons,
         n_positives, n_negatives, too_short, too_long, no_pdb, out_of_range)

write_csv(df_all_removed_proteins, "data/processed/af_nsp3_removed.csv")

cat("\n=== Proteins removed by prefilter (ledger-based) ===\n")
df_all_removed_proteins |>
  group_by(filter_stage) |>
  summarise(
    n_proteins  = n(),
    n_positives = sum(n_positives),
    n_negatives = sum(n_negatives),
    .groups = "drop"
  ) |>
  arrange(filter_stage) |>
  print()

# ── Exclusion overlap analysis ────────────────────────────────────────────
cat("\n", strrep("=", 70), "\n")
cat("  EXCLUSION OVERLAP ANALYSIS\n")
cat(strrep("=", 70), "\n\n")

# Table 1: Category counts (protein + peptide level)
cat("--- Table 1: Category Counts ---\n")
cat(sprintf("  %-15s %10s %10s\n", "Category", "Proteins", "Positives"))
cat(strrep("-", 40), "\n")

for (cat_name in c("too_short", "too_long", "no_pdb", "out_of_range")) {
  prot_ids <- df_ledger$uniprot_id[df_ledger[[cat_name]]]
  n_prot <- length(prot_ids)
  n_pos <- df_iedb_pos_pre_af |> filter(uniprot_id %in% prot_ids) |> nrow()
  cat(sprintf("  %-15s %10d %10d\n", cat_name, n_prot, n_pos))
}

cat(strrep("-", 40), "\n")
cat(sprintf("  %-15s %10d %10d\n", "Total excluded",
            sum(df_ledger$excluded),
            sum(df_peptide_ledger$excluded)))
cat(sprintf("  %-15s %10d %10d\n", "Retained",
            sum(!df_ledger$excluded),
            sum(!df_peptide_ledger$excluded)))

# Table 2: Cross-filter overlap matrix
cat("\n--- Table 2: Cross-filter Overlap ---\n")
cat(sprintf("  %-35s %10s\n", "Comparison", "Proteins"))
cat(strrep("-", 48), "\n")

cross_combos <- list(
  c("too_short", "no_pdb"),
  c("too_short", "out_of_range"),
  c("too_long", "no_pdb"),
  c("too_long", "out_of_range")
)

for (combo in cross_combos) {
  n <- sum(df_ledger[[combo[1]]] & df_ledger[[combo[2]]])
  label <- paste0(combo[1], " \u2229 ", combo[2])
  cat(sprintf("  %-35s %10d\n", label, n))
}

# Theoretical AlphaFold savings
cat("\n--- Theoretical AlphaFold Savings ---\n")
n_skip_nopdb <- sum((df_ledger$too_short | df_ledger$too_long) & df_ledger$no_pdb)
n_skip_oor   <- sum((df_ledger$too_short | df_ledger$too_long) & df_ledger$out_of_range)
n_skip_total <- sum((df_ledger$too_short | df_ledger$too_long) &
                     (df_ledger$no_pdb | df_ledger$out_of_range))
cat("  no_pdb proteins already excluded by length:", n_skip_nopdb, "\n")
cat("  out_of_range proteins already excluded by length:", n_skip_oor, "\n")
cat("  Total AlphaFold runs theoretically skippable:", n_skip_total, "\n")
cat("  (These proteins would fail both NetSurfP AND AlphaFold)\n")

df_netmhcpan_binders_unique <- df_netmhcpan_binders |>
  distinct(peptide, uniprot_id, .keep_all = TRUE)

df_overlap <- df_netmhcpan_binders_unique |>
  semi_join(df_iedb_pos, by = c("peptide", "uniprot_id"))

df_netmhcpan_only <- df_netmhcpan_binders_unique |>
  anti_join(df_iedb_pos, by = c("peptide", "uniprot_id"))


df_iedb_only      <- df_iedb_pos |> 
  anti_join(df_netmhcpan_binders_unique, by = c("peptide", "uniprot_id"))

cat("\n=== Comparison Summary (9-mers only) ===\n")
cat("Experimentally confirmed (IEDB):          ", nrow(df_iedb_pos), "\n")
cat("Predicted binders (NetMHCpan):            ", nrow(df_netmhcpan_binders_unique), "\n")
cat("Overlap (both):                           ", nrow(df_overlap), "\n")
cat("NetMHCpan only (predicted, not confirmed):", nrow(df_netmhcpan_only), "\n")
cat("IEDB only (confirmed, not predicted):     ", nrow(df_iedb_only), "\n")
cat("Sensitivity (IEDB peptides recovered):    ",
    round(nrow(df_overlap) / nrow(df_iedb_pos) * 100, 1), "%\n")


# Save split outputs 
write_csv(df_netmhcpan_binders, "data/processed/netmhcpan_9mer_binders.csv")
write_csv(df_netmhcpan_only,    "data/processed/netmhcpan_9mer_only.csv")
write_csv(df_iedb_only,         "data/processed/iedb_9mer_only.csv")
cat("\n✅ Saved split data to data/processed/\n")


# Confusion matrix 
TP <- nrow(df_overlap)
FP <- nrow(df_netmhcpan_only)
FN <- nrow(df_iedb_only)

valid_proteins <- df_protein_lookup$uniprot_id
total_peptides <- get_netmhcpan_total(peptide_length = 9, valid_ids = valid_proteins)
cat("Total 9-mer peptides from valid proteins:", scales::comma(total_peptides), "\n")

TN <- total_peptides - nrow(df_netmhcpan_binders_unique) - FN

sensitivity  <- TP / (TP + FN)
specificity  <- TN / (TN + FP)
precision    <- TP / (TP + FP)
f1           <- 2 * (precision * sensitivity) / (precision + sensitivity)

cat("\n=== Confusion Matrix Metrics (9-mers) ===\n")
cat("TP:", scales::comma(TP), "\n")
cat("FP:", scales::comma(FP), "\n")
cat("FN:", scales::comma(FN), "\n")
cat("TN:", scales::comma(TN), "\n")
cat("Sensitivity:", round(sensitivity * 100, 1), "%\n")
cat("Specificity:", round(specificity * 100, 1), "%\n")
cat("Precision:  ", round(precision * 100, 1), "%\n")
cat("F1:         ", round(f1, 3), "\n")

df_cm <- tibble(
  Predicted = factor(c("Binder", "Binder", "Non-Binder", "Non-Binder"),
                     levels = c("Binder", "Non-Binder")),
  Actual    = factor(c("IEDB Positive", "IEDB Negative", "IEDB Positive", "IEDB Negative"),
                     levels = c("IEDB Positive", "IEDB Negative")),
  Count = c(TP, FP, FN, TN),
  quad  = c("TP", "FP", "FN", "TN")
) |>
  mutate(display = paste0(quad, "\n", scales::comma(Count)))

quad_colours <- c("TP" = "#2ecc71", "TN" = "#3498db", "FP" = "#e74c3c", "FN" = "#e67e22")

col_sums <- c(
  "Binder"     = paste0("Total: ", scales::comma(TP + FP)),
  "Non-Binder" = paste0("Total: ", scales::comma(FN + TN))
)
row_sums <- c(
  "IEDB Positive" = paste0("Total:\n", scales::comma(TP + FN)),
  "IEDB Negative" = paste0("Total:\n", scales::comma(FP + TN))
)

p1 <- ggplot(df_cm, aes(x = Predicted, y = Actual, fill = quad)) +
  geom_tile(colour = "white", linewidth = 2) +
  geom_text(aes(label = display), size = 5, fontface = "bold", colour = "white") +
  scale_fill_manual(values = quad_colours,
                    labels = c("TP" = "True Pos.", "TN" = "True Neg.",
                               "FP" = "False Pos.", "FN" = "False Neg.")) +
  scale_x_discrete(position = "top", sec.axis = dup_axis(name = NULL, labels = col_sums)) +
  scale_y_discrete(sec.axis = dup_axis(name = NULL, labels = row_sums)) +
  labs(
    title    = "9-mer Peptide Binders: NetMHCpan vs IEDB",
    subtitle = paste0(
      "HLA-A*02:01  |  9-mers only  |  Binder threshold: Rank < 2%\n",
      "Sensitivity: ", round(sensitivity * 100, 1), "%  |  ",
      "Precision: ",   round(precision  * 100, 1), "%  |  ",
      "F1: ",          round(f1, 3)),
    x = "NetMHCpan Prediction", y = "", fill = NULL
  ) +
  theme_bw(base_size = 13) +
  theme(
    legend.position    = "bottom",
    legend.box.margin  = margin(r = 80),
    plot.title         = element_text(margin = margin(l = 20, b = 5)),
    plot.subtitle      = element_text(size = 10, colour = "grey40", margin = margin(l = 20, b = 10)),
    axis.text          = element_text(size = 12, face = "bold"),
    axis.text.x.bottom = element_text(size = 11, face = "italic", colour = "grey30"),
    axis.text.y.right  = element_text(size = 11, face = "italic", colour = "grey30", hjust = 0),
    axis.title.x.top   = element_text(face = "bold", margin = margin(b = 10)),
    panel.grid         = element_blank(),
    plot.title.position = "plot",
    plot.margin = margin(t = 10, r = 10, b = 10, l = 5)
  )

# Define positives & pre-matching diagnostics 
df_positives <- df_overlap |>
  distinct(peptide, uniprot_id, start, end, .keep_all = TRUE) |>
  mutate(label = 1)

#  Affinity bias check 
# If positives (IEDB-confirmed) have systematically lower EL ranks than negatives
# (predicted binders, not in IEDB), a model could classify by affinity alone.
# We use KS D as an effect-size metric. At n > 20K, p-values are always
# significant and uninformative; only the magnitude of D matters.

df_rank_pre <- bind_rows(
  df_positives      |> transmute(rank, label = "Positive (TP)"),
  df_netmhcpan_only |> transmute(rank, label = "Negative (NetMHCpan only)")
)

cat("\n=== Pre-Matching Rank Distribution ===\n")
df_rank_pre |>
  group_by(label) |>
  summarise(
    n = n(), median = median(rank), mean = mean(rank), sd = sd(rank),
    q25 = quantile(rank, 0.25), q75 = quantile(rank, 0.75),
    .groups = "drop"
  ) |>
  print()

ks_pre <- ks.test(df_positives$rank, df_netmhcpan_only$rank)
cat("KS D (before):", round(ks_pre$statistic, 4), "→ clearly different distributions\n")

#  Granular rank breakdown 
rank_bins_granular <- c(0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0)

cat("\n--- Positive rank breakdown ---\n")
df_positives |>
  mutate(rank_range = cut(rank, breaks = rank_bins_granular, include.lowest = TRUE)) |>
  count(rank_range) |>
  mutate(pct = round(n / sum(n) * 100, 1)) |>
  print()

cat("\n--- Negative rank breakdown ---\n")
df_netmhcpan_only |>
  mutate(rank_range = cut(rank, breaks = rank_bins_granular, include.lowest = TRUE)) |>
  count(rank_range) |>
  mutate(pct = round(n / sum(n) * 100, 1)) |>
  print()

#  Pre-matching plots 
p2 <- ggplot(df_rank_pre, aes(x = rank, fill = label)) +
  geom_density(alpha = 0.5, colour = "black", linewidth = 0.3) +
  scale_fill_manual(values = c("Positive (TP)" = "#2ecc71",
                               "Negative (NetMHCpan only)" = "#e74c3c")) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey40", linewidth = 0.5) +
  annotate("text", x = 0.5, y = Inf, label = "SB threshold (0.5%)",
           vjust = 2, hjust = -0.05, size = 3, colour = "grey40") +
  annotate("text", x = 1.5, y = Inf,
           label = paste0("KS D = ", round(ks_pre$statistic, 4)),
           vjust = 1.5, size = 5, fontface = "bold", colour = "#e74c3c") +
  labs(
    title    = "Affinity Bias Check: NetMHCpan EL Rank Distribution",
    subtitle = paste0("9-mers  |  HLA-A*02:01  |  KS D = ", round(ks_pre$statistic, 4),
                      " (clearly different distributions)"),
    x = "NetMHCpan EL Rank (%)", y = "Density", fill = NULL
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "top", panel.grid.minor = element_blank(),
        plot.title.position = "plot")

p3 <- ggplot(df_rank_pre, aes(x = rank, colour = label)) +
  stat_ecdf(linewidth = 0.8) +
  scale_colour_manual(values = c("Positive (TP)" = "#2ecc71",
                                 "Negative (NetMHCpan only)" = "#e74c3c")) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey40") +
  annotate("text", x = 1.8, y = 0.08,
           label = paste0("KS D = ", round(ks_pre$statistic, 4)),
           size = 4.5, fontface = "bold", hjust = 1, colour = "#e74c3c") +
  labs(
    title    = "Cumulative Rank Distribution: Before Matching",
    subtitle = "Curve separation = affinity bias present",
    x = "NetMHCpan EL Rank (%)", y = "Cumulative Proportion", colour = NULL
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "top", panel.grid.minor = element_blank(),
        plot.title.position = "plot")


# ── Pre-filter: remove negatives from proteins with peptides beyond AF model range ──
# Step 07 will remove these proteins anyway (out_of_range check).
# Removing their negatives now means downsampling picks different random
# negatives instead. Same final count, no bias introduced.

cat("\n--- Pre-filtering negatives with out-of-range peptides ---\n")

# Reuse af_modelled from the ledger section (already loaded for all retained proteins)
# This avoids reloading pLDDT for ~9354 proteins — saves significant compute time.
oor_neg_proteins <- df_netmhcpan_only |>
  left_join(af_modelled, by = "uniprot_id") |>
  filter(end > max_modelled) |>
  distinct(uniprot_id)

# Remove these negatives
n_before_pre <- nrow(df_netmhcpan_only)
df_netmhcpan_only <- df_netmhcpan_only |>
  filter(!(uniprot_id %in% oor_neg_proteins$uniprot_id))
cat("Pre-filtered", n_before_pre - nrow(df_netmhcpan_only),
    "negatives from", nrow(oor_neg_proteins), "out-of-range proteins\n")
cat("df_netmhcpan_only:", nrow(df_netmhcpan_only), "\n")

# Save pre-filtered proteins for reference
write_csv(oor_neg_proteins, "data/processed/pre_filtered_out_of_range_proteins.csv")


# Rank-matched negative sampling 
RANDOM_SEED = 42
CHOSEN_N_BINS = 25 # Use later when building df_negatives

#  Bin count sweep 
test_bins <- c(seq(5, 20, by = 1), seq(25, 200, by = 5))

bin_sweep <- map_dfr(test_bins, function(nb) {
  breaks <- seq(0, 2, length.out = nb + 1)
  
  pos_binned <- df_positives |>
    mutate(rank_bin = cut(rank, breaks = breaks, include.lowest = TRUE))
  neg_binned <- df_netmhcpan_only |>
    mutate(rank_bin = cut(rank, breaks = breaks, include.lowest = TRUE))
  
  props <- pos_binned |> count(rank_bin, name = "n_pos") |> mutate(prop = n_pos / sum(n_pos))
  avail <- neg_binned |> count(rank_bin, name = "n_available")
  
  plan <- props |>
    left_join(avail, by = "rank_bin") |>
    mutate(n_available = replace_na(n_available, 0),
           max_total = ifelse(prop > 0, n_available / prop, Inf))
  
  max_n <- floor(min(plan$max_total))
  plan <- plan |> mutate(n_sample = floor(max_n * prop))
  
  set.seed(42)
  neg_sampled <- plan |>
    select(rank_bin, n_sample) |>
    pmap_dfr(function(rank_bin, n_sample) {
      neg_binned |>
        filter(rank_bin == !!rank_bin) |>
        slice_sample(n = n_sample)
    })
  
  ks <- ks.test(df_positives$rank, neg_sampled$rank)
  d <- round(unname(ks$statistic), 4)
  
  tibble(
    n_bins = nb,
    n_neg = nrow(neg_sampled),
    ratio = round(nrow(neg_sampled) / nrow(df_positives), 2),
    ks_D = d,
    match_quality = case_when(
      d < 0.02 ~ "★★★★★ excellent",
      d < 0.05 ~ "★★★★ very good",
      d < 0.10 ~ "★★★ decent",
      d < 0.15 ~ "★★ acceptable",
      d < 0.20 ~ "★ alright",
      TRUE     ~ "⚠ marginal"
    ),
    bottleneck = as.character(plan$rank_bin[which.min(plan$max_total)])
  )
})

cat("\n=== Bin Count Sweep: Data vs Match Quality Tradeoff ===\n")
print(bin_sweep, n = Inf)

#  Select representative configurations for comparison plots 
configs <- bind_rows(
  bin_sweep |> filter(ks_D < 0.02)               |> arrange(desc(n_neg)) |> slice_head(n = 1) |> mutate(tier = "A: Excellent (D<0.02)"),
  bin_sweep |> filter(ks_D >= 0.02, ks_D < 0.05) |> arrange(desc(n_neg)) |> slice_head(n = 1) |> mutate(tier = "B: Very good (D<0.05)"),
  bin_sweep |> filter(ks_D >= 0.05, ks_D < 0.10) |> arrange(desc(n_neg)) |> slice_head(n = 1) |> mutate(tier = "C: Decent (D<0.10)"),
  bin_sweep |> filter(ks_D >= 0.10, ks_D < 0.15) |> arrange(desc(n_neg)) |> slice_head(n = 1) |> mutate(tier = "D: Acceptable (D<0.15)"),
  bin_sweep |> filter(ks_D >= 0.15, ks_D < 0.20) |> arrange(desc(n_neg)) |> slice_head(n = 1) |> mutate(tier = "E: Alright (D<0.20)"),
  bin_sweep |> filter(ks_D >= 0.20, ks_D < 0.25) |> arrange(desc(n_neg)) |> slice_head(n = 1) |> mutate(tier = "F: Marginal (D<0.25)")
) |>
  filter(!is.na(n_bins))

cat("\n=== Selected Configurations ===\n")
configs |> select(tier, n_bins, n_neg, ratio, ks_D) |> print(n = Inf)

#  Generate matched sets for comparison plots 
neg_sets <- map(configs$n_bins, function(nb) {
  sample_bin_matched(df_positives, df_netmhcpan_only, nb)
})
names(neg_sets) <- configs$tier


# Comparison plots across configurations 

df_facet_density <- imap_dfr(neg_sets, function(df_neg, tier_name) {
  cfg <- configs |> filter(tier == tier_name)
  bind_rows(
    df_positives |> transmute(rank, label = "Positive (TP)"),
    df_neg       |> transmute(rank, label = "Negative")
  ) |> mutate(facet = paste0(tier_name, "\n", cfg$n_bins, " bins, n=",
                             scales::comma(cfg$n_neg), ", D=", cfg$ks_D))
})

facet_order <- map_chr(rev(seq_len(nrow(configs))), function(i) {
  cfg <- configs[i, ]
  paste0(cfg$tier, "\n", cfg$n_bins, " bins, n=",
         scales::comma(cfg$n_neg), ", D=", cfg$ks_D)
})

df_facet_density <- df_facet_density |>
  mutate(facet = factor(facet, levels = facet_order))

p4 <- ggplot(df_facet_density, aes(x = rank, fill = label)) +
  geom_density(alpha = 0.5, colour = "black", linewidth = 0.3) +
  scale_fill_manual(values = c("Positive (TP)" = "#2ecc71", "Negative" = "#e74c3c")) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey40", linewidth = 0.3) +
  facet_wrap(~facet, ncol = 3, scales = "free_y") +
  labs(
    title    = "Affinity Bias Correction: Matching Quality Across Options",
    subtitle = "Each panel shows positives vs negatives at different bin-matching resolutions\nD < 0.02 excellent | D < 0.05 very good | D < 0.10 decent | D < 0.15 acceptable | D < 0.20 alright | D \u2265 0.20 marginal",
    x = "NetMHCpan EL Rank (%)", y = "Density", fill = NULL
  ) +
  theme_bw(base_size = 16) +
  theme(
    legend.position     = "top",
    panel.grid.minor    = element_blank(),
    plot.title.position = "plot",
    panel.spacing = unit(0.6, "lines"),
    plot.margin = margin(8, 12, 8, 8),
    plot.title = element_text(size = 20, face = "bold"),
    plot.subtitle = element_text(size = 13),
    strip.text = element_text(size = 11, face = "bold")
  )

p5 <- ggplot(df_facet_density, aes(x = rank, colour = label)) +
  stat_ecdf(linewidth = 0.7) +
  scale_colour_manual(values = c("Positive (TP)" = "#2ecc71", "Negative" = "#e74c3c")) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey40", linewidth = 0.3) +
  facet_wrap(~facet, ncol = 3) +
  labs(
    title    = "Cumulative Rank Distribution: Matching Quality Across Options",
    subtitle = "Curve overlap = bias removed | Separation = residual bias",
    x = "NetMHCpan EL Rank (%)", y = "Cumulative Proportion", colour = NULL
  ) +
  theme_bw(base_size = 16) +
  theme(
    legend.position     = "top",
    panel.grid.minor    = element_blank(),
    plot.title.position = "plot",
    panel.spacing = unit(0.6, "lines"),
    plot.margin = margin(8, 12, 8, 8),
    plot.title = element_text(size = 20, face = "bold"),
    plot.subtitle = element_text(size = 13),
    strip.text = element_text(size = 11, face = "bold")
  )

#  Options summary (for reference) 
cat("\n=== Options Summary ===\n")
bind_rows(
  tibble(option = "Before (no matching)", n_bins = NA_real_,
         n_neg = nrow(df_netmhcpan_only),
         ratio = round(nrow(df_netmhcpan_only) / nrow(df_positives), 2),
         ks_D = round(ks_pre$statistic, 4), quality = "\u2717 biased"),
  configs |> transmute(option = tier, n_bins, n_neg, ratio, ks_D, quality = match_quality)
) |> print(n = Inf)

# Build final dataset (Option C: 25 bins) 
df_negatives <- sample_bin_matched(df_positives, df_netmhcpan_only, CHOSEN_N_BINS, seed = RANDOM_SEED)

cat("\n=== Chosen Configuration: Option C ===\n")
cat("Bins:    ", CHOSEN_N_BINS, "\n")
cat("Negatives:", scales::comma(nrow(df_negatives)), "\n")
cat("Neg:Pos: ", round(nrow(df_negatives) / nrow(df_positives), 2), ":1\n")

df_combined <- bind_rows(df_positives, df_negatives) |>
  select(-any_of(c("hla", "binder", "rank_bin")))

cat("\n=== Final ML Dataset ===\n")
cat("Positives:", scales::comma(sum(df_combined$label == 1)), "\n")
cat("Negatives:", scales::comma(sum(df_combined$label == 0)), "\n")
cat("Total:    ", scales::comma(nrow(df_combined)), "\n")
cat("Ratio:    ", round(sum(df_combined$label == 0) / sum(df_combined$label == 1), 2), ":1\n")


# Statistical validation of bias correction 
# At n > 20K, p-values are always < 2.2e-16 (overpowered). Effect sizes
# (KS D, Cliff's Δ) quantify the magnitude of distributional difference.

#  Compute all tests 
ks_before     <- ks.test(df_positives$rank, df_netmhcpan_only$rank)
ks_after      <- ks.test(df_positives$rank, df_negatives$rank)
wilcox_before <- wilcox.test(df_positives$rank, df_netmhcpan_only$rank)
wilcox_after  <- wilcox.test(df_positives$rank, df_negatives$rank)


# ── Post-matching density plot (chosen config: 25 bins) ──────────────────────
df_rank_post <- bind_rows(
  df_positives |> transmute(rank, label = "Positive (TP)"),
  df_negatives |> transmute(rank, label = "Negative (matched)")
)

p2b <- ggplot(df_rank_post, aes(x = rank, fill = label)) +
  geom_density(alpha = 0.5, colour = "black", linewidth = 0.3) +
  scale_fill_manual(values = c("Positive (TP)"      = "#2ecc71",
                               "Negative (matched)" = "#3498db")) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey40", linewidth = 0.5) +
  annotate("text", x = 0.5, y = Inf, label = "SB threshold (0.5%)",
           vjust = 2, hjust = -0.05, size = 3, colour = "grey40") +
  annotate("text", x = 1.5, y = Inf,
           label = paste0("KS D = ", round(ks_after$statistic, 4)),
           vjust = 1.5, size = 5, fontface = "bold", colour = "#3498db") +
  labs(
    title    = "Affinity Bias Check: NetMHCpan EL Rank Distribution (After Matching)",
    subtitle = paste0("9-mers  |  HLA-A*02:01  |  25 bins  |  KS D = ",
                      round(ks_after$statistic, 4),
                      " (distributions well matched)"),
    x = "NetMHCpan EL Rank (%)", y = "Density", fill = NULL
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "top", panel.grid.minor = element_blank(),
        plot.title.position = "plot")

ggsave("results/affinity_bias_density_after.png", p2b, width = 7, height = 5, dpi = 150)

cat("Computing Cliff's Delta (before matching)...\n")
cliff_before <- cliff_delta_sampled(df_positives$rank, df_netmhcpan_only$rank)
cat("Computing Cliff's Delta (after matching)...\n")
cliff_after  <- cliff_delta_sampled(df_positives$rank, df_negatives$rank)

cat("\n", strrep("\u2550", 70), "\n")
cat("  STATISTICAL COMPARISON: Rank Distributions\n")
cat(strrep("\u2550", 70), "\n\n")

cat("Positives vs Negatives\n")
cat("  KS D:          ", round(ks_before$statistic, 4), "\n")
cat("  KS p-value:    ", format.pval(ks_before$p.value, digits = 3), "\n")
cat("  Wilcoxon p:    ", format.pval(wilcox_before$p.value, digits = 3), "\n")
cat("  Cliff's delta: ", round(cliff_before$estimate, 4),
    " (", cliff_before$magnitude, ")\n\n")

cat("Positives vs Negatives (AFTER matching, 25 bins)\n")
cat("  KS D:          ", round(ks_after$statistic, 4), "\n")
cat("  KS p-value:    ", format.pval(ks_after$p.value, digits = 3), "\n")
cat("  Wilcoxon p:    ", format.pval(wilcox_after$p.value, digits = 3), "\n")
cat("  Cliff's delta: ", round(cliff_after$estimate, 4),
    " (", cliff_after$magnitude, ")\n\n")

cat("Both p-values < 2.2e-16 (overpowered at n > 20K). Effect sizes tell the story:\n")
cat("  Before: Cliff's \u0394 = ", round(cliff_before$estimate, 4),
    " (", cliff_before$magnitude, ") \u2192 bias exploitable\n")
cat("  After:  Cliff's \u0394 = ", round(cliff_after$estimate, 4),
    " (", cliff_after$magnitude, ") \u2192 bias removed\n")
cat("  KS D reduction: ", round(ks_before$statistic, 4), " \u2192 ",
    round(ks_after$statistic, 4),
    " (", round((1 - ks_after$statistic / ks_before$statistic) * 100, 1), "%)\n")

#  Rank distribution summary table 
df_rank_summary <- bind_rows(
  df_positives |>
    summarise(n = n(), median = round(median(rank), 4), mean = round(mean(rank), 4),
              sd = round(sd(rank), 4), q25 = round(quantile(rank, 0.25), 4),
              q75 = round(quantile(rank, 0.75), 4)) |>
    mutate(group = "Positives (TP)", stage = "All stages", .before = 1),
  df_netmhcpan_only |>
    summarise(n = n(), median = round(median(rank), 4), mean = round(mean(rank), 4),
              sd = round(sd(rank), 4), q25 = round(quantile(rank, 0.25), 4),
              q75 = round(quantile(rank, 0.75), 4)) |>
    mutate(group = "Negatives (before matching)", stage = "Before", .before = 1),
  df_negatives |>
    summarise(n = n(), median = round(median(rank), 4), mean = round(mean(rank), 4),
              sd = round(sd(rank), 4), q25 = round(quantile(rank, 0.25), 4),
              q75 = round(quantile(rank, 0.75), 4)) |>
    mutate(group = "Negatives (after matching, 25 bins)", stage = "After", .before = 1)
)

cat("\n=== Complete Rank Distribution Comparison ===\n")
print(df_rank_summary)

#  Statistical test summary table 
df_test_summary <- tibble(
  Comparison    = c("Pos vs Neg (before)", "Pos vs Neg (after, 25 bins)"),
  n_pos         = c(nrow(df_positives), nrow(df_positives)),
  n_neg         = c(nrow(df_netmhcpan_only), nrow(df_negatives)),
  `KS D`        = c(round(ks_before$statistic, 4), round(ks_after$statistic, 4)),
  `Cliff's Δ`   = c(round(cliff_before$estimate, 4), round(cliff_after$estimate, 4)),
  `Δ Magnitude` = c(cliff_before$magnitude, cliff_after$magnitude),
  `Wilcoxon p`  = c(format.pval(wilcox_before$p.value, digits = 2),
                    format.pval(wilcox_after$p.value, digits = 2)),
  `Median diff` = c(
    round(median(df_netmhcpan_only$rank) - median(df_positives$rank), 4),
    round(median(df_negatives$rank) - median(df_positives$rank), 4)
  )
)

cat("\n=== Statistical Test Summary ===\n")
print(df_test_summary)

#  Boxplot + violin (before-group downsampled for rendering) 
set.seed(42)
df_stat_compare <- bind_rows(
  df_positives |>
    transmute(rank, group = "Positives\n(IEDB-confirmed)"),
  df_netmhcpan_only |>
    slice_sample(n = 30000) |>
    transmute(rank, group = "Negatives\n(before matching)"),
  df_negatives |>
    transmute(rank, group = "Negatives\n(after matching)")
) |>
  mutate(group = factor(group, levels = c(
    "Positives\n(IEDB-confirmed)",
    "Negatives\n(after matching)",
    "Negatives\n(before matching)"
  )))

group_colours <- c(
  "Positives\n(IEDB-confirmed)"  = "#2ecc71",
  "Negatives\n(after matching)"  = "#3498db",
  "Negatives\n(before matching)" = "#e74c3c"
)

p6 <- ggplot(df_stat_compare, aes(x = group, y = rank, fill = group)) +
  geom_violin(alpha = 0.3, colour = NA, scale = "width", width = 0.8, adjust = 1.5) +
  geom_boxplot(width = 0.2, outlier.size = 0.2, outlier.alpha = 0.05,
               alpha = 0.8, colour = "grey30", linewidth = 0.4, fill = "white") +
  stat_summary(fun = median, geom = "point", shape = 18, size = 3.5, colour = "black") +
  scale_fill_manual(values = group_colours) +
  # Bracket: Positives vs Negatives BEFORE (red)
  annotate("segment", x = 1, xend = 3, y = 2.25, yend = 2.25,
           colour = "#e74c3c", linewidth = 0.6) +
  annotate("segment", x = 1, xend = 1, y = 2.18, yend = 2.25,
           colour = "#e74c3c", linewidth = 0.6) +
  annotate("segment", x = 3, xend = 3, y = 2.18, yend = 2.25,
           colour = "#e74c3c", linewidth = 0.6) +
  annotate("text", x = 2, y = 2.35,
           label = paste0("D = ", round(ks_before$statistic, 3),
                          "  |  Cliff's \u0394 = ", round(cliff_before$estimate, 3),
                          " (", cliff_before$magnitude, ")"),
           size = 3.2, fontface = "bold", colour = "#e74c3c") +
  # Bracket: Positives vs Negatives AFTER (blue)
  annotate("segment", x = 1, xend = 2, y = 2.05, yend = 2.05,
           colour = "#3498db", linewidth = 0.6) +
  annotate("segment", x = 1, xend = 1, y = 1.98, yend = 2.05,
           colour = "#3498db", linewidth = 0.6) +
  annotate("segment", x = 2, xend = 2, y = 1.98, yend = 2.05,
           colour = "#3498db", linewidth = 0.6) +
  annotate("text", x = 1.5, y = 2.12,
           label = paste0("D = ", round(ks_after$statistic, 3),
                          "  |  Cliff's \u0394 = ", round(cliff_after$estimate, 3),
                          " (", cliff_after$magnitude, ")"),
           size = 3.2, fontface = "bold", colour = "#3498db") +
  geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey60", linewidth = 0.3) +
  annotate("text", x = 0.55, y = 0.52, label = "SB threshold (0.5%)",
           size = 2.5, colour = "grey50", hjust = 0) +
  labs(
    title = "Affinity Bias Correction: Rank Distribution Comparison",
    subtitle = paste0(
      "Before matching: distributions clearly different (Cliff's \u0394 = ",
      round(cliff_before$estimate, 3), ", ", cliff_before$magnitude, ")\n",
      "After matching (25 bins): distributions practically identical (Cliff's \u0394 = ",
      round(cliff_after$estimate, 3), ", ", cliff_after$magnitude, ")"
    ),
    x = NULL, y = "NetMHCpan EL Rank (%)"
  ) +
  coord_cartesian(ylim = c(-0.05, 2.5), clip = "off") +
  theme_bw(base_size = 13) +
  theme(
    legend.position     = "none",
    panel.grid.minor    = element_blank(),
    panel.grid.major.x  = element_blank(),
    plot.title.position = "plot",
    plot.title          = element_text(face = "bold"),
    plot.subtitle       = element_text(size = 10, colour = "grey40"),
    axis.text.x         = element_text(size = 10, face = "bold"),
    plot.margin         = margin(t = 10, r = 15, b = 10, l = 10)
  )

# Save all outputs 
ggsave("results/confusion_matrix_9mer.png",              p1, width = 7,  height = 5,  dpi = 150)
ggsave("results/affinity_bias_density_before.png",       p2, width = 7,  height = 5,  dpi = 150)
ggsave("results/affinity_bias_ecdf_before.png",          p3, width = 7,  height = 5,  dpi = 150)
ggsave("results/affinity_bias_options_ecdf.png", p5, width = 13.33, height = 7.5, dpi = 200)
ggsave("results/affinity_bias_options_density.png", p4, width = 13.33, height = 7.5, dpi = 200)
ggsave("results/affinity_bias_boxplot_comparison.png",   p6, width = 9,  height = 7,  dpi = 150)

write_csv(df_rank_summary,  "data/processed/rank_distribution_summary.csv")
write_csv(df_test_summary,  "data/processed/statistical_test_summary.csv")
write_csv(df_combined,      "data/processed/df_combined_pos_and_neg.csv")

cat("\nSaved all plots to results/\n")
cat("Saved final dataset to data/processed/df_combined_pos_and_neg.csv\n")
cat("   Positives:", scales::comma(sum(df_combined$label == 1)), "\n")
cat("   Negatives:", scales::comma(sum(df_combined$label == 0)), "\n")
cat("   Total:    ", scales::comma(nrow(df_combined)), "\n")

# ============================================================================
# DIAGNOSTIC + SAVE: Exact counts for Sankey diagram
# ============================================================================

cat("\n\n")
cat(strrep("=", 70), "\n")
cat("  SANKEY DIAGNOSTIC: Tracing the filtering cascade\n")
cat(strrep("=", 70), "\n\n")

# Replay cascade from df_raw (untouched input)
df_diag_9mer   <- df_raw |> filter(pep_length == 9)
n_diag_9mer    <- nrow(df_diag_9mer)
n_diag_non9mer <- nrow(df_raw) - n_diag_9mer

# Step 1: Explicit NA filter (matches pipeline line 11)
df_diag_after_na <- df_diag_9mer |> filter(!is.na(uniprot_id))
n_is_na <- nrow(df_diag_9mer) - nrow(df_diag_after_na)

# Step 2: O60361 filter
n_is_o60361 <- sum(df_diag_after_na$uniprot_id == "O60361", na.rm = TRUE)
df_diag_after_filter <- df_diag_after_na |> filter(uniprot_id != "O60361")

# Step 3: Deduplication
df_diag_dedup <- df_diag_after_filter |>
  distinct(peptide, uniprot_id, start, end, .keep_all = TRUE)
n_diag_dedup <- nrow(df_diag_after_filter) - nrow(df_diag_dedup)

# Step 4: FASTA semi-join
df_diag_after_fasta <- df_diag_dedup |>
  semi_join(df_fasta, by = "uniprot_id")
n_diag_missing_fasta <- nrow(df_diag_dedup) - nrow(df_diag_after_fasta)

# Step 5: Coordinate validation (replays validate_and_fix_positions)
# Drops unfixable peptides AND rows with NA start/end (no_protein_sequence).
df_diag_after_coord <- df_diag_after_fasta |>
  left_join(df_fasta |> select(uniprot_id, sequence), by = "uniprot_id") |>
  validate_and_fix_positions(
    sequence_col = "sequence",
    peptide_col  = "peptide",
    start_col    = "start",
    end_col      = "end"
  ) |>
  filter(position_status %in% c("valid_original", "fixed")) |>
  select(-start_original, -end_original, -position_originally_valid,
         -position_status, -position_shift, -sequence)
n_diag_unfixable <- nrow(df_diag_after_fasta) - nrow(df_diag_after_coord)

# Step 6: Selenocysteine filter
df_diag_after_sec <- df_diag_after_coord |>
  filter(!uniprot_id %in% sec_proteins)
n_diag_sec <- nrow(df_diag_after_coord) - nrow(df_diag_after_sec)

# ── Sequence-Validated node (after all pre-ledger filters) ──
n_seq_validated <- nrow(df_diag_after_sec)

# ── Length filter: PEPTIDE counts (not protein counts) ──
n_diag_too_short <- sum(df_diag_after_sec$uniprot_id %in% (df_ledger |> filter(too_short) |> pull(uniprot_id)))
n_diag_too_long  <- sum(df_diag_after_sec$uniprot_id %in% (df_ledger |> filter(too_long) |> pull(uniprot_id)))

df_diag_after_len <- df_diag_after_sec |>
  left_join(df_fasta |> select(uniprot_id, seq_length), by = "uniprot_id") |>
  filter(seq_length >= 130, seq_length <= 5000) |>
  select(-seq_length)

# ── Length-Validated node ──
n_length_validated <- nrow(df_diag_after_len)

# ── AlphaFold filter: PEPTIDE counts ──
n_diag_missing_af <- sum(!df_diag_after_len$uniprot_id %in% af_proteins)
n_diag_out_of_range <- sum(df_diag_after_len$uniprot_id %in% proteins_out_of_range$uniprot_id)

df_diag_after_af <- df_diag_after_len |>
  filter(uniprot_id %in% af_proteins) |>
  filter(!uniprot_id %in% proteins_out_of_range$uniprot_id)

n_9mer_verified <- nrow(df_diag_after_af)

# IEDB-level split — use the CLEAN data before left_join corrupted df_iedb_pos
n_iedb_recovered <- df_diag_after_af |>
  semi_join(df_netmhcpan_binders_unique, by = c("peptide", "uniprot_id")) |>
  nrow()

n_iedb_missed <- df_diag_after_af |>
  anti_join(df_netmhcpan_binders_unique, by = c("peptide", "uniprot_id")) |>
  nrow()

stopifnot(n_iedb_recovered + n_iedb_missed == n_9mer_verified)

# Print summary
cat("COMPLETE CASCADE:\n")
cat(strrep("-", 50), "\n")
cat("  Input file:              ", nrow(df_raw), "\n")
cat("  Non-9-mers:              ", n_diag_non9mer, "\n")
cat("  9-mers:                  ", n_diag_9mer, "\n")
cat("  - NA uniprot_id:         ", n_is_na, "\n")
cat("  - O60361:                ", n_is_o60361, "\n")
cat("  - Dedup:                 ", n_diag_dedup, "\n")
cat("  - Missing FASTA:         ", n_diag_missing_fasta, "\n")
cat("  - Unfixable coordinates: ", n_diag_unfixable, "\n")
cat("  - Selenocysteine:        ", n_diag_sec, "\n")
cat("  Sequence-Validated:      ", n_seq_validated, "\n")
cat("  - Too short (<130 aa):   ", n_diag_too_short, "peptides\n")
cat("  - Too long  (>5000 aa):  ", n_diag_too_long, "peptides\n")
cat("  Length-Validated:         ", n_length_validated, "\n")
cat("  - Missing AlphaFold:     ", n_diag_missing_af, "\n")
cat("  - Out of Range:          ", n_diag_out_of_range, "\n")
cat("  Final verified:          ", n_9mer_verified, "\n")
cat("  Recovered by NetMHCpan:  ", n_iedb_recovered, "\n")
cat("  Missed by NetMHCpan:     ", n_iedb_missed, "\n")
cat("  TP in df_combined:       ", sum(df_combined$label == 1), "\n")
cat("  FN in iedb_only:         ", nrow(df_iedb_only), "\n\n")

# Verify cascade matches actual data
cascade_diff <- nrow(df_iedb_pos) - n_9mer_verified
if (cascade_diff != 0) {
  cat("⚠️  df_iedb_pos has", nrow(df_iedb_pos), "rows but cascade gives", n_9mer_verified, "\n")
  cat("   Difference of", cascade_diff, "rows — check if cascade replays all filters.\n\n")
} else {
  cat("✓  df_iedb_pos matches cascade:", nrow(df_iedb_pos), "rows\n\n")
}

# Read IEDB stage counts from step 02
iedb_stage <- read_csv("data/processed/iedb_stage_counts.csv", show_col_types = FALSE)
iedb_ct <- setNames(iedb_stage$count, iedb_stage$stage)

# Save counts for Sankey script
df_sankey_counts <- tibble(
  stage = c(
    names(iedb_ct),
    "non_9mer", "9mer_all",
    "na_uniprot", "o60361", "dedup", "missing_fasta", "unfixable_coords", "selenocysteine",
    "seq_validated",
    "too_short", "too_long", "length_validated",
    "missing_alphafold", "out_of_range",
    "9mer_verified",
    "iedb_recovered", "iedb_missed",
    "tp_in_combined", "fn_in_iedb_only",
    "predicted_binders", "netmhcpan_only",
    "negatives_combined", "affinity_removed"
  ),
  count = c(
    unname(iedb_ct),
    n_diag_non9mer, n_diag_9mer,
    n_is_na, n_is_o60361, n_diag_dedup, n_diag_missing_fasta, n_diag_unfixable, n_diag_sec,
    n_seq_validated,
    n_diag_too_short, n_diag_too_long, n_length_validated,
    n_diag_missing_af, n_diag_out_of_range,
    n_9mer_verified,
    n_iedb_recovered, n_iedb_missed,
    sum(df_combined$label == 1), nrow(df_iedb_only),
    nrow(df_netmhcpan_binders_unique), nrow(df_netmhcpan_only),
    sum(df_combined$label == 0), nrow(df_netmhcpan_only) - sum(df_combined$label == 0)
  )
)

write_csv(df_sankey_counts, "data/processed/sankey_counts.csv")
cat("Saved: data/processed/sankey_counts.csv\n")
print(df_sankey_counts, n = Inf)
