library(tidyverse)
source("src/pipeline/r/functions.R")
set_working_directory()

# ============================================================================
# ASSARSSON EXTERNAL-VALIDATION PREPARATION (HLA-A*02:01)
#
# One-shot prep for the Assarsson et al. vaccinia epitope set, following the
# same conventions as 04 (coordinate validation) / 05 (flanks + cleavage) /
# 08 (AA encodings). Produces:
#   data/processed/df_assarsson.csv            - full feature table, fold = -1
#   data/processed/df_blosum50_assarsson.csv   - BLOSUM50 solo CSV (08 format)
#
# Labels: classification Dominant/Subdominant/Cryptic -> positive (1),
#         Negative -> 0  (agreed with user 2026-11-09).
# ============================================================================

POSITION_COLS <- c(
  paste0("N", 10:1),    # N10..N1
  paste0("P", 1:9),     # P1..P9
  paste0("C", 1:10)     # C1..C10
)

# ----------------------------------------------------------------------------
# STEP 1: LOAD ASSARSSON TABLE + PROTEIN ACCESSIONS
# ----------------------------------------------------------------------------

cat("=== STEP 1: Load Assarsson data ===\n")

df_raw <- read_csv("data/raw/external/assarsson/merged_assarsson_data.csv",
                   show_col_types = FALSE)

# Accession from mol_parent_iri (…/uniprot/P06440.1 -> P06440); rows without
# an IRI get their accession via the VACWR locus code fetched by
# src/util/fetch_assarsson_fastas.py -> assarsson_orf_accession.csv
df_orf_map <- read_csv("data/raw/external/assarsson/assarsson_orf_accession.csv",
                       show_col_types = FALSE) |>
  rename(uniprot_from_orf = uniprot_id)

df_assarsson <- df_raw |>
  mutate(
    uniprot_id = if_else(
      mol_parent_iri %in% c(NA, "NA", ""),
      NA_character_,
      str_extract(str_extract(mol_parent_iri, "[^/]+$"), "^[A-Z0-9]+") |>
        str_remove("\\.[0-9]+$")
    ),
    vacwr_code = str_extract(orf, "VACWR[0-9]+")
  ) |>
  left_join(df_orf_map, by = "vacwr_code") |>
  mutate(uniprot_id = coalesce(uniprot_id, uniprot_from_orf)) |>
  select(-uniprot_from_orf) |>
  rename(peptide = sequence, start = start_pos, end = end_pos)

cat("Rows:", nrow(df_assarsson), "\n")
cat("Rows without accession:", sum(is.na(df_assarsson$uniprot_id)), "\n")
stopifnot("every peptide needs an accession" = !any(is.na(df_assarsson$uniprot_id)))

# ----------------------------------------------------------------------------
# STEP 2: JOIN PROTEIN SEQUENCES (all FASTAs in one dir, like 05 L15-35)
# ----------------------------------------------------------------------------

cat("\n=== STEP 2: Join protein sequences ===\n")

fasta_files <- list.files("data/raw/fasta/assarsson_protein_fastas",
                          pattern = "\\.fasta$", full.names = TRUE)
cat("FASTA files:", length(fasta_files), "\n")

df_fasta <- map(fasta_files, read_fasta_df) |>
  list_rbind() |>
  mutate(
    accession  = str_extract(header, "(?<=\\|)[^|]+(?=\\|)"),
    uniprot_id = accession
  ) |>
  filter(!is.na(uniprot_id)) |>
  distinct(uniprot_id, .keep_all = TRUE) |>
  select(uniprot_id, sequence)

cat("Unique proteins:", nrow(df_fasta), "\n")

df_merged <- df_assarsson |>
  left_join(df_fasta, by = "uniprot_id")

cat("Peptides with matched sequence:",
    sum(!is.na(df_merged$sequence)), "/", nrow(df_merged), "\n")
stopifnot("all peptides must match a protein sequence" =
            !any(is.na(df_merged$sequence)))

# ----------------------------------------------------------------------------
# STEP 3: VALIDATE/FIX COORDINATES (04 L56-74 convention)
# ----------------------------------------------------------------------------

cat("\n=== STEP 3: Validate positions ===\n")

# 33 rows carry no coordinates in the paper's export. validate_and_fix_positions()
# propagates NA conditions to NA status, so seed them with 1/1 — the function
# then falls through to its search branch and recovers the real position.
df_merged <- df_merged |>
  mutate(
    start = replace_na(as.integer(start), 1L),
    end   = replace_na(as.integer(end), 1L)
  )

df_pos <- df_merged |>
  validate_and_fix_positions(
    sequence_col = "sequence",
    peptide_col  = "peptide",
    start_col    = "start",
    end_col      = "end"
  )

print(count(df_pos, position_status))
n_unfixable <- sum(df_pos$position_status == "unfixable" |
                     is.na(df_pos$position_status), na.rm = TRUE)
df_pos <- df_pos |>
  filter(position_status %in% c("valid_original", "fixed")) |>
  select(-start_original, -end_original, -position_originally_valid,
         -position_status, -position_shift)
cat("Dropped unfixable:", n_unfixable, "-> rows:", nrow(df_pos), "\n")
# 1 expected drop: MMCYFLTATL (paper) vs UniProt P16712 has MMCYFLTATP
# at pos 229 — 1-residue discrepancy, dropped per 04's unfixable convention.
stopifnot("unexpected row loss in position validation" = nrow(df_pos) >= 100)

# ----------------------------------------------------------------------------
# STEP 4: FLANKING REGIONS + PAD COUNTS (05 L48-94 convention)
# ----------------------------------------------------------------------------

cat("\n=== STEP 4: Extract flanking regions ===\n")

N_FLANK_SIZE <- 10
C_FLANK_SIZE <- 10

df_with_flanks <- df_pos |>
  extract_flanking_regions(
    sequence_col = "sequence",
    peptide_col  = "peptide",
    start_col    = "start",
    end_col      = "end",
    n_flank_size = N_FLANK_SIZE,
    c_flank_size = C_FLANK_SIZE
  ) |>
  mutate(
    rel_distance_from_n_terminus = distance_from_n_terminus / protein_length,
    rel_distance_from_c_terminus = distance_from_c_terminus / protein_length
  ) |>
  select(-c(distance_from_n_terminus, distance_from_c_terminus)) |>
  group_by(uniprot_id) |>
  mutate(
    n_pad_len = max(0L, N_FLANK_SIZE - n_flank_len_real),
    c_pad_len = max(0L, C_FLANK_SIZE - c_flank_len_real)
  ) |>
  ungroup()

cat("Row count:", nrow(df_with_flanks), "\n")
stopifnot("peptide lengths must be 9 or 10" =
            all(nchar(df_with_flanks$peptide) %in% c(9, 10)))
stopifnot("full_context must be 29 (9mers) or 30 (10mers) chars" =
            all(nchar(df_with_flanks$full_context) %in% c(29, 30)))
stopifnot("n_flank_raw must contain no X" =
            !any(grepl("X", df_with_flanks$n_flank_raw)))
stopifnot("c_flank_raw must contain no X" =
            !any(grepl("X", df_with_flanks$c_flank_raw)))
cat("  full_context lengths:", paste(sort(unique(nchar(df_with_flanks$full_context))), collapse = ", "), "\n")

# ----------------------------------------------------------------------------
# STEP 5: CLEAVAGE POSITIONS (05 L102-107 convention)
# ----------------------------------------------------------------------------

cat("\n=== STEP 5: Extract cleavage positions ===\n")

df_with_cleavage <- df_with_flanks |>
  extract_cleavage_positions(
    peptide_col = "peptide",
    n_flank_col = "n_flank",
    c_flank_col = "c_flank"
  )

stopifnot("no NA in position columns" =
            !any(is.na(df_with_cleavage |> select(all_of(POSITION_COLS)))))

# ----------------------------------------------------------------------------
# STEP 6: JOIN NetMHCpan PREDICTIONS (copied outputs, HLA-A*02:01)
# ----------------------------------------------------------------------------

cat("\n=== STEP 6: Join NetMHCpan predictions ===\n")

nmhc_cols <- c("Pos", "Peptide", "ID", "core", "icore", "Score", "Rank", "Ave", "NB")
df_nm <- read_csv("data/raw/external/assarsson/NetMHCpan_assarsson.csv",
                  skip = 2, col_names = nmhc_cols, show_col_types = FALSE) |>
  transmute(
    uniprot_id = str_split_i(ID, "_", 2),
    peptide    = Peptide,
    nm_score   = Score,
    rank       = Rank
  )

ba_cols <- c("Pos", "Peptide", "ID", "core", "icore", "Score", "Rank",
             "BA_score", "BA_Rank", "Ave", "NB")
df_ba <- read_csv("data/raw/external/assarsson/NetMHCpan_BAmode_assarsson.csv",
                  skip = 2, col_names = ba_cols, show_col_types = FALSE) |>
  transmute(
    uniprot_id = str_split_i(ID, "_", 2),
    peptide    = Peptide,
    ba_score   = BA_score,
    ba_rank    = BA_Rank
  )

df_joined <- df_with_cleavage |>
  left_join(df_nm,  by = c("uniprot_id", "peptide")) |>
  left_join(df_ba, by = c("uniprot_id", "peptide"))

cat("Rows missing NetMHCpan rank:", sum(is.na(df_joined$rank)), "\n")
cat("Rows missing BA rank:",        sum(is.na(df_joined$ba_rank)), "\n")
stopifnot("every peptide must have NetMHCpan + BA ranks" =
            !any(is.na(df_joined$rank) | is.na(df_joined$ba_rank)))

# ----------------------------------------------------------------------------
# STEP 7: LABELS + FOLD CONVENTION
# ----------------------------------------------------------------------------

cat("\n=== STEP 7: Labels ===\n")

df_joined <- df_joined |>
  mutate(
    label = case_when(
      classification %in% c("Dominant", "Subdominant", "Cryptic") ~ 1L,
      classification == "Negative"                                ~ 0L,
      TRUE                                                        ~ NA_integer_
    ),
    true_label = if_else(label == 1L, "positive", "negative"),
    fold = -1L,               # external set: no CV bucket
    pep_length = nchar(peptide)
  )

print(count(df_joined, classification, true_label))
stopifnot("labels must not be NA" = !any(is.na(df_joined$label)))
stopifnot("NetMHCpan joins must not duplicate rows" =
            nrow(df_joined) == nrow(df_pos))

write_csv(df_joined, "data/processed/df_assarsson.csv")
cat("\nSaved: data/processed/df_assarsson.csv (", nrow(df_joined), "rows x",
    ncol(df_joined), "cols)\n", sep = "")

# ----------------------------------------------------------------------------
# STEP 8: BLOSUM50 ENCODING (08 L77-92 convention, solo CSV only)
# ----------------------------------------------------------------------------

cat("\n=== STEP 8: BLOSUM50 encoding ===\n")

cat("Encoding", length(POSITION_COLS), "positions x", ncol(BLOSUM50),
    "dims =", length(POSITION_COLS) * ncol(BLOSUM50), "features\n")

features <- map_dfc(POSITION_COLS, \(pos) {
  matrix_encode_position(df_joined, pos, BLOSUM50)
})

# Validation: NA positions must encode as all-zero vectors (08 L84-91)
abs_sums <- map_dfc(POSITION_COLS, \(pos) {
  cols <- paste0(pos, "_", colnames(BLOSUM50))
  tibble(!!pos := rowSums(abs(features[, cols])))
})
zero_counts <- abs_sums |>
  summarise(across(everything(), \(x) sum(x == 0))) |>
  pivot_longer(everything(), names_to = "position", values_to = "n_zero") |>
  filter(n_zero > 0)
stopifnot("no all-zero (NA) position rows expected" = nrow(zero_counts) == 0)
cat("Validation passed: no NA positions.\n")

df_solo <- bind_cols(
  df_joined |> select(peptide, label),
  features
)
write_csv(df_solo, "data/processed/df_blosum50_assarsson.csv")
cat("Saved: data/processed/df_blosum50_assarsson.csv (",
    nrow(df_solo), "rows x", ncol(df_solo), "cols)\n", sep = "")

# ----------------------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------------------

cat("\n=== Assarsson Preparation Summary ===\n")
cat("Rows:", nrow(df_joined), "\n")
cat("Labels: positive =", sum(df_joined$label == 1),
    ", negative =", sum(df_joined$label == 0), "\n")
cat("Peptide lengths:", paste(names(table(df_joined$pep_length)),
                              table(df_joined$pep_length),
                              sep = "mer=", collapse = ", "), "\n")
cat("NetMHCpan rank range:",
    round(range(df_joined$rank, na.rm = TRUE), 4), "\n")
cat("Unique proteins:", n_distinct(df_joined$uniprot_id), "\n")
cat("\nDone.\n")
