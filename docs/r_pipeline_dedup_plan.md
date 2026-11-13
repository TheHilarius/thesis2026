# Task 1 (Oliver, R): End-of-pipeline dedup of identical context windows

Companion docs: `docs/r_pipeline_flank_flag_plan.md` (Task 2 — FLANK_SIZE
flag), `docs/python_allele_support_plan.md` (Task 3 — Python-side allele
readiness, ours).

**Scope:** one filtering block at the END of the R pipeline. No change to
negative sampling, no change to any earlier step. Future work (explicitly
NOT this task): moving the filter before negative downsampling — see
"Future" at the bottom.

## Why

Annotation duplicates (isoforms, paralogs, SwissProt/TrEMBL pairs) create
rows whose **model input window is byte-identical**
(`full_context` = `n_flank | peptide | c_flank`) while labels disagree —
the label reflects per-protein detection evidence, so the conflict is
annotation noise the model cannot resolve (304 windows / 697 rows in the
current 0201 data). Same-label duplicate windows are equally redundant.
A/B test (XGB + BLOSUM50, nested CV) on the cleaned variant:
0.7633 -> 0.7647, paired DeLong delta +0.0013, p = 0.156 (ns) —
performance-neutral. This is data hygiene, not a modelling change.

**Rule (exactly what `make_variant_dataset.py --variant remove_dup_windows`
does, so outputs must match row-for-row):**

- Group rows by `full_context` (the window: 29-mer today, 19-mer under the
  FLANK_SIZE flag — the rule is width-agnostic).
- Keep **one** row per window:
  - if the window has any positive row -> keep the **first positive** row;
  - else -> keep the **first** row.
- Drop everything else.
- The same peptide with **different** (non-identical) flanks is KEPT —
  those are distinct local contexts. Only byte-identical windows collapse.

## Where

`src/pipeline/r/07_alpha_fold_features.R`, immediately **before** the
final `write_csv(df_all, ...)` (line 215 today: `"data/processed/df_all.csv"`).

Placement rationale: **after all feature joins, immediately before the
final write.** Step 7 is the last writer of `df_all.csv`, so one block
covers everything downstream (01_datasplit, 08_aa_encoding, embeddings,
modelling). A separate post-hoc script is rejected: an extra step nobody
runs, and the next R regeneration would silently resurrect duplicates.

**Allele integration (coordinate with your 07 `pp()` switch):** the block
itself is allele-agnostic — it keys on `full_context`; only the output
path differs. Write it against the allele-aware target state:

```r
# your 07 switch:
write_csv(df_all, pp("df_all.csv", ALLELE_TAG))
# -> the dedup block sits directly above that line, unchanged for both alleles
```

If the dedup lands before your 07 switch, keep the literal path and move
nothing — the block has no path knowledge.

## Implementation sketch (R)

```r
# ── Dedup identical context windows (one row per window) ──
# Window carries both labels -> keep the positive; else keep first row.
n_before_dedup <- nrow(df_all)
ctx <- df_all$full_context
is_pos <- df_all$label == 1

first_any <- !duplicated(ctx)
first_pos <- rep(FALSE, n_before_dedup)
first_pos[which(is_pos)[!duplicated(ctx[is_pos])]] <- TRUE
has_pos <- ctx %in% unique(ctx[is_pos])
keep <- ifelse(has_pos, first_pos, first_any)

df_all <- df_all[keep, ]

n_dropped <- n_before_dedup - nrow(df_all)
cat("Context-window dedup: dropped", n_dropped, "rows (",
    n_before_dedup, "->", nrow(df_all), ")\n")

# Hard guards (any allele):
stopifnot("duplicate context windows remain" =
            !any(duplicated(df_all$full_context)))
ctx_widths <- unique(nchar(df_all$full_context))
stopifnot("full_context must be a single width of 19 or 29" =
            length(ctx_widths) == 1 && ctx_widths %in% c(19, 29))
# NOTE: once Task 2 (FLANK_SIZE flag) lands, tighten the width assert to
# nchar(full_context) == 2 * FLANK_SIZE + 9 — see flank_flag doc.
```

## Guards

**Both alleles (hard):**

| check | expected |
|---|---|
| `duplicated(full_context)` after | 0 |
| `nchar(full_context)` | single width, 19 or 29 |

**0201 only** (`if (ALLELE_TAG == "hla0201")`, hard-assert against drift):

| check | expected |
|---|---|
| rows before -> after | 46,617 -> **45,510** |
| total dropped | **1,107** |
| of which: label-conflicting windows resolved | 304 windows (352 negative rows dropped) |
| of which: duplicate positive rows dropped | 361 |
| of which: same-label duplicate negative rows dropped | 394 |
| label balance after | 21,240 pos / 24,270 neg |

**0101 only:** `cat` the dropped count, do **not** pin numbers — the 0101
dataset does not exist yet (FASTA pending). Pin them after its first run.

## Acceptance test (0201 — must be row-identical to the Python reference)

The cleaned dataset already exists from the Python side
(`make_variant_dataset.py --variant remove_dup_windows`). After adding the
R block and re-running steps 5–7:

```r
ref <- readr::read_csv("data/processed/df_all_remove_dup_windows.csv",
                       show_col_types = FALSE)
stopifnot(nrow(df_all) == nrow(ref))
stopifnot(identical(df_all$full_context, ref$full_context))
stopifnot(identical(df_all$label, ref$label))
stopifnot(identical(df_all$peptide, ref$peptide))
```

If any stopifnot fails: the R filter and the Python rule diverged —
fix before merging.

**0101 acceptance:** first run only invariants (guards above); pin the
expected row counts after the dataset is produced.

## Regeneration checklist (when this lands and the pipeline is next rerun)

1. R steps 5–7 (filter lives in 7) -> clean `df_all.csv`
2. `src/pipeline/python/01_datasplit.py` -> new folds
3. `src/pipeline/r/08_aa_encoding.R` -> row-aligned encoding CSVs
   (they key off `df_all.csv`, so they follow automatically)
4. **Raw embeddings re-run** (ESM-C / ESM-IF / MPNN / pLDDT) — unavoidable
   with any row change; this is why the filter is NOT being moved earlier
   today
5. `03_prepare_embeddings.py` -> prepared H5s
6. Model runs + comparisons

Note: the existing `remove_dup_windows` variant artifacts (split CSV,
encoding CSVs, subset prepared H5s) already implement exactly this rule —
steps 1–5 reproduce them as the canonical (non-variant) dataset.

## Future (NOT this task)

- **Filter before negative downsampling** (step 4): prevents conflicting
  negatives at the source instead of dropping them post-hoc. Requires
  re-running `sample_bin_matched` -> different sampled negatives ->
  a completely new dataset + full re-embedding. Deferred until a
  deliberate dataset version bump.
- Related, previously reverted, separate decision: peptide-level
  `anti_join(df_iedb_pos, by = "peptide")` in step 4's negative pool
  (excludes any negative whose peptide is positive in ANY protein —
  superset of the window rule, also affects different-flank rows).

Line numbers verified against tree `f9a26b3` (2026-11-13): 07 untouched by
the allele pull — `write_csv(df_all, ...)` still line 215.
