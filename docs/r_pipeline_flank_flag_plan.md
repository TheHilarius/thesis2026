# Task 2 (Oliver, R): `FLANK_SIZE` flag — flank5 (19-mer) default, flank10 opt-in

Companion docs: `docs/r_pipeline_dedup_plan.md` (Task 1 — window dedup),
`docs/python_allele_support_plan.md` (Task 3 — Python-side allele
readiness, ours).

**Goal:** flank width becomes a pipeline parameter instead of a hardcoded 10.
**Default `FLANK_SIZE=5`** (19-mer window — thesis direction: all recent
comparisons favor 19mer); `FLANK_SIZE=10` reproduces the legacy 29-mer dataset.

**Usage:** `FLANK_SIZE=10 Rscript src/pipeline/r/run_pipeline.R` (unset = 5).
Validate: stop if the value is not in `{5, 10}`.

## Allele interplay (new since the allele pull `f9a26b3`)

The two changes are **orthogonal and compose** — no ordering constraint
between them:

- **Allele** = positional Rscript arg (`hla0201|hla_a0101`) -> `pp()` paths.
- **Flag** = env var `FLANK_SIZE` -> window width, read per Rscript.

Both are read at the top of each step independently; a run is fully
described by `(allele, FLANK_SIZE)`. Nothing in this task touches allele
paths, and nothing in the allele work touches widths.

## Files to change (all in `src/pipeline/r/`)

1. **`05_protein_features.R`** — source of truth:
   - `N_FLANK_SIZE <- 10` / `C_FLANK_SIZE <- 10` (**lines 53–54** after the
     allele pull; was 48–49) -> both read from
     `as.integer(Sys.getenv("FLANK_SIZE", "5"))`
   - `stopifnot("full_context must be 29 chars")` (**line 88**; was 83) ->
     dynamic: `nchar(full_context) == 2 * FLANK_SIZE + 9` (19 or 29),
     message states the expected width
   - pad calcs (**lines 80–81**; was 75–76) already use the constants ✓
   - the allele switch (`ALLELE_TAG <- parse_allele_arg()`) added above is
     untouched by this task
2. **`functions.R` → `extract_cleavage_positions()` (line 480)** — the
   positional AA columns. The column set is hardcoded (`N10 … N1`,
   `C1 … C10`, ~lines 512–548) but the `make_n_pos(i)` / `make_c_pos(i)`
   helpers (line 497) are already templated with `!!i` — replace the literal
   columns with a loop over `FLANK_SIZE` generating `N{L}..N1` / `C1..C{L}`.
   (`extract_flanking_regions()` is already parameterized — pass the flag.)
   Refs unchanged by the pull (allele helpers were appended at EOF, line 1526+).
3. **`08_aa_encoding.R`** — `paste0("N", 10:1)` / `paste0("C", 1:10)`
   (lines 6, 8) -> `N{L}:1` / `C{1:L}` from the same flag (encoding CSVs
   must be width-matched or every downstream join breaks).
   08 is Oliver's next edit anyway (pp() + allele anchor map) — land both
   in the same pass.
4. **`10_assarsson_preparation.R`** — same flag (constants lines 130–131,
   column vectors lines 19–21): the external eval set must be extracted at
   the width the model was trained on.
5. **`run_pipeline.R`** — read the flag once, print a banner
   (`FLANK_SIZE = 5 → 19-mer window`), pass to the steps. Also the R
   gotcha: each step is a **separate `Rscript`**, so `Sys.setenv(FLANK_SIZE = ...)`
   is required for children to inherit it. Land together with your allele
   loop (`for allele in c(...)` + steps 02/08).

## Guards

- `FLANK_SIZE %in% c(5, 10)` else stop.
- Dynamic full-context length assert (19 or 29).
- Print window width at the start of steps 5 / 7 / 8.
- Task 1 (dedup) interplay: its width assert should be tightened to the
  same `2 * FLANK_SIZE + 9` once this flag lands (it currently accepts
  single-width 19 or 29 — see dedup doc).

## Acceptance tests

- `FLANK_SIZE=10` rerun of steps 5 → 7 → 8 must be **byte-identical** to
  current `df_all.csv` + encoding CSVs (zero-regression proof).
- `FLANK_SIZE=5` must be **row-identical** to the existing reference
  `data/processed/df_all_flank5.csv` (+ `df_all_with_folds_flank5.csv`)
  produced by `reduce_29mer_to_19mer.py` — compare `full_context`,
  `label`, `peptide`, and the `N5..N1` / `C1..C5` columns (same stopifnot
  pattern as the dedup doc).

## Sequencing — coordinate before merging the default flip

**Ride these three edits** (your stated next move — allele and flag land
together, one session per file):

| Your next edit | Your part | Our part (already specced) |
|---|---|---|
| `07_alpha_fold_features.R` | `pp()`/arg switch | dedup block (dedup doc) |
| `08_aa_encoding.R` | `pp()` + allele anchor map | `FLANK_SIZE` dynamic `N{L}:1`/`C{1:L}` columns |
| `run_pipeline.R` | allele loop + steps 02/08 | `FLANK_SIZE` read + `Sys.setenv` + banner |

**Python side must flip in the same window as the default change.** It
currently hardcodes 29-mer:

- `config.py`: `"window": 29` per embedding source; `POSITION_AA_COLS`
  (`N10..N1`, `C1..C10`)
- `04_modelling.py`: **`n_pad`/`c_pad` computed with hardcoded `10`**
  (lines ~1786–1793) — already wrong on flank5 data today; only affects
  runs with the `handcrafted` component selected
- `reduce_29mer_to_19mer.py`, the whole `THESIS_VARIANT=flank5` machinery,
  audits like `12_indexing_audit` (`WINDOW_LEN=29`)

Two safe options:

- **(a) recommended:** land the flag with the mechanism fully working but
  merged behind `FLANK_SIZE=10` default; the default flip to 5 happens in
  a coordinated change together with the Python-side update (our task —
  see `docs/python_allele_support_plan.md` for the path-pattern precedent
  and the coordination checklist).
- **(b)** flip both sides in one coordinated merge.

`01_datasplit.py` is already width-agnostic (infers 19/29 from the data)
— no change needed there.

No re-embedding is needed for flank5 itself: rows are unchanged (same
peptides, same order) and the 19-mer view is a slot-slice of the existing
29-slot embeddings (exactly how `esmc300m_win_padtoken_prepared_flank5.h5`
was built).

## Code snippets (before → after)

### 1. `05_protein_features.R`

```r
# BEFORE (lines 53–54; was 48–49 before the allele pull)
N_FLANK_SIZE <- 10
C_FLANK_SIZE <- 10

# AFTER
FLANK_SIZE <- as.integer(Sys.getenv("FLANK_SIZE", "5"))
if (!FLANK_SIZE %in% c(5L, 10L)) {
  stop("FLANK_SIZE must be 5 or 10, got: ", Sys.getenv("FLANK_SIZE"))
}
N_FLANK_SIZE <- FLANK_SIZE
C_FLANK_SIZE <- FLANK_SIZE
cat("\nFLANK_SIZE =", FLANK_SIZE, "->", 2 * FLANK_SIZE + 9, "-mer window\n\n")
```

```r
# BEFORE (line 88; was 83)
stopifnot("full_context must be 29 chars" =
            all(nchar(df_with_flanks$full_context) == 29))

# AFTER
win_len <- 2 * FLANK_SIZE + 9
stopifnot(sprintf("full_context must be %d chars", win_len) =
            all(nchar(df_with_flanks$full_context) == win_len))
```

### 2. `functions.R` → `extract_cleavage_positions()` (lines 480–552)

Note: the `make_n_pos(i)` helper already exists (line 497) but is
currently **unused** — the literal `N10…N1` / `C1…C10` mutates were written
by hand instead. The fix puts the helpers to work:

```r
# AFTER — new parameter + dynamic position columns
extract_cleavage_positions <- function(data,
                                       peptide_col = "peptide",
                                       n_flank_col = "n_flank",
                                       c_flank_col = "c_flank",
                                       flank_size = as.integer(Sys.getenv("FLANK_SIZE", "5"))) {
  # builds N-position expr: N1 = residue nearest the cut (unchanged logic)
  make_n_pos <- function(i) {
    rlang::expr(
      if_else(n_flank_len >= !!i,
              substr(.data[[n_flank_col]], n_flank_len - !!(i - 1), n_flank_len - !!(i - 1)),
              NA_character_)
    )
  }
  make_c_pos <- function(i) {
    rlang::expr(
      if_else(c_flank_len >= !!i,
              substr(.data[[c_flank_col]], !!i, !!i),
              NA_character_)
    )
  }
  # N{L}..N1 (left -> right toward the cut) and C1..C{L}
  n_pos_exprs <- rlang::set_names(
    lapply(seq(flank_size, 1), make_n_pos), paste0("N", seq(flank_size, 1)))
  c_pos_exprs <- rlang::set_names(
    lapply(seq_len(flank_size), make_c_pos), paste0("C", seq_len(flank_size)))

  data |>
    mutate(
      pep_len     = nchar(.data[[peptide_col]]),
      n_flank_len = nchar(.data[[n_flank_col]]),
      c_flank_len = nchar(.data[[c_flank_col]])
    ) |>
    mutate(!!!n_pos_exprs) |>          # REPLACES the literal N10..N1 block (513–524)
    mutate(                             # P1..P9 unchanged
      P1 = substr(.data[[peptide_col]], 1, 1),
      P2 = substr(.data[[peptide_col]], 2, 2),
      P3 = substr(.data[[peptide_col]], 3, 3),
      P4 = substr(.data[[peptide_col]], 4, 4),
      P5 = substr(.data[[peptide_col]], 5, 5),
      P6 = substr(.data[[peptide_col]], 6, 6),
      P7 = substr(.data[[peptide_col]], 7, 7),
      P8 = substr(.data[[peptide_col]], 8, 8),
      P9 = substr(.data[[peptide_col]], 9, 9)
    ) |>
    mutate(!!!c_pos_exprs) |>          # REPLACES the literal C1..C10 block (538–549)
    select(-pep_len, -n_flank_len, -c_flank_len)
}
```

### 3. `08_aa_encoding.R` (lines 5–8)

```r
# BEFORE
paste0("N", 10:1),    # N10, N9, ..., N2, N1   (left -> right toward the cut)
...
paste0("C", 1:10)     # C1..C10

# AFTER (read the flag once at the top of the script)
FLANK_SIZE <- as.integer(Sys.getenv("FLANK_SIZE", "5"))
...
paste0("N", FLANK_SIZE:1),   # N{L} .. N1
...
paste0("C", 1:FLANK_SIZE)    # C1 .. C{L}
```

### 4. `10_assarsson_preparation.R`

```r
# BEFORE (lines 19–21, built near the top of the file)
paste0("N", 10:1), ... paste0("C", 1:10)
# BEFORE (lines 130–131)
N_FLANK_SIZE <- 10
C_FLANK_SIZE <- 10

# AFTER — read the flag at the TOP of the file (before lines 19–21)
FLANK_SIZE <- as.integer(Sys.getenv("FLANK_SIZE", "5"))
if (!FLANK_SIZE %in% c(5L, 10L)) stop("FLANK_SIZE must be 5 or 10")
N_FLANK_SIZE <- FLANK_SIZE
C_FLANK_SIZE <- FLANK_SIZE
# then lines 19–21 become:
paste0("N", FLANK_SIZE:1), ... paste0("C", 1:FLANK_SIZE)
```

Order matters: the column vectors at line 19 execute before the constants
at line 130 — hoist the flag read to the top of the file.

### 5. `run_pipeline.R` — the R gotcha

Each step runs as a **separate `Rscript` process**, so the flag must be
exported into the environment, not kept as a plain variable:

```r
FLANK_SIZE <- as.integer(Sys.getenv("FLANK_SIZE", "5"))
if (!FLANK_SIZE %in% c(5L, 10L)) stop("FLANK_SIZE must be 5 or 10")
Sys.setenv(FLANK_SIZE = FLANK_SIZE)   # <-- child Rscript steps inherit this
cat(sprintf("FLANK_SIZE = %d (%d-mer window)\n",
            FLANK_SIZE, 2 * FLANK_SIZE + 9))
```

All line numbers verified against tree `f9a26b3` (2026-11-13).
