# Task 3 (ours, Python): allele readiness + flank-flip coordination checklist

Companion docs: `docs/r_pipeline_dedup_plan.md` (Task 1, Oliver),
`docs/r_pipeline_flank_flag_plan.md` (Task 2, Oliver).

## Part A — allele support in the Python pipeline

**IMPLEMENTED 2026-11-23** in `src/pipeline/python/config.py` (plus
`03_prepare_embeddings.py` dummy-fold support and
`14_predict_external.py`). Differences from the original spec below:

- `hla_a0101` forces `THESIS_VARIANT` OFF (the external allele is its own
  dataset dimension — fork-era decision, kept), instead of composing
  allele prefix + variant suffix.
- `SPLIT_DATA_PATH` **aliases** `RAW_DATA_PATH` for `hla_a0101` (the
  external set has no CV folds; 03 adds a dummy fold column).
- Log names in 03 carry a `hla_a0101_` prefix; figures go to
  `results/figures/models_hla_a0101/`.
- Unset `THESIS_ALLELE` verified byte-identical to HEAD config on every
  path (path/embedding/component parity check, 2026-11-23).
- Run recipe: `docs/hla_a0101_run_recipe.md`.

Original spec (kept for reference):

- `config.py`: new env `THESIS_ALLELE` (unset = `hla0201`, legacy flat
  paths, byte-identical behavior — same pattern as `THESIS_VARIANT`):

  ```python
  ALLELE = os.environ.get("THESIS_ALLELE") or "hla0201"
  def _allele(path: Path) -> Path:
      if ALLELE == "hla0201":
          return path                      # legacy flat: data/processed/...
      # 0101: data/processed/hla_a0101/<file>
      return path.parent / "hla_a0101" / path.name
  ```

- Composition order with variants: **allele prefix first, then variant
  suffix** — `THESIS_ALLELE=hla_a0101 THESIS_VARIANT=remove_dup_windows`
  resolves to `data/processed/hla_a0101/df_all_with_folds_remove_dup_windows.csv`.
  Apply `_allele()` at the `DATA_DIR`/raw-split definitions, not per-path.
- Prepared/raw embedding H5 dirs follow the same prefix (03 writes under
  `data/processed/hla_a0101/embeddings_prepared/` once R+03 run for 0101).
- Nothing else in modelling is allele-specific: encodings, folds logic,
  `01_datasplit` (width inference), and the robust split are all
  allele-agnostic. netMHCpan/anchor-map logic stays R-side.
- Acceptance: with `THESIS_ALLELE` unset, every config path is
  byte-identical to today (same guard pattern as the datasplit regression).

## Part B — Python checklist for the FLANK_SIZE default flip (Task 2 companion)

Execute in ONE coordinated change when Oliver's flag merges with default 5:

1. `config.py` — `"window": 29` per embedding source; `POSITION_AA_COLS`
   (`N10..N1` / `C1..C10`); `HAMMING_CUTOFF` untouched.
2. `04_modelling.py` — **`n_pad`/`c_pad` hardcoded `10` (lines ~1786–1793)**
   → derive from the data's window width (already known gap: wrong on
   flank5 data whenever the `handcrafted` component is selected).
3. `reduce_29mer_to_19mer.py` + `THESIS_VARIANT=flank5` machinery — retire
   or keep as historical reference (flank5 becomes native).
4. Audits with hardcoded 29 (`12_indexing_audit` `WINDOW_LEN=29`, etc.).
5. Already done / no change: `01_datasplit.py` (infers 19/29 from data,
   X-safe alphabet), variant H5 subsetting (`prepare_variant_h5.py`).

Acceptance for the flip: one modelling run on native flank5 data must
reproduce the existing `THESIS_VARIANT=flank5` run results (same rows,
same features -> same folds -> same metrics).

Spec'd 2026-11-13 against tree `f9a26b3`.
