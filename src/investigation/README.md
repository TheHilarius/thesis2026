# src/investigation/ — Script Overview

Exploratory, diagnostic, and one-off analysis scripts supporting the MHC-I antigen-processing pipeline. Not part of the numbered pipeline run order — these investigate data quality, embedding behaviour, and model results around it.

27 scripts (6 R, 21 Python), grouped by function. Numbering is chronological, not semantic — there are two 09s, two 10s, and two 11s covering different topics.

## Run conventions

- **Two Python environments, not interchangeable:**
  - **ESM-C env** — `esm` 3.x package (EvolutionaryScale), Windows anaconda. From WSL: `/mnt/c/Users/olive/anaconda3/python.exe <script>`. See `requirements_esmc_working.txt`.
  - **ESM-IF env** — `fair-esm` 2.0.0, conda `esm_gpu` on the H100 cluster. `No module named 'esm.inverse_folding'` = wrong env. See `requirements_esmif_working.txt`.
  - Inspection/QA scripts (group E) and most analysis scripts need only h5py/pandas/matplotlib/sklearn — run in either env.
- **Config import gotcha:** `02_datasplit_analysis.py`, `05_model_analysis.py`, `06_feature_importance.py` insert only `src/investigation` into `sys.path` yet `from config import ...` (config lives in `src/pipeline/python/`) → run with `PYTHONPATH=src/pipeline/python` or from that dir. `08_pca_variance_analysis.py`, `09_padmode_subset_auc.py`, `10_pad_subset_auc.py` bootstrap correctly; `07`, `09_sequence_consistency_audit`, `10_dynamic_coordinate_translation`, `11_isoform_structure_audit`, `12_indexing_audit`, `13_compare_slot_flat` don't need config.
- **R scripts:** `04`, `09_numeric`, `10`, `11` `source("src/pipeline/r/functions.R")` → launch from repo root; `set_working_directory()` then hard-`setwd`s per username (`olive` → Windows path) and **stops on unknown users** (WSL `olivers` fails). `01`, `02` use plain relative paths, no `source()` — run from repo root.
- Most scripts resolve `data/`, `models/`, `results/` relative to CWD → run from repo root unless noted.

---

## R analysis & figure scripts (6)

### `01_structural_filtering_plots.R`
**Purpose** — Diagnose the structural filtering stage: how many proteins/peptides each filter removed (length cutoff, missing AF PDB, out-of-range), and why hard-to-replace positive peptides were lost.
**Input** — `data/processed/exclusion_ledger.csv` (protein flags: `excluded`, `too_long`, `no_pdb`, `out_of_range`, `has_pdb`, `seq_length`), `sankey_counts.csv` (stage→count, from pipeline `r/04_evaluate_netmhcpan_sensitivity.R`), `df_all_old.csv` (pre-filter snapshot — orphaned, no producer in repo), `df_all.csv` (post-filter). ⚠ None of the 4 exist on disk today.
**Output** — 6 PNGs → `results/figures/filtering/`: `filtering_venn_protein`, `protein_length_alphaFold_comparison`, `filtering_stacked_bar`, `filtering_missing_comparison`, `filtering_venn_peptide_positives`, `filtering_positives_breakdown`.
**How** — `eulerr` Euler diagrams from ledger flag overlaps; ggplot2 log-scale length densities; stacked bar from sankey stages; NA counts in 6 structural columns old vs new; removed positives classified by missing pLDDT/RSA.
**Notes** — Run from repo root (getwd-relative, no `source()`). Outputs exist → run historically. Hardcoded "1,782 positives removed" in one title will silently lie if data changes. `df_all_old.csv` is an orphaned snapshot.

### `02_pca_optimization_analysis.R` — ⚠ superseded
**Purpose** — Sensitivity of joint per-region PCA component counts (peptide/N-flank/C-flank, 5×5×5 = 125 combos) for model AUC, ESM-C vs ESM-IF, RF vs LR.
**Input** — `results/figures/models/pca_optimization/joint_pca_results_handcrafted_sparse_{esmc,esmif}.json` (125 pooled AUCs + per-fold winners) — missing today.
**Output** — Single combined PNG `pca_sensitivity_analysis.png` (same dir): heatmaps, per-fold winner bars, ranked distribution, marginals, one-region sensitivity.
**How** — `jsonlite` parse → ggplot2 + `patchwork` 20×35 in composite.
**Notes** — Superseded by `08_pca_variance_analysis.py` + `09_padmode_subset_auc.py` + `10_pad_subset_auc.py` (slot/pad-mode work). Not re-runnable. Assumes exactly 6 outer folds (hardcoded 0:5).

### `04_data_overview_figures.R`
**Purpose** — Thesis data-overview figures: positives/negatives pipeline Sankeys, NetMHCpan-vs-IEDB Venn, methods schematic.
**Input** — `data/processed/sankey_counts.csv` — missing today (produced by pipeline `r/04_evaluate_netmhcpan_sensitivity.R`).
**Output** — `results/sankey/sankey_{positives,negatives}.html` (+ dep dirs); `results/figures/overview/venn_diagram_overlap.png`, `methods_generation_pipeline.png`.
**How** — `networkD3` 20-node/19-link Sankey with 8 `stopifnot` cascade-arithmetic checks; 5-node negatives Sankey; `eulerr` Venn; hand-built ggplot2 4-box schematic.
**Notes** — `functions.R` setwd gotcha (Windows-R-only as written). `results/figures/overview/` not created by script. Hardcoded "9,995 FASTA sequences" / "Rank < 2% (N = 271,428)" in schematic.

### `09_numeric_feature_analysis.R`
**Purpose** — Univariate association of numeric/binary structural features (RSA, disorder, pLDDT, Q8 point/transition indicators) with presentation; PCA/UMAP views; protein-length and terminus-distance bias diagnostics.
**Input** — `data/processed/df_all_sparse.csv` (file-top constant `dataset_type`: `"base"` → `df_all.csv`, `"sparse"`/`"blosum50"` → suffixed variants; outputs get matching suffix).
**Output** — ~40 files → `results/numeric_9mer_sparse/`: Wilcoxon per-region CSVs+PNGs, binary proportions + Fisher, PCA loadings/scree/scatter, UMAP, logistic ORs (raw + scaled, all + top-30), volcano, per-region forests, correlation heatmaps, ~16 per-feature boxplot/ECDF pairs, terminus-distance diagnostics, quadratic pLDDT GLM; plus `no_pl_no_pc/` filtered rerun excluding `protein_length` and PCs.
**How** — Fisher exact + BH; Wilcoxon (`exact=FALSE`) + SMD 95% CI; `glm(binomial)` via broom; `prcomp(center, scale)` with NA→median imputation; `uwot` UMAP (optional, guarded); `ks.test` + `cliff_delta_sampled` (helper in functions.R).
**Notes** — `functions.R` setwd gotcha; edit file top to switch dataset. Heaviest R script (~1000 lines). The `no_pl_no_pc/` pass isolates results from leakage-prone derived features — deliberate two-view design.

### `10_categorical_residue_analysis.R`
**Purpose** — Per-position amino-acid composition across the full 29-mer context (N10..N1, P1..P9, C1..C10) vs presentation.
**Input** — `data/processed/df_all.csv` (29 positional AA columns).
**Output** — `results/categorical/`: `categorical_feature_tests.csv` (chi-square, BH p, Cramér's V), `residue_enrichment_tables.csv` (log2 enrichment, Haldane–Anscombe OR), 29 grouped-bar + 29 stacked PNGs (`residue_bars/`, `residue_stacked/`), `log2_enrichment_heatmap_all.png`, `odds_ratio_dotplot_all.png`, `differential_logo_29mer.png`.
**How** — Per-position chi-square + Cramér's V (magnitude bands); per position×residue proportions per label, log2 ratio, OR (0.5-corrected); `ggseqlogo` differential logo (PPM₁ − PPM₀).
**Notes** — `residue_bars/` and `residue_stacked/` are never created by the script — must pre-exist. Stale "17 positions" comments (older N4:N1/C1:C4 scope). Output root (`results/categorical/`) differs from script 11's (`results/figures/categorical/`) — easy to confuse.

### `11_structural_preference_logo.R`
**Purpose** — Structural pseudo-logo: Q8 secondary-structure states × 5 regions (N-flank, N→pep boundary, peptide, pep→C boundary, C-flank), enrichment pattern.
**Input** — `data/processed/df_all.csv` (40 `q8point_*`/`q8trans_*` binary columns).
**Output** — `results/figures/categorical/structural_differential_logo.png` (10×5 in @300dpi).
**How** — 8×5 difference matrix = mean(label 1) − mean(label 0) per state×region column; `ggseqlogo(method="custom")` with custom color scheme (helix blues, strand reds, turn/bend greens, coil grey).
**Notes** — No significance tests — descriptive only; pair with `09_numeric_feature_analysis.R` Fisher/logit results. Missing columns only `warning()` → silent zeros possible. `functions.R` setwd gotcha.

---

## Data & structure audits (Python, 5)

### `02_datasplit_analysis.py` — ⚠ import bug
**Purpose** — Post-split data-quality report: fold validity, label balance, feature dtypes, NaN/Inf audit per label and per fold, constant columns. Run after `01_datasplit.py`, before CV.
**Input** — `data/processed/df_all_with_folds.csv` (fold column: 5 CV folds + held-out bucket 5).
**Output** — `logs/02_datasplit_analysis_log_<ts>.txt` only (tee'd stdout).
**How** — Pandas: fold validation vs `config.N_CV_FOLDS`; positional-AA audit vs canonical 20-AA set; NaN/Inf % by label (leakage warning when concentrated in one class); per-fold NaN totals; constant cols; `describe()`.
**Notes** — Broken sys.path bootstrap → needs `PYTHONPATH=src/pipeline/python`. Stale "17 columns" docstring (config has 29). Exit 1 on fold validation failure.

### `09_sequence_consistency_audit.py` — audit chain step 1
**Purpose** — Global-align each protein's canonical UniProt sequence (`df_all` `sequence`) against the AlphaFold PDB SEQRES — detect silent coordinate shifts from isoform canonicalization.
**Input** — `data/processed/df_all.csv` (`uniprot_id`, `sequence`) + `data/processed/structures/alphafold/<uid>.pdb` (~11.8k PDBs).
**Output** — `data/processed/sequence_audit/alignment_audit.csv` (per-protein lengths, `pct_identity`, `status` ∈ exact_match/mismatch/no_af_structure/parse_error), `alignment_summary.txt` (top-20 worst), `results/figures/sequence_consistency/sequence_consistency_audit.png` (identity + length-diff histograms).
**How** — Parse SEQRES (3→1-letter map incl. SEC/PYL/UNK); `Bio.Align.PairwiseAligner` global, BLOSUM62, gap open −10 / extend −0.5; `exact_match` iff 100% identity AND length diff 0; ETA logging every 500 proteins.
**Notes** — Needs biopython. Portable (no config import). `alignment_audit.csv` is the **required input** of `10_dynamic_coordinate_translation.py`. Docstring claims 2 figures but writes 1 combined.

### `10_dynamic_coordinate_translation.py` — audit chain step 2
**Purpose** — For mismatched proteins: build a UniProt→AF2 residue index map from the alignment, translate IEDB peptide coordinates to AF2 coordinates, verify by re-extracting the peptide, partition df_all into cohorts.
**Input** — `df_all.csv` + AF PDBs + `alignment_audit.csv` (from step 1).
**Output** — `data/processed/sequence_audit/{cohort_clean,cohort_rescued,cohort_quarantined}.csv` + `translation_report.txt`.
**How** — Same aligner; walk alignment strings → `{uniprot 1-based pos → af 0-based pos | None}`; translate `[start, end]`; integrity check = extracted AF2 slice == peptide exactly; failures quarantined with reason (`translation_failed` / `integrity_check_failed` / `seqres_parse_error`).
**Notes** — Needs biopython. `build_index_map` depends on Biopython's alignment-string format — version-fragile. Docstring "28 mismatched proteins" is stale narrative; code processes whatever the audit reports.

### `11_isoform_structure_audit.py` — earlier-generation audit
**Purpose** — Validate that every downloaded AF2 structure carries the correct epitope context (flanking windows); investigate (via live web APIs) whether an alternative AF2 model matches the expected isoform length.
**Input** — `data/processed/pos_EL_all_epitopes_hla0201.csv` (9-mers), `data/raw/fasta/combined_full_length.fasta`, AF PDBs, `data/processed/structures/logs/fetch_log.tsv`; web APIs `rest.uniprot.org` + `alphafold.ebi.ac.uk` (cached, rate-limited).
**Output** — `data/processed/sequence_audit/{audit_summary,audit_peptide_details,audit_mismatches}.tsv`, `audit_statistics.txt`, `O43236_report.txt` (SEPT4 case), `uniprot_cache.json`.
**How** — 4 layers: (1) FASTA vs SEQRES length sanity + isoform-suffix regex; (2) primary check — 8-flank context-window equality between UniProt and AF2 sequences at same coords (mirrors R `extract_flanks_safe`); (3) full-sequence compare for failures; (4) API search for an AF2 model whose `uniprotEnd − uniprotStart + 1` matches expected length.
**Notes** — Only network-dependent audit (layers 1–2 offline; cache makes reruns cheap). `FLANK_SIZE = 8` — older convention (later scripts use 10/29). Thematically overlaps `09_sequence_consistency_audit.py` but different inputs + API search. O43236/SEPT4 is the motivating case.

### `12_indexing_audit.py`
**Purpose** — Full-row internal-consistency audit of df_all coordinate/flank/context columns against the `sequence` column — locks the dataset to the `1idx_inclusive` convention.
**Input** — `data/processed/df_all.csv` only.
**Output** — `data/processed/sequence_audit/{indexing_audit_summary.txt, indexing_audit_mismatches.tsv}` (header-only file when clean, artifact always exists).
**How** — 6 checks over ALL rows (`WINDOW_LEN=29`, `FLANK_SIZE=10`, `PEPTIDE_LEN=9`): convention detection (0idx half-open vs 1idx inclusive — `RuntimeError` on drift, message tells you to update the embedding script AND this audit); coordinate validation; raw-flank validation (terminus-aware); padded-flank X-padding to 10; `full_context` 29-mer reconstruction; per-protein pad counts.
**Notes** — Pure pandas, portable, no CLI. Deliberately brittle schema tripwire — hard fail, not a tolerant report. Validates upstream R feature extraction (not structures). Docstring: mirrors indexing logic that used to live in `tools/esm/embed_peptides_esmc.py`.

---

## Post-modelling & PCA/pad-mode analysis (Python, 8)

### `05_model_analysis.py` — ⚠ import bug
**Purpose** — Per-run deep-dive report from `04_modelling.py` cv_results JSONs: publication figures + text summary.
**Input** — `models/cv_results_*.json` (CLI `--results`, globs OK; keys: `fold_predictions`, `held_out_*`, `cv_summary`, `avg_feature_weights`, `config`).
**Output** — Per run → `results/figures/models/<model_key>_<features_key>_<ts>/`: `roc_curve`, `pr_curve`, `confusion_matrix`, `fold_metrics`, `feature_weights` (top-25), `score_distribution`, `calibration` PNGs + `summary_report_<run_tag>.txt`; if >1 run: `model_comparison.png`.
**How** — sklearn `roc_curve`/`auc` (interp mean±std band at 200 points), PR curves with prevalence baseline, threshold-0.5 CM + metrics panel (acc/F1/MCC/sens/spec/PPV/NPV), `calibration_curve` 10 bins, top-25 weight bars; matplotlib Agg @300dpi.
**Notes** — `--results` required. Broken bootstrap (PYTHONPATH workaround). Colors hardcoded per model key (rf green, lr_l2 blue, xgb red). Docstring mentions pickle models — not actually read.

### `06_feature_importance.py` — ⚠ import bug
**Purpose** — Extract/aggregate/plot feature weights from pickled per-fold models; classify features handcrafted / AA-encoding / embedding-PCA; LR direction shown via hatching.
**Input** — `models/<model_key>_<tag>_feature_names.json` + `models/<model_key>_<tag>_model_fold<0..4>.pkl`; `config.POSITION_AA_COLS`.
**Output** — `results/tables/<tag>/feature_importance_<model>_<tag>.csv`; `results/figures/models/<tag>/` top-N barh per model + pairwise comparison PNGs (`--top_compare`, `--top_big`).
**How** — Unpickle per fold; unwrap Pipeline (last step); extract: XGB `booster.get_score(importance_type)` mapped back via `f{i}` indices; `coef_` signed (multiclass → row-norm); else `feature_importances_`. Mean±std across folds; classification by name pattern (`_PC` / `esmc_`/`esmif_` prefixes / positional-AA parts).
**Notes** — CLI: `--features` (tag) required, `--models` default `lr_l2 lr_elasticnet rf`, `--top 30`, `--n_folds 5`, `--xgb_importance gain`. CWD-dependent (`models/` relative). `'PC' in name` heuristic can misclassify handcrafted features. Broken bootstrap. Stale docstring path (script moved from `pipeline/python/`).

### `07_compare_heldout_roc.py`
**Purpose** — Ad hoc cross-run ROC overlay comparison: held-out or pooled CV (out-of-fold) predictions, one combined or per-model PNGs.
**Input** — `models/cv_results_*.json` (CLI `--results` globs, `--model` filter, `--latest-per-setting` dedup).
**Output** — One PNG (`--out`) or per-model PNGs (`--out-prefix`), dpi 300.
**How** — `roc_curve`/`auc` per file, tab10 overlay + chance diagonal; legend: model, feature set, inferred embedding mode (via `component_info`, legacy n_features fallbacks 150/781/931), val/CV AUC + MCC.
**Notes** — Only fully portable py script here (no config import, no bootstrap). Complements `05` (cross-run vs per-run). `--split heldout|cv`.

### `08_pca_variance_analysis.py`
**Purpose** — How many PCA components per toolkit × pad mode for the 29-slot window embeddings; 3 PCA fit modes matching `04_modelling.py`.
**Input** — `data/processed/embeddings/esmc_context_embeddings_{zeropad,impute_boundary,impute_bos_eos,padtoken}.h5` (`window_embeddings` [N,29,1152] + `pad_mask`) and `esm-if_test_{zero,pad,boundary,eos_bos_repeat}.h5` (`window_if_struct` [N,29,512] + `pad_mask`). Existence-checked upfront.
**Output** — `results/figures/models/pca_optimization/<mode>/pca_variance_windows_<key>.{png,csv}` (+`_full` variants for flat-full; per-toolkit combined overlays). CSV: `component_idx, explained_variance_ratio, cumulative_variance`.
**How** — `slot`: IncrementalPCA on real non-pad slots only (2048-row chunks, batch 32768) — matches `04_modelling` per-position PCA. `flat`: full-memory `PCA(svd_solver="randomized", random_state=42)` on fully-real rows flattened to 29·D, capped `--flat-n-components 1024`. `flat-full`: exact full spectrum via streamed Gram matrix `XᵀX` + center correction (~18 GB peak ESM-C / ~4 GB ESM-IF). Cross-mode validation (flat-full): zero/boundary/eos_bos_repeat spectra must match within 1e-6 per toolkit. Threshold table at 50/80/85/90/95/99%.
**Notes** — CLI: `--pca-mode slot|flat|flat-full`, `--force`. CSV caching with validity checks. matplotlib optional (CSVs still written without it). Outputs consumed by `13_compare_slot_flat.py`.

### `09_padmode_subset_auc.py`
**Purpose** — Pad-mode comparison on the FLAT modelling grid: overall AUC cannot separate pad modes (real-slot signal identical) → recompute AUC on only the informative padded rows (~1443 terminus peptides per toolkit), XGBoost only.
**Input** — `models/flat/cv_results_xgb_*_flat_*.json` (`outer_val_predictions` per fold 0..5), `data/processed/df_all_with_folds.csv` (`label`, `fold`), `pad_mask` + `row_indices` from raw H5s (one per toolkit; masks identical across modes within toolkit).
**Output** — `results/padmode_subset_auc.csv` (per toolkit × padtype × learner × pca: `auc_padded`, `n_padded`, `auc_nonpadded`, `auc_overall`, fold std) + stdout pivot tables.
**How** — "Informative" = ≥1 pad AND ≥1 real slot (excludes all-pad notfound rows); scatter H5 row order → df order via `row_indices`; scatter fold predictions back onto full df (length + label-consistency checks, every row covered once); `roc_auc_score` on subsets.
**Notes** — No CLI; hardcoded `models/flat` (prints server rsync hint if missing). Filename regex supports lr_l2/rf but reads xgb only. Overlaps `10_pad_subset_auc.py` — see relationship map.

### `10_pad_subset_auc.py`
**Purpose** — Same question on the SLOT grid (default) + pairwise probability deltas between modes.
**Input** — `models/slot/cv_results_xgb_handcrafted_sparse_esmc_win_*_pca9_*.json` (`--json-dir`, `--pattern`), `data/processed/embeddings_prepared/esmc_win_zeropad_prepared.h5` (`--pad-h5`: `pad_mask`, `folds`, `labels`, `pad_counts`), `df_all_with_folds.csv`.
**Output** — `results/tables/pad_subset_auc_<tag>/{<tag>.csv, _pivot.csv, _pairwise.csv, _rows.csv}` (`--save-rows`: per-row probabilities per mode + metadata).
**How** — `pad_row = pad_mask.any(axis=1)` (any-pad — differs from 09's informative mask); reconstruct out-of-fold probability vectors via fold scatter; pad/real/global ROC-AUC + pad PR-AUC; pairwise |Δprob| stats between modes; label checks vs H5.
**Notes** — CWD-dependent output. Generic `--json-dir`/`--pattern` could subsume `09_padmode`'s grid, but `09_padmode` stays purpose-built (flat grid, both toolkits) — both kept.

### `13_compare_slot_flat.py`
**Purpose** — Slot-PCA vs flat-PCA head-to-head anchored on ~50% cumulative explained variance (component counts differ between modes, EV makes them comparable).
**Input** — `server_results/**/cv_results_*.json` (`--root server_results`); EV curves from `08`: `server_results/results/figures/models/pca_optimization/slot/*.csv` + `runs/padfull_flat_20261001_1645/plots/pca_variance/*_full.csv` (run folder hardcoded).
**Output** — `results/tables/slot_vs_flat_{results,leaderboard,deltas_50ev}.tsv` + `results/figures/slot_vs_flat/{slot_vs_flat_auc, slot_vs_flat_curve, slot_vs_flat_padding}.png`.
**How** — 8 feature sets × 3 models × 2 PCA modes × 3 PCA sweep values; map pca count → EV% off the 08 variance curve; dedup latest timestamp per config; flat−slot deltas at the ~50% EV anchor; coverage-gap report (expects 12 results per cell).
**Notes** — Slot 85/95% EV points exceed the flat EV ceiling (~72%) → only the ~50% anchor is shared. Depends on 08 outputs + the modelling grid. Scope-locked PCA sets (`SLOT_PCA`/`FLAT_PCA` constants filter the sweep).

### `plot_padmode_data_compare.py`
**Purpose** — 11-bar held-out AUC-ROC figure for the data-comparison grid: sparse one-hot AA, BLOSUM50, BLOSUM50+engineered, 4 ESM-C pad modes, 4 ESM-IF pad modes.
**Input** — `--tsv`: aggregate TSV from `extract_model_metrics.py` (e.g. `runs/<...>/results/model_matrix_*.tsv`; needs `model_key`, `feature_set`, metric column).
**Output** — `results/figures/investigation/padmode_data_compare_auc_roc.{png (dpi 200), csv}`.
**How** — Filter `--model` (default xgb); fixed `BAR_ORDER` of 11 feature keys; group colors (csv grey, esmc red, esmif blue); value labels; dashed separators after bars 3 and 7; title "embeddings @ 90% EV, held-out AUC-ROC".
**Notes** — `--metric heldout_auc_roc` default. CWD-relative out dir. Missing feature sets warned, not fatal. Pure matplotlib — either env.

---

## ESM pad-mode validation probes (Python, 4)

All four print to stdout only — no files written. These are the experiments that justified the pad-mode design in `tools/esm/embed_peptides_esmc.py` and `tools/esm/embed_peptides_esmif.py`.

### `embed_test_esmc.py`
**Purpose** — Minimal probe of the ESM-C tokenizer and pad-token embedding behaviour.
**Input** — None from disk (hardcoded 9-mer `BASE = "ACDEFGHIK"`); weights `ESMC.from_pretrained("esmc_600m")` auto-download.
**Output** — stdout only.
**How** — Decode vocab ids 0–32; encode literals `X - . | <mask> <pad> <cls> <eos> <unk>` + space; embed via `model.encode` + `logits(return_embeddings=True)` under `no_grad`; compare pad-slot vs real-residue L2 norms for pads `[X, -, <pad>, <mask>, <unk>]`; expected-id sanity checks.
**Notes** — ESM-C env (`/mnt/c/Users/olive/anaconda3/python.exe src/investigation/embed_test_esmc.py` from WSL). No CLI. Key finding encoded in the script: pad norms are ~real-residue scale → a downstream CNN cannot detect "empty" by magnitude alone → explicit `is_pad` mask channel needed. Predecessor of `embed_test_esmc_pad.py`.

### `embed_test_esmc_pad.py`
**Purpose** — Scaled ESM-C padding test with explicit attention masking on ~100 real proteins.
**Input** — `data/processed/df_all.csv` (`sequence` column); ESMC-600M.
**Output** — stdout (summary table + interpretation).
**How** — Filter to canonical-AA sequences len ≥ 40; sample `--n` (default 100, rng seed 42); per protein keep a fixed 19-residue real core (10 N-flank + 9 peptide), replace the next `--k` (default 10) tail residues with a pad char from `[<pad>, X, -, <mask>]`; 3 forwards per pad char (auto / explicit bool `sequence_id` mask / fully unmasked) under `no_grad` + bfloat16 autocast; metrics: `real_cos_auto_vs_mask` (≈1 → model already auto-masks), `real_cos_full_vs_mask` (leak), `pad_norm`, `pad_real_cos`.
**Notes** — CLI `--n`, `--k`. ESM-C env; cwd = repo root. Mask semantics from `esm/layers/attention.py` (`mask = seq_id_i == seq_id_j`) — real↔pad blocked but pad↔pad still attends (issue #299 quirk). Results fed the design of `tools/esm/embed_peptides_esmc.py` → the 4 `esmc_context_embeddings_*.h5`.

### `embed_test_esmif.py`
**Purpose** — Minimal ESM-IF probe: are NaN-coordinate "gap" residues (ESM-IF's in-distribution "no structure here" signal from span-masking training) finite and sane as a pad representation?
**Input** — `--pdb` optional (default: smallest file among `data/processed/structures/alphafold/*.pdb`); ESM-IF1 (`esm_if1_gvp4_t16_142M_UR50`).
**Output** — stdout (per-probe checks + verdict section).
**How** — `load_coords` (chain A) → `[L, 4, 3]`; build 10 all-NaN rows; append (C-flank) and prepend (N-flank); check finiteness of the full representation + gap-slot norms vs real-residue median norm.
**Notes** — fair-esm env (conda `esm_gpu`, H100 cluster). cwd = repo root. Verdict logic: both finite → NaN-coord gaps usable as the ESM-IF pad representation (mirrors AntiFold's gap tokens); NaN propagation → fall back to post-model zero-vector padding. Predecessor of `embed_test_esmif_pad.py`.

### `embed_test_esmif_pad.py`
**Purpose** — Ports `embed_test_esmc_pad.py` metrics to ESM-IF (pad = NaN-coord gap residue); validates the exact `pad_rep`/`eos_rep` constants that `tools/esm/embed_peptides_esmif.py` uses for its fill modes (esp. eos-repeat) before the full embedding run.
**Input** — `--pdb-dir data/processed/structures/alphafold` (samples the `--n` default 20 smallest structures); ESM-IF1.
**Output** — stdout (per-structure progress + summary + interpretation).
**How** — `CoordBatchConverter.forward`; representation offset check (`rep_len − L == 2`, BOS/EOS); baseline `rep_nogap`; append `--k` (default 10) NaN rows; gap-masked vs forced-unmasked reps; metrics: finite gap embeddings, `real_cos_gapmask_vs_nogap` (inertness), `real_cos_unmasked_vs_gapmask` (leak), pad/real norms + cosines, `eos_norm`/`eos_real_cos` (the eos-repeat constant).
**Notes** — fair-esm env; CLI `--n`, `--k`, `--pdb-dir`; cwd = repo root. Key conclusions: (a) gapmask ≈ nogap → safe to embed the full chain and post-fill pads; (b) never override the auto `padding_mask` (forces NaN into real positions); (c) non-finite gap embeddings → fall back to zero vectors.

---

## Embedding inspection & QA (Python, 4)

### `inspect_window_embeddings.py` — current generation
**Purpose** — Inspect and cross-compare the 8 window-mode pad-mode H5s: do the padding modes differ only in padded slots (real slots identical across modes)?
**Input** — One or more window H5s (canonical: the 8 files in `data/processed/embeddings/` + optional legacy files) + `csv_path` (`df_all.csv`, `label` column only).
**Output** — `--out_dir` (default `results/embedding_inspection/windows*`): `pairwise_mode_diffs.csv`, `comparison_table.csv`, `slot_norm_curve.png`, `pairwise_diff_hist.png`, `status_breakdown.png`, `pca_windows.png` + stdout.
**How** — Per-file probe streamed in `--chunk` (2048) row chunks — never loads a full ~2.6 GB array; schema detection (`window29` / `legacy_esmc` / `legacy_esmif` / `context2d`); attrs dump; `pad_group` from the `pad_mode` attr; `status` counts (0=primary, 1=af2, 2=missing, 3=notfound); per-slot mean norms + coverage; pad vs real slot median norms. Cross-mode (≥2 window29 files): row alignment via `row_indices`, `pad_mask` identity check, **max abs diff on REAL slots** (≈0 ⇒ modes differ only in padding), pad-constancy checks (zero-mode exact 0; padtoken per-dim std ≈ 0), pairwise per-row Frobenius diffs, cosine of mean-pooled real slots, PCA-lite (≤`--pca_n` 10k rows).
**Notes** — CLI: positional `h5_paths...` + `csv_path`, `--out_dir`, `--chunk 2048`, `--pca_n 10000`, `--seed 42`. Any subset of files OK. Either env, no GPU. Slot bands: 0–9 N-flank, 10–18 peptide, 19–28 C-flank.

### `inspect_embeddings.py` — legacy generation
**Purpose** — Summarize + visualize single-context or 3-region (peptide/N-flank/C-flank) ESM embedding H5s, positives vs negatives.
**Input** — `h5_path` (legacy keys `context_emb`/`context_if_struct` or `peptide_emb`/`n_flank_emb`/`c_flank_emb` + `*_if_struct`) + `csv_path`. Docstring example H5s no longer on disk.
**Output** — `--out_dir` (default `results/embedding_inspection/`): 8 PNGs (L2 norm distributions, per-dim mean diff, cross-region cosine (3-region only), inter-sample cosine, PCA, t-SNE, top discriminative dims, norm vs length) + `embedding_summary.csv` + per-protein zero-embedding coverage stdout.
**How** — Format auto-detect; row alignment via `row_indices`, else 1:1, else key join on peptide/uniprot; Mann-Whitney on norms and per-dim (Bonferroni-corrected, rank-biserial effect sizes); 2000 random pair inter-sample cosines; PCA ≤50 comps; t-SNE ≤`--max_tsne` 5000.
**Notes** — Either env, no GPU. For the current 29-slot window files use `inspect_window_embeddings.py` instead.

### `report_structure_coverage.py` — legacy generation
**Purpose** — Quantify structure coverage (real vs missing structure) for an ESM-IF embedding run by joining dataset CSV, embedding H5, and a known-missing list.
**Input** — `--df` (e.g. `df_all.csv`, needs `uniprot_id`), `--h5` (`uniprot_ids` + `context_if_struct`/`context_emb`), `--missing` optional TSV (no header, first column = uniprot_id; treated as empty if path doesn't exist).
**Output** — `--out` dir: `structure_coverage_report.json` (protein-level, peptide-level, protein-peptide-level) + `protein_level_missingness.csv` + stdout.
**How** — "Missing structure" = all-zero embedding row; protein-level set intersection/coverage rate; peptide-level zero-row rate; per-protein mean `ctx_missing_rate` → all/any-missing counts.
**Notes** — Assumes H5 rows positionally aligned with df rows (only valid if H5 written in df order). Legacy schema — for window files, equivalent coverage comes from the `status` dataset via `inspect_window_embeddings.py`. Either env.

### `summarize_embeddings.py` — legacy generation
**Purpose** — Quick zero-vector-rate comparison across one or more single-context embedding H5s.
**Input** — One or more H5 paths as plain positional args (needs 2D `context_if_struct` or `context_emb` keys); optional `uniprot_ids`, attrs `device`/`cache_size`.
**Output** — stdout only: per-file detail + comparison table (Rows / Embedded / Zero% / Proteins / MeanNorm).
**How** — Reads the full 2D context array; zero = L2 norm 0; unique-protein counts from `uniprot_ids`; per-file errors caught, never abort the batch.
**Notes** — No argparse (plain argv, sorted). Loads the full array — don't point it at multi-GB window tensors (would `KeyError` anyway, 2D keys only). Either env. Lightest sibling of `inspect_embeddings.py` / `report_structure_coverage.py`.

---

## Script relationships

```
Pad-probe chains (validated design → production writers):
  embed_test_esmc.py → embed_test_esmc_pad.py
      → tools/esm/embed_peptides_esmc.py → data/processed/embeddings/esmc_context_embeddings_{zeropad,padtoken,impute_bos_eos,impute_boundary}.h5
  embed_test_esmif.py → embed_test_esmif_pad.py
      → tools/esm/embed_peptides_esmif.py → data/processed/embeddings/esm-if_test_{zero,pad,boundary,eos_bos_repeat}.h5

Inspection:
  8 window H5s → inspect_window_embeddings.py (real-slot identity across pad modes)
  legacy H5s (no longer on disk) → inspect_embeddings.py / report_structure_coverage.py / summarize_embeddings.py

Modelling/analysis chain:
  window H5s → pipeline 03_prepare_embeddings.py → embeddings_prepared/ → 04_modelling.py (slot|flat grids)
      → models/{slot,flat}/cv_results_*.json
      → 05_model_analysis.py (per-run deep dive) / 07_compare_heldout_roc.py (cross-run overlay) / 06_feature_importance.py (from pickles)
      → 08_pca_variance_analysis.py (EV curves) → 13_compare_slot_flat.py (slot vs flat @ ~50% EV)
      → 09_padmode_subset_auc.py (flat grid, pad modes) / 10_pad_subset_auc.py (slot grid, pad modes)
      → extract_model_metrics.py TSV → plot_padmode_data_compare.py (11-bar figure)

Structure/coordinate audit chain:
  11_isoform_structure_audit.py (earlier-gen, API-based) — standalone
  09_sequence_consistency_audit.py (step 1: alignment_audit.csv)
      → 10_dynamic_coordinate_translation.py (step 2: clean/rescued/quarantined cohorts)
  12_indexing_audit.py — standalone df-internal tripwire (1idx_inclusive convention)

R analyses:
  01_structural_filtering_plots.R + 04_data_overview_figures.R — depend on pipeline r/04 artifacts
  09_numeric + 10_categorical + 11_structural logo — feature-association analyses on df_all*.csv
```

`09_padmode_subset_auc.py` vs `10_pad_subset_auc.py` — same scientific question, different grids: 09 = flat grid, both toolkits, XGB, informative mask (≥1 pad AND ≥1 real) from raw H5 + `row_indices`; 10 = slot grid (default), ESM-C prepared H5, any-pad mask, pairwise Δprob + per-row export. 10's generic CLI could cover 09's grid, but 09 stays purpose-built; both kept.

## Cross-cutting gotchas

1. **Broken config import** — `02_datasplit_analysis.py`, `05_model_analysis.py`, `06_feature_importance.py` need `PYTHONPATH=src/pipeline/python` (their bootstrap inserts only `src/investigation`). Not fixed yet; documented workaround.
2. **R setwd double-bind** — `04`, `09_numeric`, `10`, `11` need repo-root launch for `source(functions.R)`, then `set_working_directory()` hard-setwds per username and stops for unknown users (WSL `olivers` fails; `olive` branch is a Windows path).
3. **Not re-runnable today (inputs missing)** — `01_structural_filtering_plots.R` (4 CSVs gone), `02_pca_optimization_analysis.R` (JSONs gone; also superseded), `04_data_overview_figures.R` (needs pipeline r/04 rerun for `sankey_counts.csv`).
4. **Legacy vs current H5 generation** — `inspect_embeddings.py`, `report_structure_coverage.py`, `summarize_embeddings.py` target single-context/3-region H5s no longer on disk; current generation is `inspect_window_embeddings.py` + the 8 window H5s.
5. **Numbering collisions** — two 09s, two 10s, two 11s (different topics, chronological accretion).
6. **Uncreated output dirs** — several scripts `ggsave`/`png()` into dirs they never `dir.create` (e.g. `results/categorical/residue_bars/`, `results/figures/overview/`) — must pre-exist.

## Current runnability

| Script | Input present? | Runnable as-is? |
|---|---|---|
| `01_structural_filtering_plots.R` | No (4 CSVs missing) | No |
| `02_pca_optimization_analysis.R` | No (JSONs missing) | No — superseded |
| `04_data_overview_figures.R` | No (`sankey_counts.csv`) | No — needs pipeline r/04 rerun |
| `09_numeric_feature_analysis.R` | Yes | Yes (user `olive`, Windows R, repo root) |
| `10_categorical_residue_analysis.R` | Yes | Yes (same caveat; 2 subdirs must pre-exist) |
| `11_structural_preference_logo.R` | Yes | Yes (same caveat) |
| `02_datasplit_analysis.py` | Yes | Yes, with PYTHONPATH workaround |
| `05_model_analysis.py` | Depends on models/ JSONs | Yes, with PYTHONPATH workaround |
| `06_feature_importance.py` | Depends on models/ pickles | Yes, with PYTHONPATH workaround |
| `07_compare_heldout_roc.py` | Depends on models/ JSONs | Yes — fully portable |
| `08_pca_variance_analysis.py` | Yes (8 window H5s) | Yes |
| `09_padmode_subset_auc.py` | Needs `models/flat/` JSONs | Yes if dir present |
| `10_pad_subset_auc.py` | Needs `models/slot/` JSONs | Yes |
| `13_compare_slot_flat.py` | Needs `server_results/` | Yes if present |
| `plot_padmode_data_compare.py` | Needs run TSV | Yes |
| `embed_test_esmc*.py` | Yes (df_all.csv / none) | Yes — ESM-C env |
| `embed_test_esmif*.py` | Yes (PDBs) | Yes — fair-esm env |
| `inspect_embeddings.py`, `report_structure_coverage.py`, `summarize_embeddings.py` | Legacy H5s gone | Only against legacy files |
| `inspect_window_embeddings.py` | Yes (8 window H5s) | Yes |
| `09/10_dynamic`, `11_isoform`, `12_indexing` audits | Yes (df_all + PDBs) | Yes (11 partially networked) |
