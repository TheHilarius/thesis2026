# Thesis Overnight — 2026-11-04

## Summary

origin/hilarius did not move since the previous report: the reviewed range 76735ad..origin/hilarius is empty (0 commits) and the tip is still 76735ad (2026-11-02), so ## Changes has nothing to add. Every finding from the 2026-11-03 report was re-verified today directly against the origin tree and all remain open — the stale third ESM-IF result table, the eosrepeat purge tail, the shared-helper duplication, the untracked redo outputs, and the README gaps. The only thing that changed is the calendar: the 7-day window rolled forward, dropping 2026-10-28's six commits. No tracked result numbers moved.

## Past 7 Days

Calendar window 2026-10-29..2026-11-04 contains 8 commits. Commits per day:

      2 2026-10-29
      1 2026-10-30
      5 2026-11-02

(10-31, 11-01, 11-03 and today 11-04 have none; the most recent day with activity is 2026-11-02.) Theme-level view — this section intentionally overlaps earlier reports, so per-file detail stays under ## Changes only:

- **ESM-IF pad-mode consolidation (2026-11-02)** — all ESM-IF investigation arms rerouted from eosrepeat to eos/bos-repeat, the eosrepeat configuration purged, the EV ladder and top-variance sweep tables/figures recomputed under the new key, a figure-label style pass, and a new dimselect-vs-variance/PCA/BLOSUM50 benchmark script.
- **Dimensional-reduction alternatives** — the `--vardims` top-variance flag beside PCA, fold-nested variance selection with EV-vs-PCA and variance-optimum figure scripts plus DeLong testing, and the EV-ladder dimension tables with sweep figures/results.
- **Sweep bookkeeping** — investigation-script updates for the 20/24/pca-sweep comparisons and removal of the tracked `dimselect_sweep` result files (moved to `_legacy`).
- **CNN baseline arm** — pad mask removed for simplicity, a 2-layer slurm runner, and analysis of the 2-layer CNN models.
- **Window-embedding pooling figures** — a pooling plot update (2026-10-29).
- **Dataset-variant construction** — dropping the 715 negatives from proteins with no positive peptides (2026-10-30).

## Changes

Reviewed range 76735ad..origin/hilarius = **0 commits**. origin/hilarius is still 76735ad (2026-11-02 13:39:56 +0100), identical to the previous report's anchor; `git fetch --dry-run` reports nothing pending. Nothing to report here, and nothing from earlier reports needs repeating.

## Experiments / Results

Repository-supported results only; re-read from the origin tree today. Still 0 tracked `cv_results_*.json` and 0 tracked files under `data/`, so every number below comes from tracked tables.

- **ESM-IF (eos/bos-repeat) variance ladder vs the raw ceiling** — `runs/ev_matched_topXvariance_sweep/results/variance_sweep_vs_raw_delong.tsv:12-21`: best rung ev85 = 0.648705 (392 dims) vs raw 0.649814, delta −0.001109, p_holm 6.311e-01, `ns` (`:20`); ev65 = 0.644742, p_holm 6.948e-04, `***` (`:16`); ev70 = 0.647190, p_holm 1.543e-01, `ns` (`:17`, first ns rung). Note ev65 is `***` against raw while ev70 above it is `ns` — the ladder is not monotone in significance.
- **ESM-IF vs PCA (same directory)** — `ev_variance_vs_pca_delong.tsv:12-21`: ev50/ev60/ev65/ev70 are `ns` against their PCA-k match (ev65 delta −0.000332, p_holm 1.000e+00), ev75/ev80 `**`, ev85/ev90 `***` (ev90 delta +0.015829, p_holm 9.347e-17).
- **ESM-C** — vs raw: ev90 = 0.704873 vs raw 0.703964, p_holm 6.227e-01, `ns` and marked `is_best` (`variance_sweep_vs_raw_delong.tsv:11`); first ns rung is ev80 (`:9`). vs PCA: ev80 vs k=2278 delta +0.004466, p_holm 3.833e-03, `**` (`ev_variance_vs_pca_delong.tsv:9`).
- **The configured default rung sits below the raw ceiling for ESM-IF** — `src/pipeline/python/config.py:64-66` keeps `"esmif_eos_bos_repeat": 65` with the comment "ev65 ***, ev70 = smallest ns rung — kept at the 260-dim budget on purpose"; observation of the tracked table, not an interpretation.
- **Two tracked tables in one directory still disagree** — the third table, `variance_sweep_vs_best_delong.tsv:12-21`, still reports `esmif_eosrepeat` with best ev85 = 0.648827 (vs best-vs-raw 0.648827/0.649814 at `:21`), versus 0.648705 in the recomputed raw table. See ## Code Review 2.
- **Tracked figures** — both esmif PNGs under `runs/ev_matched_topXvariance_sweep/plots/` were regenerated in the consolidation; not regenerated: `runs/esmif_eos_sweep/plots/pca_sweep_auc_full.png` and `pca_sweep_auc_line.png` (last changed 2026-10-23) even though `pca_sweep_AUC_delong_comparison.py` changed 2026-11-02, and the new `pca_sweep_auc_full_withfull_baselines.png` has no tracked copy. `runs/dim21_eosbos/` and `runs/varfold_pilot_eosbos/` have 0 tracked files.

## Code Review — Shortcomings & Issues

No new commits in this range, so all findings are carried from the 2026-11-03 report and were re-verified today against origin/hilarius (76735ad) with fresh commands; ordered by redundancy/reinvention first.

1. **Helper duplication remains the top redundancy finding (carried)** — `src/investigation/24_dimselect_delong.py:158` defines `star()` byte-identical to `21_dim21_delong.py:83`, `:109` re-implements `load_oof` while `26_ev_variance_vs_pca.py` and `27_variance_optimum.py` do `load_oof, star = _d21.load_oof, _d21.star`, and `:51` adds an `importlib.util.spec_from_file_location` load. Re-counted today: `def star` in 10 files, `def delong_test` in 6, `def holm_adjust` in 5, `spec_from_file_location` in 8 files (7 investigation + `src/pipeline/python/05_cnn_modelling.py`). The script does reuse `_p15.delong_test`/`holm_adjust` — the copy-paste stopped one helper short. One shared ~40-line module (`delong_stats.py`) removes all of it; severity: maintainability (over-engineering by duplication). [fd3a515]
2. **Two tracked result tables in one directory disagree about the ESM-IF arms (carried, re-verified)** — `variance_sweep_vs_raw_delong.tsv:12-21` and `ev_variance_vs_pca_delong.tsv:12-21` were recomputed for `esmif_eos_bos_repeat`, but `variance_sweep_vs_best_delong.tsv:12-21` still reports `esmif_eosrepeat` (best ev85 = 0.648827; `:21` is the only tracked `best_vs_raw` row for ESM-IF and describes a key `config.py` no longer defines). "Best ESM-IF rung" from this directory reads 0.648827 or 0.648705 depending on the file. Severity: risk (results consistency). [cbf243c changed two of three tables]
3. **The "eos_bos EV ladder" is content-identical to the eosrepeat ladder (carried, question for the authors)** — every one of the 12 renames in 216c2f8 differs from its source by exactly one line (`embedding_key`); the dim lists, `n_dims` (78/122/174/230/260/291/323/357/392/429) and `top21` are unchanged, and the eosrepeat base ranking CSV was deleted with no `dim_variance_esmif_eos_bos_repeat.csv` added (ESM-C keeps its base CSV). The variance arms were recomputed and do differ (0.640985 vs 0.638911 at ev30), so the ladder that selected their dims either reran to an identical result or was relabelled — git cannot tell which. Severity: risk (results provenance). [216c2f8]
4. **A job-tag collision is hardcoded in four files (carried)** — both `dim21_eosbos` jobs wrote filenames containing `simple21_var21`, so the first-21 arm is distinguished only by timestamp (`20_dim21_barplot.py:85-86`, `24_varfold_delong.py:62-66`). The same literals appear at `20_dim21_barplot.py:87-94`, `21_dim21_delong.py:53-54`, `24_varfold_delong.py:64-66`, `25_varfold_barplot.py:43-45`; any rerun needs four coordinated edits, and a glob over `*var21*` matches both arms. Severity: risk (arm identification). [86b26bd]
5. **The purge stopped at configuration (carried, re-verified)** — `src/investigation/28_cnn_compare_barplot.py:94-95` docstring asserts the feature_set *is* `emb_esmif_eosrepeat`, `:105` still accepts it as fallback, `:238` lists it in `ESMIF_KEYS`; `runs/ev_matched_topXvariance_sweep/README.md:13` still lists `emb_esmif_eosrepeat`; `runs/varfold_pilot/` still tracks the eosrepeat varfold record. Severity: maintainability (mixed-arm fallback + stale docs). [fd3a515]
6. **Y-axis relabel is partial across one figure family (carried)** — 76735ad renamed to "CV-AUC" at `27_variance_optimum.py:215` (and `pca_sweep_AUC_delong_comparison.py:480,523`), while `25_varfold_barplot.py:140` and `26_ev_variance_vs_pca.py:226` still emit `"AUC-ROC (nested CV, mean +/- std)"` (re-grepped today), and `20_dim21_barplot.py:205` / `28_cnn_compare_barplot.py:165,275` do too. Two label conventions will sit side by side in the thesis. Severity: style. [76735ad]
7. **Tracked figures lag the script that draws them (carried, re-verified)** — `runs/esmif_eos_sweep/plots/pca_sweep_auc_full.png` and `pca_sweep_auc_line.png` last changed 2026-10-23 while `pca_sweep_AUC_delong_comparison.py` changed 2026-11-02; the new `pca_sweep_auc_full_withfull_baselines.png` (`pca_sweep_AUC_delong_comparison.py:767`) has 0 tracked copies; `runs/dim21_eosbos/` and `runs/varfold_pilot_eosbos/` have 0 tracked files while the eosrepeat predecessor (`runs/varfold_pilot/plots/varfold_vs_refs_auc.{png,csv}`, `results/varfold_delong_pairs.tsv`) is tracked. Severity: risk (which figures are citable). [76735ad, 86b26bd]
8. **`n_a`/`n_b` columns mean different things in different tracked TSVs (carried)** — feature counts in `variance_sweep_vs_best_delong.tsv` (793 / 27840), dim count + PCA-k in `ev_variance_vs_pca_delong.tsv:11` (793 / 5883), explicit `var_n_dims`/`n_features` headers in `variance_sweep_vs_raw_delong.tsv`. Severity: style (misread risk when quoting a row). [carried tables]
9. **Carried leftovers, re-verified unchanged** — dead `load_delong_pick` with no caller in its own file (`src/investigation/22_pool_barplot.py:100`; the copy at `20_dim21_barplot.py:107` *is* used at `:160-162`); dead `esmif_` branch in the arm-redirect loop (`23_pool_delong.py:97`); `src/investigation/README.md` still references the nonexistent `10_pad_subset_auc.py` 7 times and indexes none of `24_dimselect_delong.py`–`28_cnn_compare_barplot.py` (grep for `24_`–`28_` returns nothing); zero-byte tracked logs, the `.gitignore` `runs/*/models/` triplicate and `paired_tests_OLD.csv` all still present. Severity: maintainability/style.
10. **What passes the ladder (carried)** — fd3a515 was a net deletion with no new abstraction; `26_ev_variance_vs_pca.py:71-80` replaced a per-key if/else path build with one glob template; `24_dimselect_delong.py` reuses `_p15`'s DeLong/Holm instead of adding a 7th copy; the `ref_bars` flag in 76735ad is display-only and keeps existing CSVs/verdicts byte-identical, but it is the 4th boolean-ish variant of the same bar figure (`special_k`, `refs_mode`, `pending`, `ref_bars`).

## Potential Issues

1. **Which construction is authoritative? (carried, blocking for citation)** — the R re-sampled variant writes `data/processed/df_combined_pos_and_neg_remove715negonly.csv` (`04_evaluate_netmhcpan_sensitivity.R:805-807`) that nothing consumes, while `run_pipeline.R:37` / `05_protein_features.R:13` hardcode the base CSV. Authors' call: is `make_variant_dataset.py` the definition of record?
2. **Is the eos_bos EV ladder recomputed or relabelled? (carried, from ## Code Review 3)** — if relabelled, every `--vardims default` ESM-IF run selected dims by eosrepeat variance ranking on eos/bos-repeat embeddings; if recomputed, the identical dim lists at all 10 rungs need a word of explanation in the run README. Nothing in git decides it.
3. **Provenance (carried)** — 0 tracked `cv_results_*.json`, 0 tracked `data/` files; recomputed ev-sweep tables/figures are tracked, but the dim21/varfold redo outputs are not, `variance_sweep_vs_best_delong.tsv` is stale, and the raw-arm AUC 0.649814 appears only inside the two DeLong TSVs — no tracked table shows where `raw_flat` came from.
4. **Carried open questions, none answered** — variant folds regenerated rather than derived (`01_datasplit.py`/`01b`); ESM-IF pooling arms dropped from the 22/23 investigations; no ESM-C 6B embeddings in git; structure-audit CA-only vs complete-backbone residue semantics; the two restyled netmhcpan PNGs (`results/figures/netmhcpan/confusion_matrix_9mer.png`, `affinity_bias_density_after.png`) still last changed 2026-09-14 vs script 2026-10-30.
5. **Local checkout is stale (process, carried, re-verified)** — local `hilarius` at 26ec48d, exactly 147 commits behind `origin/hilarius` (`git rev-list --count hilarius..origin/hilarius` → 147), working tree clean, `git fetch --dry-run` reports nothing pending. The ESM-IF reroute, the config purge and all recomputed tables are absent from the working tree; anything edited locally targets content the remote no longer has.

## GitHub

From the collected GitHub data (not re-run): no open PRs (`gh pr list` → []), no open issues (`gh issue list` → []), no workflow runs (`gh run list` → []); repository `TheHilarius/thesis2026`, default branch `hilarius`. From git: origin/hilarius = 76735ad (2026-11-02 13:39:56 +0100), unchanged since the previous report; `git fetch --dry-run` shows nothing pending. Local branch `hilarius` is 147 commits behind with a clean working tree. `reports/overnight/` is gitignored (`.gitignore:115`), so this report exists only in the local working tree.

## Recommended Next Steps

Carried from the 2026-11-03 report; none were resolved (no commits landed); prioritized, maximum five:

1. **Decide the variant's construction of record (carried, blocking)**: choose the Python filter or the R restricted-pool re-sampling, delete or wire the other (`04_evaluate_netmhcpan_sensitivity.R:805-807` vs `run_pipeline.R:37` / `05_protein_features.R:13`), record the variant's row/label/protein counts somewhere citable, and state whether variant folds are row-matched to base.
2. **Reconcile the ev-sweep result table trio (carried, top results-integrity item)**: recompute or drop `runs/ev_matched_topXvariance_sweep/results/variance_sweep_vs_best_delong.tsv:12-21` so the directory has one ESM-IF truth, and answer whether `results/tables/dim_variance_esmif_eos_bos_repeat_ev*.json` were recomputed or relabelled (add the missing `dim_variance_esmif_eos_bos_repeat.csv` either way).
3. **Consolidation commit (carried, code-debt list)**: one `src/investigation/delong_stats.py` (~40 lines: `delong_test`, `holm_adjust`, `star`, `load_oof`) replacing 6 + 5 + 10 copies and the 8 `spec_from_file_location` sites.
4. **Provenance for the redo outputs (carried)**: `git add` the tracked-worthy `runs/dim21_eosbos/results|plots` and `runs/varfold_pilot_eosbos/results|plots` (or state the server is the record), regenerate `runs/esmif_eos_sweep/plots/pca_sweep_*.png` after the 76735ad label change, and refresh or untrack the two stale netmhcpan PNGs.
5. **Docs and purge tail (carried)**: delete the 7 dead `10_pad_subset_auc.py` references in `src/investigation/README.md`, index `24_dimselect_delong.py`–`28_cnn_compare_barplot.py`, drop `emb_esmif_eosrepeat` from `28_cnn_compare_barplot.py:94-95,105,238` and the run READMEs, document `THESIS_VARIANT` end-to-end, and delete the dead `load_delong_pick` (`22_pool_barplot.py:100`) and `esmif_` arm (`23_pool_delong.py:97`).

Nothing was resolved since the previous report — origin/hilarius did not move, so all carried items remain open at 76735ad.

## Reviewed Through

76735ad
