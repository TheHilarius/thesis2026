# Thesis Overnight — 2026-11-03

## Summary

origin/hilarius moved for the first time since the last report: 6bf423f → 76735ad, 5 commits, all on 2026-11-02. The day's work is one theme — every ESM-IF investigation arm is rerouted from the eosrepeat pad mode to eos/bos-repeat: investigation scripts point at the redone runs, `config.py` drops the `esmif_eosrepeat` keys, the EV-ladder JSONs are renamed, and the top-variance sweep CSV/TSV/figure artifacts are recomputed under the new key. Two smaller changes ride along: a y-axis relabel to "CV-AUC" in the PCA-sweep/variance-optimum figure scripts, and a new display-only figure variant that draws baseline-encoding bars. One new 377-line investigation script (`24_dimselect_delong.py`) benchmarks the dimselect campaign against variance/PCA/BLOSUM50. Several tracked tables and READMEs still describe the purged eosrepeat arms (see ## Code Review).

## Past 7 Days

Calendar window 2026-10-28..2026-11-03 contains 14 commits. Commits per day:

      6 2026-10-28
      2 2026-10-29
      1 2026-10-30
      5 2026-11-02

(10-31, 11-01 and today 11-03 have none; the most recent day with activity is 2026-11-02.) Theme-level view — this section intentionally overlaps earlier reports, so per-file detail stays under ## Changes only:

- **Dimensional-reduction alternatives** — the `--vardims` top-variance flag as a first-class reduction beside PCA, fold-nested variance selection plus EV-vs-PCA and variance-optimum figure scripts with DeLong, and the EV-ladder dimension tables with sweep figures/results (models/logs gitignored).
- **Sweep bookkeeping** — investigation-script updates for the 20/24/pca-sweep comparisons, and removal of the tracked `dimselect_sweep` result files (moved to `_legacy`).
- **CNN baseline arm** — pad mask removed for simplicity plus a 2-layer slurm runner, and analysis of the 2-layer CNN models.
- **Window-embedding pooling figures** — a pooling plot update.
- **Dataset-variant construction** — dropping the negatives that come from proteins with no positive peptides.
- **ESM-IF pad-mode consolidation (2026-11-02)** — all ESM-IF investigation arms moved to eos/bos-repeat, the eosrepeat configuration purged, the EV ladder and the top-variance sweep tables/figures recomputed under the new key, plus a figure-label style pass and a new dimselect benchmark script.

## Changes

Reviewed range 6bf423f..origin/hilarius = 5 commits (86b26bd, cbf243c, fd3a515, 216c2f8, 76735ad), all 2026-11-02, 34 files. Nothing here was covered by earlier reports.

**Rerouting ESM-IF arms to eos/bos-repeat [86b26bd]**
- `src/investigation/20_dim21_barplot.py:87-94` gains `D21_ESMIF`, two hardcoded `cv_results_…_simple21_var21_<timestamp>.json` paths (default runtag now `dim21_eosbos`, `:133-134`); new helper `load_cv_json` at `:97-105`; legend/caption text rewritten (`:240`, `:262-265`).
- `src/investigation/21_dim21_delong.py:53-54` swaps the two esmif arms to the same timestamped files; `--runtag` default `dim21_eosbos` (`:90-91`).
- `src/investigation/24_varfold_delong.py:60-66` adds `VF_BOS` + `D21_BOS_VAR21`, reroutes `esmif_vf21/vf50/global21` (`:78-82`), empties `NOTES` (`:99` — the padmode-mismatch caveat is gone), output moves to `runs/varfold_pilot_eosbos/…` (`:114-118`).
- `src/investigation/25_varfold_barplot.py:43-45,66-73` likewise; dimselect globs widened to `**/dimselect_sweep/…` (`:57`, `:72`); output defaults move to `runs/varfold_pilot_eosbos/…` (`:100-110`).
- `src/investigation/26_ev_variance_vs_pca.py:71-80` — `FAM` gains a per-embedding variance-arm glob template (esmc → `models/simple21`, esmif → `models/vardims`), replacing an inline path (`:129`); key check and captions updated (`:133`, `:230-231`, `:271-272`).
- `src/investigation/27_variance_optimum.py:44-49,95,106-107,217,241-247` — same template change plus caption/title cleanup.

**Recomputing the top-variance sweep artifacts [cbf243c]**
- `runs/ev_matched_topXvariance_sweep/plots/ev_variance_vs_pca_auc.csv` (10 rows re-keyed to `esmif_eos_bos_repeat`, values changed), `results/ev_variance_vs_pca_delong.tsv:12-21`, `results/variance_sweep_vs_raw_delong.tsv:12-21` recomputed; both esmif PNGs replaced (`ev_variance_vs_pca_auc_esmif_eos_bos_repeat.png`, `variance_sweep_optimum_esmif_eos_bos_repeat.png` added, eosrepeat versions deleted).
- Not recomputed in the same directory: `results/variance_sweep_vs_best_delong.tsv` (still `esmif_eosrepeat` rows 12-21) and `README.md:13` (still `emb_esmif_eosrepeat`) — see ## Code Review.

**Purging `esmif_eosrepeat` from configuration [fd3a515]**
- `src/pipeline/python/config.py` — ~91 lines removed: the `esmif_eosrepeat`, `esmif_eosrepeat_d21_first` and `esmif_eosrepeat_d21_var` entries from `EMBEDDING_SOURCES`, `FEATURE_COMPONENTS` and `FEATURE_SETS`; `DEFAULT_VAR_EV` re-keyed with a rewritten comment (`:57-66`, now documenting ev65 ***/ev70-ns and "kept at the 260-dim budget on purpose").
- `src/investigation/10_dim_variance.py:47` `DEFAULT_KEYS` → `esmif_eos_bos_repeat`; help-text updates at `src/pipeline/python/04_modelling.py:1473-1474` and `04b_run_matrix.py:366-367`.
- New `src/investigation/24_dimselect_delong.py` (377 lines): dimselect-vs-variance/PCA/BLOSUM50 curve figure plus `results/dimselect_curve_values.tsv` and `results/dimselect_delong_pairs.tsv`; DeLong/Holm reused from `15_padmode_significance.py` via the `importlib` file-load pattern (`:51-55`).

**EV-ladder tables [216c2f8]**
- 12 JSONs renamed `dim_variance_esmif_eosrepeat_ev{30..95}.json` / `_top21.json` → `…_esmif_eos_bos_repeat_…`; the only content change in each is the `embedding_key` line (verified by diffing 6bf423f vs origin/hilarius copies of ev30/ev65/top21).
- `results/tables/dim_variance_esmif_eosrepeat.csv` (513 lines) deleted with no `esmif_eos_bos_repeat` counterpart added; ESM-C keeps its base CSV (`results/tables/dim_variance_esmc300m_win_padtoken.csv`).

**Figure style pass [76735ad]**
- y-axis → `"CV-AUC"` at `src/investigation/27_variance_optimum.py:215` and `src/investigation/pca_sweep_AUC_delong_comparison.py:480,523` (docstring `:48`).
- New display-only figure `pca_sweep_auc_full_withfull_baselines.png` behind a `ref_bars` parameter (`pca_sweep_AUC_delong_comparison.py:373-377`, `:429-452`, `:470-477`, `:761-768`): BLOSUM50/Sparse drawn as grey bars right of a dashed divider, plus a two-entry legend.

## Experiments / Results

Repository-supported results only; no `cv_results_*.json` is tracked (`git ls-tree -r origin/hilarius | grep -c cv_results` → 0) and `data/` still has 0 tracked files, so every number below comes from tracked tables.

- **Recomputed ESM-IF (eos/bos-repeat) variance ladder vs the raw ceiling** — `runs/ev_matched_topXvariance_sweep/results/variance_sweep_vs_raw_delong.tsv:12-21`: best rung ev85 = 0.648705 (392 dims) vs raw 0.649814, delta −0.001109, p_holm 6.311e-01, `ns` (`:20`); ev65 = 0.644742 vs raw, p_holm 6.948e-04, `***` (`:16`); ev70 = 0.647190, p_holm 1.543e-01, `ns` (`:17`, first ns rung). Under the old eosrepeat key the same table's sibling reported best ev85 = 0.648827 vs raw, p = 0.405, `ns` — that row now lives only in the un-recomputed `variance_sweep_vs_best_delong.tsv:21`.
- **ESM-C unchanged** — ev90 = 0.704873 vs raw 0.703964, p_holm 6.227e-01, `ns` (`variance_sweep_vs_raw_delong.tsv:11`); ev80 vs PCA k=2278 delta +0.004466, p_holm 3.833e-03, `**` (`ev_variance_vs_pca_delong.tsv:9`).
- **Star labels moved on the recomputed ESM-IF rungs** — vs raw: ev50 `***`→`ns`, ev60 `**`→`*`, ev75/ev80 `***`→`**`; vs PCA: ev75/ev80 `***`→`**` (`ev_variance_vs_pca_delong.tsv:18-19`). Observed change in the table, not an interpretation of it.
- **The configured default rung sits below the raw ceiling for ESM-IF** — `config.py:64-66` keeps `"esmif_eos_bos_repeat": 65` with the comment "ev65 ***, ev70 = smallest ns rung — kept at the 260-dim budget on purpose"; the ESM-C entry is annotated "(raw 0.7040, ns)". The previous report's quoted rationale ("chosen rungs are DeLong-typed with the raw ceiling") was removed in this range.
- **Tracked figures updated**: both `ev_matched_topXvariance_sweep/plots/` esmif PNGs and the ESM-C optimum PNG regenerated under cbf243c. Not updated: `runs/esmif_eos_sweep/plots/pca_sweep_*.png` (last touched 3190849, 2026-10-23) despite the script changing at 76735ad, and the dim21/varfold redo outputs (`runs/dim21_eosbos/`, `runs/varfold_pilot_eosbos/`) have 0 tracked files.

## Code Review — Shortcomings & Issues

Findings from the range, ordered by redundancy/reinvention first; carried items re-verified against origin/hilarius (76735ad).

1. **A new script copied helpers its neighbours already share (top redundancy finding)** — `src/investigation/24_dimselect_delong.py:158` defines `star()` byte-identical to `21_dim21_delong.py:83` (whose own docstring says "same as 15_/pca_sweep figures"), `:109` re-implements `load_oof` while `26_ev_variance_vs_pca.py:69` and `27_variance_optimum.py` already do `load_oof, star = _d21.load_oof, _d21.star`, and `:51` adds an 8th `importlib.util.spec_from_file_location` site (was 7). `def star` now exists in 10 files (was 9), `def delong_test` in 6, `def holm_adjust` in 5. The script does reuse `_p15.delong_test`/`holm_adjust` (`:173`, `:281`) — the copy-paste stopped one helper short. One shared ~40-line module removes all of it; severity: maintainability (over-engineering by duplication). [fd3a515]
2. **Two tracked result tables in one directory now disagree about the ESM-IF arms** — `runs/ev_matched_topXvariance_sweep/results/variance_sweep_vs_raw_delong.tsv:12-21` and `ev_variance_vs_pca_delong.tsv:12-21` were recomputed for `esmif_eos_bos_repeat`, but `variance_sweep_vs_best_delong.tsv:12-21` still reports `esmif_eosrepeat` (best ev85 = 0.648827; line 21 is the only tracked `best_vs_raw` row for ESM-IF and it describes a key that `config.py` no longer defines). A reader taking "the best ESM-IF rung" from the directory gets two different values (0.648827 vs 0.648705). Severity: risk (results consistency). [cbf243c changed only two of the three tables]
3. **The "eos_bos EV ladder" is content-identical to the eosrepeat ladder** — every one of the 12 renames in 216c2f8 differs from its 6bf423f source by exactly one line (`embedding_key`); the dim lists, `n_dims` (78/122/174/230/260/291/323/357/392/429) and `top21` are unchanged, and the eosrepeat base ranking CSV was deleted while no `dim_variance_esmif_eos_bos_repeat.csv` was added (ESM-C keeps its base CSV). The variance arms themselves were recomputed and do differ (0.640985 vs 0.638911 at ev30), so the ladder that selected their dims either was rerun to an identical result or was relabeled — git cannot tell which. Severity: risk (results provenance); this is a question for the authors, not a defect I can settle. [216c2f8]
4. **A job-tag collision is now hardcoded in four files** — both `dim21_eosbos` jobs wrote filenames containing `simple21_var21`, so the first-21 arm's file is named `…_simple21_var21_20261029_235831.json` and is distinguished only by timestamp, documented in comments (`20_dim21_barplot.py:85-86`, `24_varfold_delong.py:62-66`). The same two literals appear at `20_dim21_barplot.py:87-94`, `21_dim21_delong.py:53-54`, `24_varfold_delong.py:64-66`, `25_varfold_barplot.py:43-45`; any rerun or rename needs four coordinated edits, and a glob over `*var21*` silently matches both arms. Severity: risk (arm identification). [86b26bd]
5. **The purge stopped at configuration** — `src/investigation/28_cnn_compare_barplot.py:105` and `:238` still accept `emb_esmif_eosrepeat` (as a fallback after `emb_esmif_eos_bos_repeat`) and its docstring `:94-95` asserts the on-disk feature_set *is* `emb_esmif_eosrepeat`; `runs/ev_matched_topXvariance_sweep/README.md:13` still lists `emb_esmif_eosrepeat`; `runs/varfold_pilot/README.md` and `runs/varfold_pilot/results/varfold_delong_pairs.tsv` (6 eosrepeat rows) are the tracked record of an arm the scripts no longer read. Severity: maintainability (mixed-arm fallback + stale docs). [fd3a515]
6. **Y-axis relabel is partial across one figure family** — 76735ad renamed to "CV-AUC" in `27_variance_optimum.py:215` and `pca_sweep_AUC_delong_comparison.py:480,523`, while scripts edited the same day still emit `"AUC-ROC (nested CV, mean +/- std)"` (`25_varfold_barplot.py:140`, `26_ev_variance_vs_pca.py:226`) and siblings do too (`20_dim21_barplot.py:205`, `28_cnn_compare_barplot.py:165,275`). Two label conventions will appear side by side in the thesis. Severity: style. [76735ad]
7. **Tracked figures lag the script that draws them** — `runs/esmif_eos_sweep/plots/pca_sweep_auc_*.png` last changed in 3190849 (2026-10-23) while `pca_sweep_AUC_delong_comparison.py` changed at 76735ad (2026-11-02), and the new `pca_sweep_auc_full_withfull_baselines.png` (`:767`) has no tracked PNG; meanwhile the dim21/varfold redo outputs under `runs/dim21_eosbos/` and `runs/varfold_pilot_eosbos/` are entirely untracked while the eosrepeat varfold predecessor is tracked (`runs/varfold_pilot/plots/varfold_vs_refs_auc.{png,csv}`, `results/varfold_delong_pairs.tsv`). Severity: risk (which figures are citable). [76735ad, 86b26bd]
8. **`n_a`/`n_b` columns mean different things in different tracked TSVs** — feature counts in `variance_sweep_vs_best_delong.tsv` (793 / 27840 = 29×960), dim count + PCA-k in `ev_variance_vs_pca_delong.tsv:11` (793 / 5883), while `variance_sweep_vs_raw_delong.tsv` uses explicit `var_n_dims` / `n_features` headers. Severity: style (misread risk when quoting a row). [carried tables, cbf243c]
9. **Carried leftovers, re-verified unchanged** — dead `load_delong_pick` with no caller in its own file (`src/investigation/22_pool_barplot.py:100`; note a second copy at `20_dim21_barplot.py:107` that *is* used); dead `esmif_` branch in the arm-redirect loop (`23_pool_delong.py:97`); tautological orphan asserts (`make_variant_dataset.py:53-55`, only `:47` can fail); README still documents the nonexistent `10_pad_subset_auc.py` 7 times and does not index `24_dimselect_delong.py` (or 25-28); zero-byte tracked logs, the `.gitignore` `runs/*/models/` triplicate and `paired_tests_OLD.csv` all still present. Severity: maintainability/style. [76735ad]
10. **What passes the ladder** — fd3a515 is a net deletion with no new abstraction; `26_ev_variance_vs_pca.py:71-80` replaced a per-key if/else path build with one glob template (less code, one place to edit); `24_dimselect_delong.py` reuses `_p15`'s DeLong/Holm instead of adding a 7th copy; the recomputation in cbf243c reused the existing table/figure plumbing rather than a new script. The `ref_bars` figure flag in 76735ad is the one addition worth watching: it is display-only and explicitly keeps existing CSVs/verdicts byte-identical, but it is the 4th boolean-ish variant of the same bar figure (`special_k`, `refs_mode`, `pending`, now `ref_bars`).

## Potential Issues

1. **Which construction is authoritative? (carried, blocking for citation)** — the R re-sampled variant writes `data/processed/df_combined_pos_and_neg_remove715negonly.csv` (`04_evaluate_netmhcpan_sensitivity.R:805-807`) that nothing consumes, while `run_pipeline.R:37` / `05_protein_features.R:13` hardcode the base CSV. Authors' call: is `make_variant_dataset.py` the definition of record?
2. **New: is the eos_bos EV ladder recomputed or relabeled?** (from ## Code Review 3) — if relabeled, every `--vardims default` ESM-IF run selected dims by eosrepeat variance ranking on eos/bos-repeat embeddings; if recomputed, the identical dim lists at all 10 rungs need a word of explanation in the run README. Nothing in git decides it.
3. **Provenance (carried, partially advanced)** — 0 tracked `cv_results_*.json`, 0 tracked `data/` files; the recomputed ev-sweep tables/figures *are* tracked now (progress), but the new dim21/varfold redo outputs are not, `variance_sweep_vs_best_delong.tsv` is stale, and the raw-arm AUC 0.649814 appears only inside the two DeLong TSVs — no tracked table shows where `raw_flat` came from (its `n_b` columns are feature counts, 14848 = 29×512, so there is no row-count puzzle, just no source table).
4. **Carried open questions, none answered** — variant folds regenerated rather than derived (`01_datasplit.py`/`01b`); ESM-IF pooling arms dropped from the 22/23 investigations; no ESM-C 6B embeddings in git; structure-audit CA-only vs complete-backbone residue semantics; the two restyled netmhcpan PNGs (`results/figures/netmhcpan/confusion_matrix_9mer.png`, `affinity_bias_density_after.png`) still last changed 2026-09-14 vs script 2026-10-30.
5. **Local checkout is stale (process, carried)** — local `hilarius` at 26ec48d, 147 commits behind `origin/hilarius`, working tree clean, `git fetch --dry-run` reports nothing pending. The ESM-IF reroute, the config purge and all recomputed tables are absent from the working tree; anything edited locally targets content the remote no longer has.

## GitHub

From the collected GitHub data (not re-run): no open PRs (`gh pr list` → []), no open issues (`gh issue list` → []), no workflow runs (`gh run list` → []); repository `TheHilarius/thesis2026`, default branch `hilarius`. From git: origin/hilarius = 76735ad (2026-11-02 13:39:56 +0100), advanced from 6bf423f by the 5 commits reviewed above; `git fetch --dry-run` shows nothing pending. Local branch `hilarius` is 147 commits behind with a clean working tree at the session snapshot. `reports/overnight/` is gitignored (`.gitignore:115`), so this report exists only in the local working tree.

## Recommended Next Steps

Carried from the 2026-11-02 report; prioritized, maximum five:

1. **Decide the variant's construction of record (carried, blocking)**: choose the Python filter or the R restricted-pool re-sampling, delete or wire the other (`04_evaluate_netmhcpan_sensitivity.R:805-807` vs `run_pipeline.R:37` / `05_protein_features.R:13`), record the variant's row/label/protein counts somewhere citable, and state whether variant folds are row-matched to base.
2. **Reconcile the ev-sweep result table trio (new, top results-integrity item)**: recompute or drop `variance_sweep_vs_best_delong.tsv:12-21` so the directory has one ESM-IF truth, and answer whether `results/tables/dim_variance_esmif_eos_bos_repeat_ev*.json` were recomputed or relabeled (add the missing `dim_variance_esmif_eos_bos_repeat.csv` either way).
3. **Consolidation commit (carried, code-debt list)**: one `src/investigation/delong_stats.py` (~40 lines: `delong_test`, `holm_adjust`, `star`, `load_oof`) replacing 6 + 5 + 10 copies and the 8 `spec_from_file_location` sites — the range added a copy instead of removing one.
4. **Provenance for the redo outputs (carried, partially advanced)**: `git add` the tracked-worthy `runs/dim21_eosbos/results|plots` and `runs/varfold_pilot_eosbos/results|plots` (or state the server is the record), regenerate `runs/esmif_eos_sweep/plots/pca_sweep_*.png` after the 76735ad label change, and refresh or untrack the two stale netmhcpan PNGs.
5. **Docs and purge tail (carried)**: delete the 7 dead `10_pad_subset_auc.py` references in `src/investigation/README.md` (including the section at line 142), index `24_dimselect_delong.py`, drop `emb_esmif_eosrepeat` from `28_cnn_compare_barplot.py:105,238` and the two run READMEs, document `THESIS_VARIANT` end-to-end, and delete the dead `load_delong_pick` (`22_pool_barplot.py:100`) and `esmif_` arm (`23_pool_delong.py:97`).

Carried items now resolved in part: the `dimselect_sweep` reachability concern is addressed — `25_varfold_barplot.py:57,72` now glob `**/dimselect_sweep/…` and `24_dimselect_delong.py:61-66` explicitly scans `runs/`, `runs/_legacy/` and `runs/dimselect_matched/`, so the dimselect arms are no longer tied to the untracked directory. Everything else carried (variant decision, consolidation, provenance policy, README staleness, 6bf423f leftovers) remains open at origin/hilarius = 76735ad.

## Reviewed Through

76735ad
