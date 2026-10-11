# Thesis Overnight — 2026-10-11

## Summary

One commit since the previous report (93b86d2..origin/hilarius = f2a65b4, no source changes): the first ESM-C 300M modelling run, committed as a 635-line nested-CV log plus two data symlinks — XGB on 300m padtoken windows only, flat PCA to k=5883 (91.2% EV), val AUC-ROC 0.6886 ± 0.0056 over 6 outer buckets, MCC 0.2687 ± 0.0097, 21.4 h wall time. Its result JSON is gitignored, so the log is again the only in-repo evidence for the number. All carried blockers were re-verified unfixed at tip: the ESM-IF embedder's tuple-unpack crash (seventh consecutive report), 15_padmode_significance.py's import of three names 09 no longer defines, and the run README's reproduce commands that don't match 09's CLI.

## Changes

Only f2a65b4 ("esmc 300m run with padtoken", 2026-10-10 20:03 +0200), 3 files, +637, no files under src/ touched:

- runs/esmc300vs600/data/df_all_with_folds.csv — new symlink (mode 120000) → ../../../data/processed/df_all_with_folds.csv.
- runs/esmc300vs600/data/esmc300m_win_padtoken_prepared.h5 — new symlink → ../../../data/processed/embeddings_prepared/esmc300m_win_padtoken_prepared.h5. Both symlinks follow the existing run-data convention (the same layout as runs/padfull_flat_20261001_1645/data/).
- runs/esmc300vs600/logs/flat/04_modelling_xgb_emb_esmc300m_win_padtoken_pca5883_flat_log_20261009_224239.txt — full harness log: run started 2026-10-09 22:42, completed 2026-10-10 19:35 (75,153 s ≈ 21.4 h, 30 models = 6×5 nested CV).

## Experiments / Results

**ESM-C 300M padtoken modelling run** (first 300m modelling artifact; all figures from the committed log):
- Setup: feature set emb_esmc300m_win_padtoken only, 46,617 samples, windows 29×960 (flat 27,840 dims), flat-mode PCA refit inside every outer fold on streamed rows (~30,100–30,140 fully-real rows kept of ~31,077 streamed), k=5883 at 91.2% EV. XGBoost nested CV: 6 outer buckets × 5 inner folds, random_state=42 (log lines 8-18, 70-145).
- Per-outer-bucket validation AUC-ROC: 0.6839, 0.6859, 0.6815, 0.6933, 0.6890, 0.6979 → summary Val AUC-ROC 0.6886 ± 0.0056; Val MCC 0.2687 ± 0.0097; inner-fold AUCs 0.638–0.659.
- Context (observation, not a controlled comparison): the committed 600m padtoken figure in runs/padmode_choice is held-out AUC 0.6880 at k=6389 (@90% EV). The near-equality 0.6886 vs 0.6880 is suggestive only — the two runs differ in PCA k/EV and the committed 600m number is the grid's held-out column, not a stated 6-bucket mean. Whether this constitutes evidence about model size is a question for the authors.
- Not reproducible from the repo: runs/esmc300vs600/models/flat/cv_results_xgb_emb_esmc300m_win_padtoken_pca5883_flat_20261009_224239.json (models/ gitignored), and both committed data symlinks dangle in a fresh clone (data/ untracked).

No other new results in the range. Note (mitigating a carried concern): the run loaded data/processed/df_all_with_folds.csv at 22:42 on 2026-10-09, after the mean-only 104-column datasplit completed at 14:34 that day, so this run appears to be post-refactor — unlike the padmode-choice grid.

## Code Review — Shortcomings & Issues

No new code shipped in this range (two symlinks and a log, both following existing conventions), so no new findings. Carried findings, re-verified at f2a65b4:

1. **15_padmode_significance.py still cannot run** — 15:80-83 does `importlib.import_module("09_padmode_subset_auc")` and grabs PADMASK_FILES/PADTYPE_MAP/load_padded_masks; grep of 09 at tip finds none of the three names, so import raises AttributeError. This also fails the run README's reproduce step (runs/padmode_choice/README.md:118). Severity: bug. [carried]
2. **README reproduce command uses a flag 09 doesn't have** — runs/padmode_choice/README.md:115-117 invokes `09_padmode_subset_auc.py --models-dir ...`; 09 at tip has five add_argument calls (:174-199), none named --models-dir. Severity: risk (reproducibility of headline numbers). [carried]
3. **ESM-IF embedder still crash-broken (top blocker)** — src/tools/esm/embed_peptides_esmif.py:517 unpacks an 8-element tuple into 7 targets (`Wz, Wp, Wb, We, pm, npad, cpad = (rep, ns, ps, pe, ce, pad_rep, eos_rep, bos_rep)`) → ValueError on the first matched peptide; live caller run_embeddings_esmif_windows.sh:34. Severity: bug. [carried — seventh report]
4. **Triplicated pad-mask/OOF and duplicated stat helpers** — three implementations of pad-mask loading + OOF reconstruction across 09 (load_pad_metadata:55), 14 (:90, :225-271) and 15 (:104-118), which already disagree (any-pad vs informative); duplicated Holm/BH/paired-t helpers in 14 vs 15. One small shared module deletes both this and finding 1. Severity: maintainability. [carried]
5. **Stale docs and dead weight** — investigation/README.md still references the deleted 10_pad_subset_auc.py (:13, :33, :139-141, :248, :262, :289); paired_tests_OLD.csv and a zero-byte log still tracked; run_prepare_windows_slurm.sh:25-27 comment describes 8 keys while the command prepares 4; duplicate .gitignore block; stale embed docstrings and dead code (build_windows, ZERO_WINDOW, args=None) in embed_peptides_esmif.py. Severity: maintainability/style. [carried]
6. **`pca_components: 64` in config is not what flat-mode runs use** — config.py:438-443 lists pca_components 64 for esmc300m_win_padtoken (and 64 for every window key, :387-436), but this run produced 5883 components (EV-driven flat PCA overrides the field; log line 77). A reader treating config as the run spec gets the wrong k. Severity: style. [new, from this range]

Clean for the record: f2a65b4's symlinks match the established run-data layout; the log shows the PCA refit correctly nested inside each outer fold with a variance cross-check (91.2% at k=5883 over 27,840 dims), consistent with the 600m harness behaviour.

## Potential Issues

1. **Provenance chain still broken (carried, blocking for quoted numbers)** — the 300m run's cv_results JSON is gitignored, 15 crashes on import, and the README reproduce block doesn't match 09's CLI. The 0.6886 exists only inside a log file.
2. **The 300m-vs-600m comparison rests on one arm and a cross-protocol reading (new)** — only the 300m padtoken arm has been run, and the only committed 600m counterpart (0.6880) comes from a different PCA budget. Question for the authors: will the remaining 300m pad modes be run, and will the size comparison be made under matched k/EV and a matched metric before it is quoted?
3. **ESM-IF pipeline broken at tip (carried, blocking)** — any re-embedding or new-structure run fails; existing prepared ESM-IF H5s are unaffected. One-line fix pending (next step 2).
4. **Committed matrices predate the mean-only refactor (carried, now partly mitigated)** — the 13 padmode-choice runs remain pre-refactor; the new 300m run appears post-refactor (see Experiments), so the two families of committed numbers may sit on different feature tables. Worth one label or one line in the run READMEs.
5. **data/ remains fully untracked (carried)** — fresh-clone runs need rsync; the two new symlinks and the 300m H5 add to the untracked set; no campaign's cv_results_*.json is committed; the 72-run winmat_flat aggregate is still uncommitted and the committed grids still lack ESM-C flat_raw rows.
6. **Report storage convention (carried, coherent)** — hilarius gitignores reports/overnight/ (published only to the `reports` branch); origin/reports has advanced to 0133d55 and remains unmerged into hilarius by design. This report follows the same convention; publication is a separate step outside this run's read-only scope.

## GitHub

From the collected GitHub data (not re-run):

- **PRs:** none open (`gh pr list` returned []).
- **Issues:** none open (`gh issue list` returned []).
- **CI:** no workflow runs (`gh run list` returned []); no workflows configured.
- **Branch/state:** `TheHilarius/thesis2026`, default branch `hilarius`. From git: origin/hilarius = f2a65b4 (1 commit since the previous report's 93b86d2); local branch HEAD is still 26ec48d, 34 commits behind origin (all reviewed commits exist only in the remote-tracking ref in this clone). origin/reports tip is 0133d55 and is not an ancestor of hilarius.

## Recommended Next Steps

1. Restore the pad-mask API so the significance script runs: either put PADMASK_FILES/PADTYPE_MAP/load_padded_masks (informative per-toolkit masks) back into 09_padmode_subset_auc.py, or move them plus the OOF-reconstruction helper into one small shared module imported by 09/14/15 — then fix the run README's `--models-dir` reproduce command and regenerate padmode_subset_auc.csv with the committed script. [carried — top priority]
2. Revert the broken line: restore the `build_windows(...)` call at src/tools/esm/embed_peptides_esmif.py:517, smoke-test via run_embeddings_esmif_windows.sh. [carried — blocking, unfixed for seven reports]
3. Decide the comparison design for the 300m campaign: confirm whether the remaining 300m pad modes will be run and whether 300m-vs-600m will be compared at matched k/EV and metric; meanwhile commit (or rsync-note) the 300m cv_results JSON so 0.6886 has a machine-readable source. [new]
4. Aggregate the 72 winmat_flat runs on the server (`extract_model_metrics.py --results "runs/winmat_flat_20260928_164804/models/flat/cv_results_*.json"`), commit the TSV next to the logs, and decide the missing ESM-C flat_raw arms; label the pre-refactor padmode matrices in the run README. [carried]
5. Housekeeping in one pass: update investigation/README.md (six references to the deleted 10_pad_subset_auc.py), delete paired_tests_OLD.csv and the zero-byte log, fix the run_prepare_windows_slurm.sh comment, drop the duplicate .gitignore block, delete the carried dead code, refresh the stale embed docstrings, and clarify that config's `pca_components: 64` is overridden by flat-mode EV PCA. [carried + new]

Resolved since the 2026-10-10 report: the "no 300m modelling artifacts" gap (first arm committed: padtoken, 0.6886). Nothing else changed: every carried finding above was re-verified unfixed at f2a65b4, and the ESM-IF embedder remains the top blocker, unchanged for seven consecutive reports.

## Reviewed Through

f2a65b4
