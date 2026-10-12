# Thesis Overnight — 2026-10-12

## Summary

No changes in the reviewed range: origin/hilarius is still f2a65b4 ("esmc 300m run with padtoken", 2026-10-10), so the range f2a65b4..origin/hilarius is empty and nothing new to report on commits, experiments, or code. All carried blockers were re-verified unfixed at tip: the ESM-IF embedder's tuple-unpack crash (eighth consecutive report), 15_padmode_significance.py's import of three names 09 no longer defines, and the padmode-choice run README's reproduce command using a flag 09 doesn't have. The only movement in the ecosystem is on the reports branch (origin/reports advanced from 0133d55 to 68c606f, the 2026-10-11 report), which is outside the hilarius branch by design.

## Changes

Empty range. `git rev-list --count f2a65b4..origin/hilarius` returns 0; origin/hilarius tip commit f2a65b4 is exactly the last commit covered by the previous report (2026-10-11). Local working tree is clean at 26ec48d (34 commits behind origin, unchanged), so no uncommitted source changes are visible either.

## Experiments / Results

No new results in the range. The most recent committed experiment remains the ESM-C 300m padtoken modelling run from f2a65b4 (nested-CV log: val AUC-ROC 0.6886 ± 0.0056, MCC 0.2687 ± 0.0097 over 6 outer buckets, k=5883 flat PCA @91.2% EV, 21.4 h wall time — figures as reported on 2026-10-11; its cv_results JSON remains gitignored, so the log is still the only in-repo source).

## Code Review — Shortcomings & Issues

No new code shipped in this range, so no new findings. Carried findings, all re-verified unfixed at origin/hilarius (f2a65b4):

1. **15_padmode_significance.py still cannot run** — src/investigation/15_padmode_significance.py:80-83 does `importlib.import_module("09_padmode_subset_auc")` then reads `PADMASK_FILES`, `PADTYPE_MAP`, `load_padded_masks`; grep of src/investigation/09_padmode_subset_auc.py at tip finds none of the three names, so the import raises AttributeError. Also fails the run README's reproduce step (runs/padmode_choice/README.md:117). Severity: bug. [carried]
2. **README reproduce command uses a flag 09 doesn't have** — runs/padmode_choice/README.md:115-116 invokes `09_padmode_subset_auc.py --models-dir ...`; 09's five add_argument calls (:174-200) are `--results`, `--since`, `--out`, `--pad-h5`, `--save-rows` — no `--models-dir`. Severity: risk (reproducibility of the headline padmode numbers). [carried]
3. **ESM-IF embedder still crash-broken (top blocker)** — src/tools/esm/embed_peptides_esmif.py:517 unpacks an 8-element tuple into 7 targets (`Wz, Wp, Wb, We, pm, npad, cpad = (rep, ns, ps, pe, ce, pad_rep, eos_rep, bos_rep)`) → ValueError on the first matched peptide; live caller run_embeddings_esmif_windows.sh:34. Severity: bug. [carried — eighth report]
4. **Triplicated pad-mask/OOF and duplicated stat helpers** — three implementations of pad-mask loading + OOF reconstruction across 09 (load_pad_metadata), 14 and 15, which already disagree (any-pad vs informative); duplicated Holm/BH/paired-t helpers in 14 vs 15. One small shared module deletes both this and finding 1. Severity: maintainability. [carried]
5. **Stale docs and dead weight** — investigation/README.md still references the deleted 10_pad_subset_auc.py; paired_tests_OLD.csv and a zero-byte log still tracked; run_prepare_windows_slurm.sh comment describes 8 keys while the command prepares 4; duplicate .gitignore block; stale embed docstrings and dead code (build_windows, ZERO_WINDOW, args=None) in embed_peptides_esmif.py. Severity: maintainability/style. [carried]
6. **`pca_components: 64` in config is not what flat-mode runs use** — src config pca_components lists 64 for every window key, but the committed 300m run produced 5883 components (EV-driven flat PCA overrides the field). A reader treating config as the run spec gets the wrong k. Severity: style. [carried from the 2026-10-11 report]

## Potential Issues

1. **Provenance chain still broken (carried, blocking for quoted numbers)** — the 300m run's cv_results JSON is gitignored, 15 crashes on import, and the README reproduce block doesn't match 09's CLI. The 0.6886 exists only inside a log file.
2. **The 300m-vs-600m comparison still rests on one arm and a cross-protocol reading (carried)** — only the 300m padtoken arm has been run, and the only committed 600m counterpart (0.6880, runs/padmode_choice) comes from a different PCA budget and a different metric column. Question for the authors: will the remaining 300m pad modes be run, and will the size comparison be made under matched k/EV and a matched metric before it is quoted?
3. **ESM-IF pipeline broken at tip (carried, blocking)** — any re-embedding or new-structure run fails; existing prepared ESM-IF H5s are unaffected. One-line fix pending.
4. **Committed matrices predate the mean-only refactor (carried, partly mitigated)** — the 13 padmode-choice runs remain pre-refactor; the 300m run appears post-refactor, so the two families of committed numbers may sit on different feature tables. Worth one line in the run READMEs.
5. **data/ remains fully untracked (carried)** — fresh-clone runs need rsync; no campaign's cv_results_*.json is committed; the 72-run winmat_flat aggregate is still uncommitted and the committed grids still lack ESM-C flat_raw rows.
6. **Report storage convention (carried, coherent)** — hilarius gitignores reports/overnight/ (published only to the `reports` branch); origin/reports advanced this cycle from 0133d55 to 68c606f (the 2026-10-11 report) and remains unmerged into hilarius by design. This report follows the same convention; publication is a separate step outside this run's read-only scope.

## GitHub

From the collected GitHub data (not re-run):

- **PRs:** none open (`gh pr list` returned []).
- **Issues:** none open (`gh issue list` returned []).
- **CI:** no workflow runs (`gh run list` returned []); no workflows configured.
- **Branch/state:** `TheHilarius/thesis2026`, default branch `hilarius`. From git: origin/hilarius = f2a65b4 (unchanged since the previous report, 0 commits in range); local branch HEAD is still 26ec48d, 34 commits behind origin. origin/reports advanced from 0133d55 to 68c606f and is not an ancestor of hilarius.

## Recommended Next Steps

1. Restore the pad-mask API so the significance script runs: either put PADMASK_FILES/PADTYPE_MAP/load_padded_masks (informative per-toolkit masks) back into 09_padmode_subset_auc.py, or move them plus the OOF-reconstruction helper into one small shared module imported by 09/14/15 — then fix the run README's `--models-dir` reproduce command and regenerate padmode_subset_auc.csv with the committed script. [carried — top priority]
2. Revert the broken line: restore the `build_windows(...)` call at src/tools/esm/embed_peptides_esmif.py:517, smoke-test via run_embeddings_esmif_windows.sh. [carried — blocking, unfixed for eight reports]
3. Decide the comparison design for the 300m campaign: confirm whether the remaining 300m pad modes will be run and whether 300m-vs-600m will be compared at matched k/EV and metric; meanwhile commit (or rsync-note) the 300m cv_results JSON so 0.6886 has a machine-readable source. [carried]
4. Aggregate the 72 winmat_flat runs on the server (`extract_model_metrics.py --results "runs/winmat_flat_20260928_164804/models/flat/cv_results_*.json"`), commit the TSV next to the logs, and decide the missing ESM-C flat_raw arms; label the pre-refactor padmode matrices in the run README. [carried]
5. Housekeeping in one pass: update investigation/README.md (references to the deleted 10_pad_subset_auc.py), delete paired_tests_OLD.csv and the zero-byte log, fix the run_prepare_windows_slurm.sh comment, drop the duplicate .gitignore block, delete the carried dead code, refresh the stale embed docstrings, and clarify that config's `pca_components: 64` is overridden by flat-mode EV PCA. [carried]

Nothing new resolved this cycle: the range contained no commits, so every carried item above stands unchanged, and the ESM-IF embedder remains the top blocker, now unfixed for eight consecutive reports.

## Reviewed Through

f2a65b4
