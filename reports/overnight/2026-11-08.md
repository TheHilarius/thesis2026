# Thesis Overnight — 2026-11-08

## Summary

The reviewed range 9419d46..origin/hilarius contains **0 commits**: a live `git ls-remote` still resolves `refs/heads/hilarius` to 9419d468a058be909d6113b240aad171bc3c38f7 (2026-11-06 13:42:13 +0100), so the tip is unchanged from the previous report's anchor and no code, figures, tables, READMEs, or configs landed since 2026-11-06. Consequently there are no new changes, no new repository-supported results, and no new code-review findings; every carried item from the 2026-11-07 report was re-verified at 9419d46 and remains open, and no previously reported number changed. The past-7-days window is the same 13 commits already reported. The local checkout is still 155 commits behind origin/hilarius with a clean tree.

## Past 7 Days

Calendar window 2026-11-02..2026-11-08: 13 commits. Commits per day (count then date):

      5 2026-11-02
      2 2026-11-05
      6 2026-11-06

2026-11-07 and 2026-11-08 had **0** commits; the most recent day with activity is 2026-11-06 (2026-11-04 and 11-03 also had none; one logo commit was authored 11-04 and committed 11-06). Theme level only — per-file detail stays under ## Changes:

- **ESM-IF eos/bos-repeat consolidation (2026-11-02, five commits)** — investigation figures rerouted to the corrected eos/bos-repeat key, ev-sweep figures recomputed, eosrepeat removed from config/scripts, eos_bos EV ladder tracked in place of the eosrepeat ladder, and a y-axis rename pass on two figure scripts.
- **CNN scheme change, homology audit, legacy purge (2026-11-05, two commits)** — nested CNN CV replaced by plain 6-fold CV with a k5/k7/k9 kernel sweep and a SLURM runner; a peptide homology cluster map and a fold-difficulty permutation audit added; seven legacy scripts deleted.
- **Sequence-logo suite (2026-11-04/06)** — ggseqlogo figure set (differential, per-class, per-fold) plus a DiffLogo script with a per-position permutation test and a tracked p-value TSV; the old quarto logo HTML report and its assets deleted.
- **Full-vs-reduced dataset A/B and flank5 variant (2026-11-06)** — a comparison script pairing DeLong and bootstrap tests on the shared peptide subset between the base and remove715negonly datasets, plus a 29mer→19mer flank5 derivation script (no re-embedding).
- **Figure refreshes (2026-11-06)** — logo PNGs regenerated for the new data; netmhcpan confusion matrix and affinity-bias density re-rendered (binary-only changes).

## Changes

**No changes: the reviewed range 9419d46..origin/hilarius is empty (0 commits).** `git log --oneline 9419d46..origin/hilarius` returns nothing and `git ls-remote origin refs/heads/hilarius` confirms the remote branch tip is still 9419d46, so nothing not already covered by the 2026-11-07 report can be listed here.

## Experiments / Results

No new results in the range (0 commits). Re-verified at 9419d46; all previously reported numbers stand:

- **DiffLogo per-position table (tracked)** — `results/figures/logo/difflogo_pvalues.tsv` still has 29 data rows; 27 rows with raw p < 0.05 and 26 with BH < 0.05 (N8 p_raw 0.5203, N9 p_BH 0.05345, C10 not significant), matching the previous report. New detail: the `significant_05` column follows the **raw** p-value (`significant_05 = pvals < 0.05`, `13_difflogo_significance.R:211`), so the N9 row reads `p_BH` 0.05345 (> 0.05) yet `significant_05 TRUE` — the flag is not the BH verdict.
- **Still code-only** — 0 tracked `cv_results_*.json`, 0 tracked files under `data/`, 0 tracked files under `runs/ab_dataset_test/` (the path is not gitignored); the DeLong/bootstrap A/B verdict still exists only as the script and its printed caption.
- **ev-sweep table unchanged** — `runs/ev_matched_topXvariance_sweep/results/variance_sweep_vs_best_delong.tsv` still reports `esmif_eosrepeat` best 0.648827 vs `best_vs_raw` 0.649814 (its last data row), i.e. the raw-flat arm remains nominally higher than the reported "best".
- Cluster map and fold-audit tables remain untracked; the kernel sweep has no tracked output; no A/B or logo-pipeline numeric file other than `difflogo_pvalues.tsv` is tracked.

## Code Review — Shortcomings & Issues

No new code entered the range (0 commits), so there are no new findings; the list below re-verifies carried findings at 9419d46, redundancy/over-engineering first.

1. **PPM helper + alphabet + data preamble copied three times (carried, maintainability — redundant, re-verified)** — `get_ppm_for_label` exists at `src/investigation/10_categorical_residue_analysis.R:269`, `12_sequence_logos.R:49`, `13_difflogo_significance.R:84`, each with its own `aa_alphabet` and `df_clean` block; the scripts' own comments say "copied verbatim" (`12:12`, `13:31`). All three already `source("src/pipeline/r/functions.R")` (`10:13`, `12:19`, `13:46`), which is where the single copy belongs. [carried from 4ef552e, 9419d46]
2. **`variant_path` re-implements `config._variant` (carried, style — reinvents an existing helper, re-verified)** — `src/pipeline/python/reduce_29mer_to_19mer.py:88-91` mirrors `config.py:23-26` (its own docstring says "Mirror config._variant()"); setting `config.VARIANT` and calling `_variant()` deletes the copy. [8ed1a40]
3. **Printed caption hardcodes an agreement claim (carried, risk)** — `src/investigation/29_ab_dataset_compare.py:376-377` asserts "Both tests give the same verdict" unconditionally, so the paste-ready caption can contradict its own `ab_dataset_delong.tsv`; build the sentence from the `star`/`star_boot` fields already in `test_rows`. [f9f5a62]
4. **Variant fallback can mislabel full-dataset runs (carried, risk, low)** — `29_ab_dataset_compare.py:100` treats `n != full_n` as "reduced", so a full-dataset arm with any missing OOF row enters the paired table as the variant; the marker test on the same line is the reliable half. [f9f5a62]
5. **`significant_05` label is raw-based next to a `p_BH` column (re-verified, style, low)** — `13_difflogo_significance.R:211`; in the tracked TSV the N9 row has `p_BH 0.05345` with `significant_05 TRUE`, so a reader counting significant positions off that column gets 27 (raw) rather than 26 (BH). One-word fix: name the column `significant_raw_05` or flag on the BH column.
6. **Tracked figure with no producer (carried, risk)** — `results/figures/logo/differential_logo_29mer_trim5.png` still has 0 hits for `trim5` in any tracked file, so it cannot be regenerated from the repository. [b8f1eed]
7. **DiffLogo is an undeclared R dependency (carried, risk, low)** — `renv.lock` contains no DiffLogo entry, so a frozen `renv::restore()` will not provide it; `13_difflogo_significance.R` only fails at runtime with install instructions. [9419d46]
8. **Deleted HTML still referenced (carried, maintainability)** — `logo_comparison.html` was removed in b8f1eed, but `src/logo_comparison.qmd:5` still targets it and `src/README.md:44,60` still cite it as the rendered-output example. [b8f1eed]
9. **README untouched while three scripts landed (carried, maintainability — re-verified)** — `src/investigation/README.md` has 0 hits for `12_sequence_logos`/`13_difflogo`/`29_ab_dataset`, its count at `:5` still reads "28 scripts (6 R, 22 Python)" against 48 tracked `.py`/`.R` files in that directory, and the R group header at `:20` still says "(6)" against 8 R files; `src/README.md:12` still lists the deleted `01b_datasplit_protein_level.py`. [4ef552e..9419d46]
10. **Mechanism debt carried forward (maintainability, re-verified counts)** — `spec_from_file_location` import sites: 9 files in `src/` (the reuse is right, the import mechanism is the debt); `def star` in 10 files, `def delong_test` in 6, `def holm_adjust` in 5; dead `load_delong_pick` at `22_pool_barplot.py` and the unreachable `esmif_` arm at `23_pool_delong.py:97`; Y-axis label split between "AUC-ROC (nested CV, mean +/- std)" (six sites) and "CV-AUC" (`27_variance_optimum.py:215`, `pca_sweep_AUC_delong_comparison.py:480,523`); 4 zero-byte tracked `.txt` logs; `.gitignore` `runs/*/models/` triplicate at `:125`, `:128`, `:160` (plus `runs/*/models/**/*.pkl` at `:142`).
11. **What passes the ladder (carried)** — no new Python dependency or framework (numpy/pandas/h5py/matplotlib only; the normal-tail p-value uses stdlib `erfc` instead of re-reaching for scipy); DiffLogo reuses one 1,000-permutation run for both display scales; the flank5 script asserts every transformation rather than trusting the slice.

## Potential Issues

1. **Which logo is citable? (carried)** — three tracked differential logos from two pipelines plus the orphan `trim5` PNG; only the DiffLogo TSV carries numbers. Authors' call: which figure enters the thesis, and what produced the trim5 variant.
2. **A/B result artifacts are code-only (carried)** — `runs/ab_dataset_test/` has 0 tracked files and is not gitignored, no README entry exists for `29_ab_dataset_compare.py`, and the verdict lives only in a printed caption — same provenance class as the long-standing "0 tracked `cv_results_*.json` / 0 tracked `data/`" issue.
3. **flank5 has no consumer in the repo (carried)** — `flank5` appears only inside `reduce_29mer_to_19mer.py`; nothing sets `THESIS_VARIANT=flank5`, and `05_cnn_models.py:97` still defaults `window=29`. The script's declared caveat (handcrafted structural aggregates stay 10-wide) means a flank5 handcrafted run would silently mix 5- and 10-wide features unless R steps 06/07 are re-run. Authors' call: which model families run on flank5, and whether the structural set is in or out.
4. **Which CNN CV scheme is citable? (carried, blocking for citation)** — `cnn_nested_cv` vs `cnn_cv` still share the same JSON shape and differ only in `mode`; `28_cnn_compare_barplot.py` never reads a `mode` field (its JSON reader at `:133-141` keys on `model_key`/`features_key` and timestamp only), and no tracked `cv_results_*.json` exists to compare.
5. **Untracked results and the ev-sweep mismatch (carried)** — `variance_sweep_vs_best_delong.tsv` unchanged (still `esmif_eosrepeat` best 0.648827 vs raw 0.649814); cluster map, fold audit, all `cv_results` JSONs and everything under `data/` remain untracked.
6. **Which dataset construction is authoritative? (carried, blocking)** — `make_variant_dataset.py` and the R restricted-pool path (`04_evaluate_netmhcpan_sensitivity.R`) both still exist; the A/B script exercises the Python variant only (`29_ab_dataset_compare.py:118-121`), which is evidence of intent, not a recorded decision.
7. **No commits for two days (process)** — origin/hilarius has not moved since 2026-11-06 13:42; the report anchor therefore did not advance, and tomorrow's daily range will stay empty until new commits land.
8. **Local checkout stale (carried)** — local `hilarius` at 26ec48d (2026-10-02), still **155** commits behind, working tree clean (0 modified files); all reviewed commits are absent locally.
9. **Carried open questions, none answered** — eos_bos EV ladder recomputed vs relabelled; variant folds regenerated rather than derived; ESM-IF pooling arms dropped from the 22/23 investigations; no ESM-C 6B embeddings in git; structure-audit CA-only vs complete-backbone residue semantics.

## GitHub

From the collected GitHub data (used verbatim, not re-run): no open PRs (`gh pr list` → []), no open issues (`gh issue list` → []), no workflow runs (`gh run list` → []); repository `TheHilarius/thesis2026`, default branch `hilarius`. From git: origin/hilarius = **9419d46** (2026-11-06 13:42:13 +0100) — **identical to the previous report's anchor**, confirmed against the remote with `git ls-remote`, so 0 commits were added since 2026-11-07. Local branch `hilarius` is 155 commits behind with a clean working tree. `reports/overnight/` is gitignored (`.gitignore:119`), so this report exists only in the local working tree.

## Recommended Next Steps

Carried items are marked; all five remain open. Prioritized, maximum five:

1. **Decide the variant's construction of record (carried, blocking)**: choose the Python filter or the R restricted-pool re-sampling, delete or wire the other (`make_variant_dataset.py` vs `04_evaluate_netmhcpan_sensitivity.R`), and record row/label/protein counts somewhere citable — the A/B script reads the Python variant (`29_ab_dataset_compare.py:118-121`), so confirm that is the decision.
2. **Reconcile the ev-sweep result table trio (carried, top results-integrity item)**: fix `runs/ev_matched_topXvariance_sweep/results/variance_sweep_vs_best_delong.tsv` so the directory has one ESM-IF truth, answer recomputed-vs-relabelled for `results/tables/dim_variance_esmif_eos_bos_repeat_ev*.json`, and add the missing base ranking CSV.
3. **Settle the CNN CV scheme (carried)**: state `cnn_cv` vs `cnn_nested_cv` in the `05_cnn_modelling.py` docstring + README, and key `28_cnn_compare_barplot.py:133-141` on the `mode` field so old and new JSONs cannot mix in one figure.
4. **Land the A/B and logo evidence (carried)**: commit `runs/ab_dataset_test/plots|results` (or state the server is the record), add README sections for 12/13/29_ab, fix the count at `src/investigation/README.md:5` and the R header at `:20`, resolve the trim5 figure's provenance, and either add DiffLogo to `renv.lock` or document it as a manual install.
5. **Consolidate the R logo preamble (carried)**: move `get_ppm_for_label` + `aa_alphabet` + the `df_clean` block into `src/pipeline/r/functions.R` (already sourced by 10/12/13), replacing three verbatim copies in one edit — plus the one-word `significant_05` column rename while that file is open.

Resolved since the previous report: none — the range added no commits, so every carried item above remains open. (The two items marked resolved in earlier reports — the `02_datasplit_analysis.py` import bootstrap and the `29_build_cluster_map.py` README index — remain resolved.)

## Reviewed Through

9419d46
