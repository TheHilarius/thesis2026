# Thesis Overnight — 2026-11-09

## Summary

The reviewed range 9419d46..origin/hilarius contains exactly **1 commit** — 906360a, a 5-line guard change in `04_modelling.py` that lets `--ae-latent == D` so the autoencoder can run as a no-compression control arm. That is the first movement of `origin/hilarius` since 2026-11-06 13:42, so the report anchor advances 9419d46 → 906360a. No results, READMEs, configs, or figures changed; every carried item from the 2026-11-08 report was re-verified at 906360a and remains open, and no previously reported number moved. The local checkout is 156 commits behind origin with a clean tree.

## Past 7 Days

Calendar window 2026-11-03..2026-11-09: 9 commits. Commits per day (count then date):

      2 2026-11-05
      6 2026-11-06
      1 2026-11-08

2026-11-03, 2026-11-04, 2026-11-07 and 2026-11-09 (today) have **0** commits by commit date — one sequence-logo commit was authored 11-04 and committed 11-06; the most recent day with activity is 2026-11-08. Theme level only (file:line detail stays under ## Changes):

- **CNN CV simplification, kernel sweep, homology audit, legacy purge (2026-11-05)** — nested CNN CV replaced by plain 6-fold CV with a k5/k7/k9 kernel sweep and a SLURM runner; a peptide homology cluster map and a fold-difficulty permutation audit added; seven legacy scripts deleted.
- **Sequence-logo suite (2026-11-04/06)** — ggseqlogo differential/per-class/per-fold figure set, plus a DiffLogo dual-scale pipeline with a per-position permutation test and a tracked p-value TSV; the old quarto logo HTML report and its assets deleted.
- **Full-vs-reduced dataset A/B (2026-11-06)** — two-panel figure pairing paired DeLong against paired bootstrap on the shared peptide subset, with the accompanying comparison script.
- **flank5 variant derivation (2026-11-06)** — 29mer→19mer reduction script (no re-embedding).
- **Figure refreshes (2026-11-06)** — new netmhcpan confusion matrix and regenerated logo PNGs for the new data.
- **AE no-compression control arm (2026-11-08)** — autoencoder latent bound relaxed to allow `latent == D`.

## Changes

One commit in range, parent 9419d46.

- **906360a** `fix: allow --ae-latent == D for no-compression AE control arm` — `src/pipeline/python/04_modelling.py` only (1 file, +5/−3): the condition at `04_modelling.py:1890` flips from `args.ae_latent >= emb_data["emb_dim"]` to `>`, so `latent == D` now passes the AE wiring block; the error text at `:1892-1893` changes from "latent must shrink" to "latent must not exceed D"; a two-line rationale comment at `:1888-1889` names `latent == D` as the no-compression control (encoder D→D→D). Downstream already tolerates it, so nothing else needed touching: `WindowAE` sets `hidden = max(dim // 2, latent)` (`13_autoencoder_windows.py:78`), which equals D at `latent == D`, and the feature count is `window * latent` (`04_modelling.py:1901`, `:1027`), i.e. the flat dimension — matching the intended control.

## Experiments / Results

No new results in the range: 906360a touches one source file and adds no output artifact. Re-verified at 906360a; previously reported numbers stand:

- **DiffLogo per-position table (tracked)** — `results/figures/logo/difflogo_pvalues.tsv`: 29 data rows; 27 with `p_raw` < 0.05, 26 with `p_BH` < 0.05, and 27 rows flagged `significant_05 TRUE` — the flag follows the raw p-value (`13_difflogo_significance.R:211`), so N9 reads `p_BH 0.05345` with `significant_05 TRUE`.
- **ev-sweep table unchanged** — `runs/ev_matched_topXvariance_sweep/results/variance_sweep_vs_best_delong.tsv` last data row is `best_vs_raw`: `esmif_eosrepeat` best 0.648827 vs `raw_flat` 0.649814, i.e. the raw-flat arm remains nominally higher than the reported "best".
- **Still code-only** — 0 tracked `cv_results_*.json`, 0 tracked files under `data/`, 0 tracked files under `runs/ab_dataset_test/`; cluster-map and fold-audit tables remain untracked; the kernel sweep still has no tracked output.
- **New gap** — 0 tracked path matches an autoencoder run (no `_lat*` run directory, no AE output file), so the control arm enabled by 906360a has no repository evidence of having run.

## Code Review — Shortcomings & Issues

Range findings first (commit 906360a), then carried findings re-verified at 906360a.

1. **The new bound is specified only in the runtime error (style, low — 906360a)** — the `--ae-latent` help at `04_modelling.py:1493-1496` never stated the old bound and still states none; after 906360a the sole specification of `1 ≤ d ≤ D` is the `ValueError` at `:1890-1893`. One clause in the help ("d ≤ D; d == D = no-compression control") makes the flag self-describing. The diff itself passes the ladder: five lines, one condition flip, no new abstraction, no dependency, and no change needed in `13_autoencoder_windows.py`, which already handled `latent == D` — the guard was the single choke point all invocations pass through, and it is the only place that changed.
2. **Control arm has no tracked invocation (risk, low — 906360a)** — `git grep -i autoencoder 906360a` matches only `04_modelling.py` and `13_autoencoder_windows.py`; no `.sh`/`.sbatch` runner and no README mentions `--ae-latent`, and no tracked path matches an AE run tag. The arm this commit exists to enable therefore can only be launched from a typed command line, leaving its provenance outside the repository.
3. **PPM helper + alphabet + data preamble copied three times (carried, maintainability — redundant)** — `get_ppm_for_label` at `10_categorical_residue_analysis.R:269`, `12_sequence_logos.R:49`, `13_difflogo_significance.R:84`, each with its own `aa_alphabet`/`df_clean` block; the scripts' own comments say "copied verbatim". All three already `source("src/pipeline/r/functions.R")`, which is where the single copy belongs. [4ef552e]
4. **`variant_path` re-implements `config._variant` (carried, style — reinvents an existing helper)** — `reduce_29mer_to_19mer.py:88-91` mirrors `config.py:23-26`; its docstring says "Mirror config._variant()". Setting `config.VARIANT` and calling `_variant()` deletes the copy. [8ed1a40]
5. **Printed caption hardcodes an agreement claim (carried, risk)** — `29_ab_dataset_compare.py:374-377` asserts "Both tests give the same verdict" unconditionally, so the paste-ready caption can contradict its own `ab_dataset_delong.tsv`; build it from the `star`/`star_boot` fields already in `test_rows`. [f9f5a62]
6. **Variant fallback can mislabel full-dataset runs (carried, risk, low)** — `29_ab_dataset_compare.py:100` treats `n != full_n` as "reduced", so a full-dataset arm with a missing OOF row enters the paired table as the variant; the `REDUCED_MARKER` test on the same line is the reliable half. [f9f5a62]
7. **`significant_05` is a raw-p flag sitting next to a `p_BH` column (carried, style, low)** — `13_difflogo_significance.R:211`; in the tracked TSV the N9 row has `p_BH 0.05345` with `significant_05 TRUE`, so a reader counting off that column gets 27 (raw) rather than 26 (BH). One-word fix: rename to `significant_raw_05` or flag on the BH column.
8. **Tracked figure with no producer (carried, risk)** — `results/figures/logo/differential_logo_29mer_trim5.png` still has 0 `trim5` hits repo-wide at 906360a, so it cannot be regenerated from the repository. [b8f1eed]
9. **DiffLogo is an undeclared R dependency (carried, risk, low)** — `renv.lock` has 0 DiffLogo entries, so a frozen `renv::restore()` will not provide it; `13_difflogo_significance.R` only fails at runtime with install instructions.
10. **Deleted HTML still referenced (carried, maintainability)** — `logo_comparison.html` was removed in b8f1eed, but `src/logo_comparison.qmd:5` still targets it and `src/README.md:44,60` still cite it as the rendered-output example.
11. **READMEs untouched while scripts shipped (carried, maintainability)** — `src/investigation/README.md:5` still reads "28 scripts (6 R, 22 Python)" and `:20` "(6)" against 48 tracked `.py`/`.R` files in that directory (8 R); 0 index hits for `12_sequence_logos`/`13_difflogo`/`29_ab_dataset`; `src/README.md:12` still lists the deleted `01b_datasplit_protein_level.py`.
12. **Mechanism debt (carried, maintainability — counts re-verified at 906360a)** — `spec_from_file_location` import sites in `src/` (9 files); `def star` in 10 files, `def delong_test` in 6; dead `load_delong_pick` at `22_pool_barplot.py:100`; unreachable `esmif_` arm at `23_pool_delong.py`; hardcoded `"mode": "nested_cv"` at `04_modelling.py:2164` against `"mode": "cnn_cv"` at `05_cnn_modelling.py:344`, while `28_cnn_compare_barplot.py:136` keys results on `model_key`/`features_key` only; Y-axis label split ("AUC-ROC (nested CV, mean +/- std)" vs "CV-AUC"); 4 zero-byte tracked `.txt` logs; `.gitignore` `runs/*/models/` triplicate at `:125`, `:128`, `:160` (plus `runs/*/models/**/*.pkl` at `:142`).
13. **What passes the ladder (carried, re-verified)** — no new Python dependency or framework (numpy/pandas/h5py/matplotlib only; the normal-tail p-value uses stdlib `erfc` rather than re-reaching for scipy); DiffLogo reuses one 1,000-permutation run for both display scales; the flank5 script asserts every transformation instead of trusting the slice.

## Potential Issues

1. **Which logo is citable? (carried)** — three tracked differential logos from two pipelines plus the orphan `trim5` PNG; only the DiffLogo TSV carries numbers. Authors' call: which figure enters the thesis, and what produced the trim5 variant.
2. **A/B result artifacts are code-only (carried)** — `runs/ab_dataset_test/` has 0 tracked files and is not gitignored, no README entry exists for `29_ab_dataset_compare.py`, and the verdict lives only in a printed caption — same provenance class as the long-standing "0 tracked `cv_results_*.json` / 0 tracked `data/`" issue.
3. **Which CNN CV scheme is citable? (carried, blocking for citation)** — `cnn_nested_cv` vs `cnn_cv` share the same JSON shape and differ only in `mode`; `28_cnn_compare_barplot.py:136` never reads `mode`, and no tracked `cv_results_*.json` exists to compare.
4. **Which dataset construction is authoritative? (carried, blocking)** — `make_variant_dataset.py` and the R restricted-pool path (`04_evaluate_netmhcpan_sensitivity.R`) both still exist; the A/B script exercises the Python variant only (`29_ab_dataset_compare.py:118-121`), which is evidence of intent, not a recorded decision.
5. **flank5 has no consumer in the repo (carried)** — `flank5` appears only inside `reduce_29mer_to_19mer.py`; nothing sets `THESIS_VARIANT=flank5`, and `05_cnn_models.py:97` still defaults `window=29`. Authors' call: which model families run on flank5, and whether the 10-wide structural aggregates stay in (the script's declared caveat means a flank5 run without re-running R steps 06/07 would mix 5- and 10-wide features).
6. **Untracked results and the ev-sweep mismatch (carried)** — `variance_sweep_vs_best_delong.tsv` unchanged (0.648827 vs 0.649814); cluster map, fold audit, all `cv_results` JSONs and everything under `data/` remain untracked.
7. **New: the AE control arm is enabled but unrecorded (authors' question)** — 906360a makes `latent == D` legal, yet no tracked command invokes `--ae-latent` and no AE run output is tracked. Authors' call: does the no-compression arm run as a reported experimental arm (with a citable runner/command and a committed result), or is it a debugging affordance?
8. **Sparse commit days (process)** — 0 commits on 2026-11-07 and on 2026-11-09 (today); one commit landed 2026-11-08 21:39 after two quiet days, so the daily range stays small until more lands.
9. **Local checkout stale (carried)** — local `hilarius` at 26ec48d (2026-10-02), now **156** commits behind, working tree clean (0 modified files); all reviewed commits are absent locally, so file:line cites above come from `git show 906360a:<path>`.
10. **Carried open questions, none answered** — eos_bos EV ladder recomputed vs relabelled; variant folds regenerated rather than derived; ESM-IF pooling arms dropped from the 22/23 investigations; no ESM-C 6B embeddings in git; structure-audit CA-only vs complete-backbone residue semantics.

## GitHub

From the collected GitHub data (used verbatim, not re-run): no open PRs (`gh pr list` → []), no open issues (`gh issue list` → []), no workflow runs (`gh run list` → []); repository `TheHilarius/thesis2026`, default branch `hilarius`. From git: origin/hilarius = **906360a** (2026-11-08 21:39:18 +0100), one commit ahead of the previous report's anchor 9419d46, with 906360a^ = 9419d46. Local branch `hilarius` is 156 commits behind with a clean working tree. `reports/overnight/` is gitignored (`.gitignore:118-119`, "published only to the `reports` branch"), so this report exists only in the local working tree.

## Recommended Next Steps

All five items carried from the 2026-11-08 report remain open; prioritized, maximum five:

1. **Decide the variant's construction of record (carried, blocking)**: choose the Python filter or the R restricted-pool re-sampling, delete or wire the other (`make_variant_dataset.py` vs `04_evaluate_netmhcpan_sensitivity.R`), and record row/label/protein counts somewhere citable — the A/B script reads the Python variant (`29_ab_dataset_compare.py:118-121`), so confirm that is the decision.
2. **Reconcile the ev-sweep result table trio (carried, top results-integrity item)**: fix `runs/ev_matched_topXvariance_sweep/results/variance_sweep_vs_best_delong.tsv` so the directory has one ESM-IF truth, answer recomputed-vs-relabelled for `results/tables/dim_variance_esmif_eos_bos_repeat_ev*.json`, and add the missing base ranking CSV.
3. **Settle the CNN CV scheme (carried)**: state `cnn_cv` vs `cnn_nested_cv` in the `05_cnn_modelling.py` docstring + README, and key `28_cnn_compare_barplot.py:136` on the `mode` field so old and new JSONs cannot mix in one figure.
4. **Land the A/B and logo evidence (carried)**: commit `runs/ab_dataset_test/plots|results` (or state the server is the record), add README sections for 12/13/29_ab, fix the count at `src/investigation/README.md:5` and the R header at `:20`, resolve the trim5 figure's provenance, and either add DiffLogo to `renv.lock` or document it as a manual install.
5. **Consolidate the R logo preamble (carried)**: move `get_ppm_for_label` + `aa_alphabet` + the `df_clean` block into `src/pipeline/r/functions.R` (already sourced by 10/12/13), replacing three verbatim copies in one edit — plus the one-word `significant_05` column rename while that file is open.

Resolved since the previous report: the "no commits for two days" process item — origin/hilarius moved on 2026-11-08 (906360a), so the anchor advanced and the daily range is no longer empty. None of the five items above is resolved by that commit (it touches one validation guard only). Still resolved from earlier reports: the `02_datasplit_analysis.py` import bootstrap and the `29_build_cluster_map.py` README index.

## Reviewed Through

906360a
