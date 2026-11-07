# Thesis Overnight — 2026-11-07

## Summary

origin/hilarius moved from 35ac9ca to 9419d46: **6 commits** (all committed 2026-11-06, one authored 11-04) adding a sequence-logo suite (two new R scripts, one with a DiffLogo permutation test), a full-vs-reduced dataset A/B comparison script, a 29mer→19mer flank5 derivation script, 8 new tracked figures/tables, and deleting the old quarto HTML logo report — net −3,739 lines across 55 files. No previously reported numeric table changed, so all earlier result numbers stand; the only new citable numbers are the tracked `difflogo_pvalues.tsv`. Every item carried from the previous report (CNN CV scheme decision, README cleanup, ev-sweep TSV mismatch, untracked result artifacts) is untouched — the range changed no README, config, or existing pipeline logic. The local checkout is now 155 commits behind origin/hilarius with a clean tree.

## Past 7 Days

Calendar window 2026-11-01..2026-11-07: 13 commits. Commits per day (count then date):

      5 2026-11-02
      2 2026-11-05
      6 2026-11-06

(Counts by committer date: one logo commit was authored 2026-11-04 and committed 2026-11-06; 11-01 and 11-03 had none.) Theme level only — per-file detail stays under ## Changes:

- **ESM-IF eos/bos-repeat consolidation (2026-11-02, five commits)** — investigation figures rerouted to the corrected eos/bos-repeat key, ev-sweep figures recomputed, eosrepeat removed from config/scripts, eos_bos EV ladder tracked and the eosrepeat ladder dropped, plus a y-axis rename pass on two figure scripts.
- **CNN scheme change + homology audit + legacy purge (2026-11-05, two commits)** — nested CNN CV replaced by plain 6-fold CV with a k5/k7/k9 kernel sweep and a SLURM runner; a peptide homology cluster map and a fold-difficulty permutation audit added; seven legacy scripts deleted (−2,766 lines).
- **Sequence-logo suite (2026-11-04/06)** — ggseqlogo figure set (differential, per-class, per-fold) plus a DiffLogo script with its own per-position permutation test and a tracked p-value TSV; the old quarto logo HTML report and its assets deleted.
- **Full-vs-reduced dataset A/B (2026-11-06)** — new comparison script pairing DeLong and bootstrap tests on the shared peptide subset between the base dataset and the remove715negonly variant.
- **flank5 19mer variant (2026-11-06)** — derivation script slicing 29mer CSVs and window H5s down to 19 positions, no re-embedding.
- **Figure refreshes (2026-11-06)** — logo PNGs regenerated for the new data; netmhcpan confusion matrix and affinity-bias density re-rendered (binary-only change).

## Changes

Reviewed range 35ac9ca..origin/hilarius = **6 commits**: 4ef552e, f9f5a62, b8f1eed, 211ded5, 8ed1a40, 9419d46 (all committed 2026-11-06; 4ef552e authored 2026-11-04 09:50). Net 55 files, +1,131/−4,870. Nothing in this range was covered by previous reports, and no README, config, or pre-existing pipeline script was modified.

**A. Sequence-logo scripts [4ef552e, 9419d46]**
- `src/investigation/12_sequence_logos.R` (new, 164 lines) — reads `data/processed/df_all_with_folds.csv` (`:21`); PPM helper and alphabet declared "copied verbatim" from `10_categorical_residue_analysis.R` (`:12`, helper at `:49`, source copy at `10:269`); differential logo + per-class bits logos (`:85-133`); 6-panel per-fold differential logo as fold-stability QC (`:136-164`).
- `src/investigation/13_difflogo_significance.R` (new, 224 lines) — DiffLogo (BMC Bioinformatics 2015) with a 1,000-permutation per-position test (`:133-134`); the p-values are reused for the JS-scale display instead of paying the ~97 s permutation run twice (`:140`); hardcoded chemistry palette to match ggseqlogo's default (`:106`); base-graphics renderer with axis/region labels (`:146`); per-position TSV with `p_raw`/`p_BH` (`:204-215`); header states the permutation null is sampling-noise-only at n1/n2 ~20k+, so effect sizes are the informative numbers (`:21-25`).
- **b8f1eed** — 7 new PNGs plus `results/figures/logo/difflogo_pvalues.tsv`; deleted `results/figures/logo/logo_comparison.html` (1,572 lines) and its quarto `figure-html/` + `libs/` assets (~3,300 lines). **9419d46** regenerated `differential_logo_29mer_difflogo.png` and added `differential_logo_29mer_difflogo_js.png`.

**B. Full-vs-reduced A/B comparison [f9f5a62]**
- `src/investigation/29_ab_dataset_compare.py` (new, 383 lines) — scans `runs/ab_dataset_test/models/*/cv_results_*.json` (`:169-186`); variant detected from JSON content marker with a row-count fallback (`:100`); shared-subset alignment between the base and remove715negonly split CSVs with abort-on-mismatch and per-fold y_true assertions (`:115-153`); paired DeLong (`:231`) + 10k paired bootstrap (`:235`), Holm across features in both (`:250`, `:259`); two-panel figure (`:270-330`); prints a paste-ready caption (`:358-378`).

**C. flank5 19mer derivation [8ed1a40]**
- `src/pipeline/python/reduce_29mer_to_19mer.py` (new, 330 lines) — pops `THESIS_VARIANT` before importing config so base artifacts are read (`:61`); trims flanks, rebuilds pad bookkeeping, and checks X-padding consistency (`:96-151`); drops far-position columns from encoded CSVs and recomputes compositional `frac_*` over 19 positions with a row-sum validation (`:153-225`); slices window H5s at slots `5:24` with post-write pad/label/fold validation (`:227-307`); declared caveat: handcrafted structural aggregates stay 10-wide (`:30-37`).

**D. Figures-only [b8f1eed, 211ded5]**
- `results/figures/netmhcpan/confusion_matrix_9mer.png` and `affinity_bias_density_after.png` re-rendered binary-only; the producer `src/pipeline/r/04_evaluate_netmhcpan_sensitivity.R` last changed at 6bf423f (2026-10-30) and is untouched in this range.
- `results/figures/logo/differential_logo_29mer_trim5.png` (441 KB) added with no producer: 0 references to `trim5` anywhere at the tip.

## Experiments / Results

Repository-supported results only.

- **DiffLogo per-position table (tracked, new)** — `results/figures/logo/difflogo_pvalues.tsv`: 29 positions, `n_obs` 45,864–46,617; raw p < 0.05 at 27/29 (N8 p=0.5203 and C10 p=0.1182 not significant); BH < 0.05 at 26/29 (N9, p_BH=0.05345, drops out too); 14 positions report `p_raw = 0` (underflow at this sample size); largest JS divergence C1 0.01617, then P5 0.01457, C2 0.01441; terminal positions weakest (N10 5.4e-4, N9 4.7e-4, C10 4.1e-4) — consistent with the stated motivation for the flank5 variant.
- **No A/B dataset numbers exist in the repository** — 0 tracked files under `runs/ab_dataset_test/` (the path is not gitignored; the artifacts simply were not committed), so the DeLong/bootstrap verdict exists only as the script and its printed caption.
- **Carried, unchanged at 9419d46** — no tracked `cv_results_*.json`, no tracked files under `data/`, cluster map and fold-audit tables untracked, kernel sweep has no tracked output, and `runs/ev_matched_topXvariance_sweep/results/variance_sweep_vs_best_delong.tsv:12-21` still reads `esmif_eosrepeat` best 0.648827 vs `best_vs_raw` 0.649814.
- No previously reported numeric table changed in this range; all earlier reported numbers stand.

## Code Review — Shortcomings & Issues

New findings first (redundancy / over-engineering / reinvented-helper priority), then carried items re-verified at 9419d46.

1. **PPM helper + alphabet + data preamble copied verbatim three times (new, maintainability)** — `get_ppm_for_label` exists at `10_categorical_residue_analysis.R:269`, `12_sequence_logos.R:49`, `13_difflogo_significance.R:84`, each with its own `aa_alphabet` (`:51` / `:34` / `:62`) and `df_clean` block. All three scripts already `source("src/pipeline/r/functions.R")`, which is where the single copy belongs; the scripts' own comments say "copied verbatim", explicitly acknowledging the duplication. [4ef552e, 9419d46]
2. **`variant_path` re-implements `config._variant` (new, style)** — `reduce_29mer_to_19mer.py:88-91` mirrors `config.py:23-26` (its own docstring says "Mirror config._variant()"). Setting `config.VARIANT = "flank5"` and calling `_variant()` deletes the copy; it exists only because the env var must be popped for input paths (`:61`). [8ed1a40]
3. **Printed caption hardcodes an agreement claim (new, risk)** — `29_ab_dataset_compare.py:376-377` asserts "Both tests give the same verdict" unconditionally, regardless of the DeLong/bootstrap p-values just computed, so the paste-ready caption can contradict its own `ab_dataset_delong.tsv`. Lazy fix: build the sentence from the `star`/`star_boot` fields already in `test_rows`. [f9f5a62]
4. **Variant fallback can mislabel full-dataset runs (new, risk, low)** — `29_ab_dataset_compare.py:100` treats `n != full_n` as "reduced", so a full-dataset arm with any missing OOF row is classified as the variant and enters the paired table on the wrong footing. Marker-only detection (same line, first clause) is reliable; the fallback trades correctness for convenience. [f9f5a62]
5. **Tracked figure with no producer (new, risk)** — `results/figures/logo/differential_logo_29mer_trim5.png` (b8f1eed) has 0 hits for `trim5` in any tracked file, so it cannot be regenerated from the repository. [b8f1eed]
6. **DiffLogo is an undeclared R dependency (new, risk, low)** — `13_difflogo_significance.R:34-38` fails at runtime with install instructions, but `renv.lock` (the repo's R lockfile) contains no DiffLogo entry, so a frozen `renv::restore()` will not provide it. [9419d46]
7. **Three tracked differential logos of the same deliverable (new, maintainability)** — 12 writes `differential_logo_29mer.png` while 13 writes `_difflogo.png` and `_difflogo_js.png` from the same PPMs; no README or doc says which is citable. Question for the authors rather than a removal recommendation. [b8f1eed, 9419d46]
8. **Ninth `spec_from_file_location` import site (new instance of a carried pattern, maintainability)** — `29_ab_dataset_compare.py:41-45` pulls DeLong/bootstrap/Holm from `15_padmode_significance.py`; the count across `src/` is now 9 files (was 8). The reuse itself is the right ladder rung; the file-location import mechanism is the debt. [f9f5a62]
9. **Y-axis convention split grows to six sites (carried, style, new instance)** — `29_ab_dataset_compare.py:324` uses "AUC-ROC (nested CV, mean +/- std)" (same as `20:205`, `22:172`, `25:140`, `26:226`, `28:165`) versus "CV-AUC" at `27_variance_optimum.py:215` and `pca_sweep_AUC_delong_comparison.py:480,523`. [f9f5a62]
10. **Deleted HTML still referenced (new, maintainability)** — `logo_comparison.html` removed in b8f1eed, but `src/logo_comparison.qmd:5` still targets it and `src/README.md:60` still cites it as the example rendered output. [b8f1eed]
11. **README untouched while three scripts landed (carried, now larger, maintainability)** — `src/investigation/README.md` has no section for `12_sequence_logos.R`, `13_difflogo_significance.R`, `29_ab_dataset_compare.py` (0 grep hits); the R group header (`:20`) still says "(6)" against 8 R files; the count at `:5` still reads "28 scripts (6 R, 22 Python)" against 48 files in the directory (40 `.py` + 8 `.R`); the six deleted-script sections (`:169-212`), the seven `10_pad_subset_auc.py` references and the "current generation" pointer at `:270` all remain; `src/README.md:12` still lists the deleted `01b_datasplit_protein_level.py`. No commit in the range touched any README. [4ef552e..9419d46]
12. **Carried findings re-verified unchanged at 9419d46** — helper duplication (`def star` in 10 files, `def delong_test` in 6, `def holm_adjust` in 5); dead `load_delong_pick` at `22_pool_barplot.py:100` (the copy at `20:107` is used) and the unreachable `esmif_` arm at `23_pool_delong.py:97`; `emb_esmif_eosrepeat` at `28_cnn_compare_barplot.py:94,105,238`; 4 zero-byte tracked `.txt` logs; `.gitignore` `runs/*/models/` triplicate (`:125`, `:128`, `:160` plus `runs/**/models/` at `:150`).
13. **What passes the ladder (carried)** — no new Python dependency or framework (numpy/pandas/h5py/matplotlib all pre-existing; the normal-tail p-value uses stdlib `erfc` instead of re-reaching for scipy); 13 reuses one permutation run for both display scales (`:140`); the flank5 script validates every transformation with explicit assertions rather than trusting the slice.

## Potential Issues

1. **Which logo is citable? (new)** — three tracked differential logos from two pipelines plus an orphan `trim5` PNG, and only 13's TSV carries numbers. Authors' call: which figure enters the thesis, and what produced the trim5 variant.
2. **A/B result artifacts are code-only (new)** — `runs/ab_dataset_test/` has 0 tracked files and is not gitignored, no README entry exists for `29_ab_dataset_compare.py`, and the verdict currently lives only in a printed caption. Same provenance class as the long-standing "0 tracked `cv_results_*.json` / 0 tracked `data/`" issue.
3. **flank5 has no consumer in the repo (new)** — `flank5` appears only in `reduce_29mer_to_19mer.py` at the tip; nothing sets `THESIS_VARIANT=flank5`, and `05_cnn_models.py:97` still defaults `window=29`. The declared caveat that handcrafted structural aggregates stay 10-wide (`:30-37`) means a flank5 handcrafted run would silently mix 5- and 10-wide features unless R steps 06/07 are re-run. Authors' call: which model families run on flank5, and whether the structural set is in or out.
4. **Which CNN CV scheme is citable? (carried, blocking for citation)** — `cnn_nested_cv` vs `cnn_cv` still share the same JSON shape and differ only in `mode`; `28_cnn_compare_barplot.py:133-141` still ignores it; no tracked `cv_results_*.json` exists to compare.
5. **Untracked results and the ev-sweep mismatch (carried)** — `variance_sweep_vs_best_delong.tsv:12-21` unchanged; cluster map, fold audit, all `cv_results` JSONs and everything under `data/` remain untracked.
6. **Which dataset construction is authoritative? (carried, blocking)** — `make_variant_dataset.py` and the R restricted-pool path (`04_evaluate_netmhcpan_sensitivity.R`) both still exist; the new A/B script exercises the Python variant only (`29_ab_dataset_compare.py:118-121`), which is evidence of intent, not a recorded decision.
7. **Local checkout stale (process, carried, worse)** — local `hilarius` at 26ec48d (2026-10-02), now **155** commits behind (was 149), working tree clean (0 modified files). All six reviewed commits are absent locally.
8. **Carried open questions, none answered** — eos_bos EV ladder recomputed vs relabelled; variant folds regenerated rather than derived; ESM-IF pooling arms dropped from the 22/23 investigations; no ESM-C 6B embeddings in git; structure-audit CA-only vs complete-backbone residue semantics.

## GitHub

From the collected GitHub data (used verbatim, not re-run): no open PRs (`gh pr list` → []), no open issues (`gh issue list` → []), no workflow runs (`gh run list` → []); repository `TheHilarius/thesis2026`, default branch `hilarius`. From git: origin/hilarius = **9419d46** (2026-11-06 13:42:13 +0100), moved from the previous anchor 35ac9ca (2026-11-05 14:14:35 +0100) — 6 commits. Local branch `hilarius` is 155 commits behind with a clean working tree. `reports/overnight/` is gitignored (`.gitignore:119`), so this report exists only in the local working tree.

## Recommended Next Steps

Carried items are marked. Prioritized, maximum five:

1. **Decide the variant's construction of record (carried, blocking)**: choose the Python filter or the R restricted-pool re-sampling, delete or wire the other (`make_variant_dataset.py` vs `04_evaluate_netmhcpan_sensitivity.R`), record row/label/protein counts somewhere citable — the new A/B script reads the Python variant (`29_ab_dataset_compare.py:118-121`), so confirm that is the decision.
2. **Reconcile the ev-sweep result table trio (carried, top results-integrity item)**: fix `runs/ev_matched_topXvariance_sweep/results/variance_sweep_vs_best_delong.tsv:12-21` so the directory has one ESM-IF truth, answer recomputed-vs-relabelled for `results/tables/dim_variance_esmif_eos_bos_repeat_ev*.json`, and add the missing base ranking CSV.
3. **Settle the CNN CV scheme (carried)**: state `cnn_cv` vs `cnn_nested_cv` in the `05_cnn_modelling.py` docstring + README, and key `28_cnn_compare_barplot.py:133-141` on the `mode` field it already parses so old and new JSONs cannot mix in one figure.
4. **Land the A/B and logo evidence (new)**: commit `runs/ab_dataset_test/plots|results` (or state the server is the record), add README sections for 12/13/29_ab, fix the count at `README:5` and the R header at `README:20`, resolve the trim5 figure's provenance, and either add DiffLogo to `renv.lock` or document it as a manual install.
5. **Consolidate the R logo preamble (new)**: move `get_ppm_for_label` + `aa_alphabet` + the `df_clean` block into `src/pipeline/r/functions.R` (already sourced by 10/12/13), replacing three verbatim copies in one edit.

Resolved since the previous report: none — the range touched no README, config, or result-table file, so every carried item above remains open. (The two items marked resolved in the previous report — the `02_datasplit_analysis.py` import bootstrap and the `29_build_cluster_map.py` README index — remain resolved.)

## Reviewed Through

9419d46
