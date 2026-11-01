# Thesis Overnight — 2026-11-01

## Summary

No commits landed since the previous report: origin/hilarius is still 6bf423f (2026-10-30), the same tip the 2026-10-31 report reviewed. The daily range 6bf423f..origin/hilarius is empty, so ## Changes below contains nothing new, and no data, figures, or results changed either. Every open item from the previous report — the dual variant construction, provenance policy, README staleness, and the carried code debt — remains exactly as documented at 6bf423f.

## Past 7 Days

Calendar window 2026-10-26..2026-11-01 contains 12 commits (3 on 10-26, 6 on 10-28, 2 on 10-29, 1 on 10-30; 10-27, 10-31 and today 11-01 have zero — the most recent day with activity is 2026-10-30). Theme-level view (this section intentionally overlaps earlier reports; per-file detail lives in earlier reports' ## Changes sections):

- **Dataset-variant experiment** — the peptide-set scope change: a config-level path redirect (`THESIS_VARIANT`), a builder script that drops negatives from positive-less proteins (2,271 negatives from 715 proteins, asserted), and a parallel differently-constructed version inside the NetMHCpan evaluation R script; plus a restyle of two of that script's figures.
- **Dimensional reduction / selection alternatives** — the `--vardims` top-variance flag as a first-class reduction beside PCA, fold-nested variance selection (varfold), EV-vs-PCA and variance-optimum figure scripts with DeLong, EV-ladder dim tables and sweep figures, and investigation-script updates (20/24/pca-sweep comparison).
- **Window-embedding pooling** — padmode-style bar charts for fixed-budget sweeps, pooling analysis with DeLong, and the figure reduced to a single ESM-C naive figure anchored on the 643-dim cumulative-variance arm.
- **CNN baseline arm** — pad mask removed (naive GAP), a 2-layer slurm runner, and an 8-bar comparison of 1-layer vs 2-layer CNNs against BLOSUM and top-variance XGB arms on both embeddings.
- **Sweep bookkeeping** — EV-ladder tables and sweep figures committed while models/logs stayed gitignored.

## Changes

The reviewed range 6bf423f..origin/hilarius contains zero commits: `git log --oneline 6bf423f..origin/hilarius` returns nothing and `git rev-parse --short origin/hilarius` = 6bf423f, the same tip already reviewed in the 2026-10-31 report. There is nothing new to document here; the variant switch, builder script, R-side re-sampling, and figure restyles from 6bf423f were covered in that report and are not repeated.

## Experiments / Results

Repository-supported only. No new results: the empty commit range brings no new tracked artifacts, no new `runs/` outputs, no `cv_results_*.json`, no DeLong tables, and no figure or CSV changes in git. The carried numbers are unchanged (ESM-C raw operating point 0.703964 vs best variance rung 0.704873, ns; ESM-IF raw 0.649814 vs 0.648827, ns). The variant CSVs and their asserted counts (2,271 negatives removed / 715 orphan proteins) remain gitignored and recorded only in `make_variant_dataset.py`'s docstring and `--expect-orphans` guard. The two restyled netmhcpan PNGs (`results/figures/netmhcpan/confusion_matrix_9mer.png`, `affinity_bias_density_after.png`) are still stale relative to their producing script.

## Code Review — Shortcomings & Issues

No new code entered the range, so no new findings. Ponytail ordering preserved from the previous report; all items verified unchanged at origin/hilarius = 6bf423f:

1. **Two assertions that cannot fail** — `src/pipeline/python/make_variant_dataset.py:51-56` and `src/pipeline/r/04_evaluate_netmhcpan_sensitivity.R:803-804` re-check "no orphan proteins remain" after a mask/`semi_join` that makes the property true by construction; only the row-count assert (`make_variant_dataset.py:44-47`) can catch drift. Two dead checks that read as validation coverage that does not exist. Severity: style. [6bf423f]
2. **Path/filename list re-typed instead of reused** — `make_variant_dataset.py:20-28` re-derives `DATA_DIR` and hardcodes the seven encoding filenames already held by `config.py:42-49` and the feature components (`config.py:676-706`); the variant name sits in three unlinked places (`config.py:16` comment, `make_variant_dataset.py:33` default, `04_evaluate_netmhcpan_sensitivity.R:806` filename). Severity: maintainability. [6bf423f]
3. **Two differently-constructed datasets share one variant name, one a dead end** — the R restricted-pool re-sampling (`04…R:797-807`) writes a file nothing consumes (grep over origin: `run_pipeline.R:37` and `05_protein_features.R:13` hardcode the base `df_combined_pos_and_neg.csv`); only the Python filter reaches modelling via `config.py:36-37`. Severity: risk (which file is the citable dataset?). [6bf423f]
4. **Variant negatives never get the affinity-bias check the base negatives get** — the KS/Wilcox/Cliff's battery runs only on base `df_negatives` (`04…R:829-830`, `:869-892`, `:921`); the Python filter performs no rank-bin balance check either. Severity: risk (low; interpretation); question for the authors. [6bf423f]
5. **The variant switch is never set and never documented; run dirs would record the wrong split** — only `config.py:20` reads `THESIS_VARIANT`; no run script exports it, no README mentions it, and run scripts still symlink the base `df_all_with_folds.csv` into run directories (`src/run/run_encodings_sweep.sh:58-59`, `run_pca_sweep.sh:39-40`, `run_matrix_windows_flat_slurm.sh:33`, `run_padmode_data_compare_slurm.sh:32`). Severity: risk (provenance). [6bf423f]
6. **`df_all_<enc>.csv` has no variant path** — `make_variant_dataset.py:25-28` only writes the singles, and `08_aa_encoding.R:13` hardcodes the base `df_all.csv`, so analysis scripts reading `df_all_<enc>.csv` cannot see the variant. Severity: maintainability. [6bf423f]
7. **Carried debt, unchanged** — dead `load_delong_pick` (`src/investigation/22_pool_barplot.py:100`); dead `esmif_` arm in the anchor guard (`src/investigation/23_pool_delong.py:96`); the statistical boilerplate spread (6 `delong_test`, 5 `holm_adjust`, ~9 star helpers, 6 `spec_from_file_location` sites); hardcoded `n_features == 781` (`extract_model_metrics.py:69`); README staleness (7 dead `10_pad_subset_auc.py` references, scripts 16–28 unindexed); four zero-byte tracked logs; `.gitignore` `runs/*/models/` triplicated (`:125`/`:128`/`:160`) with the contradictory `:159` comment; `run_pca_sweep.sh:2` still names `run_pca_k.sh`; `runs/padmode_choice/results/paired_tests_OLD.csv` still tracked; the two orphaned `runs/pool_xgb/*.png`. Severity: maintainability/style (carried).
8. **No over-engineering found in the variant switch itself** — `_variant()` is `os.environ` + `Path.with_name`, stdlib, six lines, no new dependency. That part passes the ladder as-is.

## Potential Issues

1. **Which construction is authoritative? (carried, blocking for citation)** — the R re-sampled variant and the Python filtered variant produce different rows under the same variant name; the R one terminates in a file nothing reads. Authors' call needed: is `make_variant_dataset.py` the definition of record?
2. **Variant folds are regenerated, not derived (carried question)** — with `THESIS_VARIANT` set, `01_datasplit.py`/`01b` re-emit folds from the variant raw (`config.py:36-37`, `01_datasplit.py:657-658`, `:810`); nothing asserts retained peptides keep their base fold. Question for the authors: are base and variant folds row-matched, and does any paired/DeLong comparison depend on that?
3. **Provenance (carried)** — 0 tracked `cv_results_*.json`; variant CSVs and any variant runs gitignored (`.gitignore:54`); the two restyled netmhcpan PNGs stale relative to their script. Open question: is the server the citable record, or will `git add -f` / a checksum manifest be committed?
4. **Carried open questions, none answered** — ESM-IF pooling arms dropped from 22/23; `dimselect_sweep` untracked while globs in `25_varfold_barplot.py:52`/`:69` and `13_dim_screening_report.py:63` still point at it; operating-point numbers unchanged (no new runs); no ESM-C 6B embeddings in git; structure-audit CA-only vs complete-backbone residue semantics.
5. **Local checkout is stale (process, carried)** — local `hilarius` sits at 26ec48d, 142 commits behind `origin/hilarius`, working tree clean; the variant code is absent from the working tree, so anything edited from it targets content that no longer exists on the remote.

## GitHub

From the collected GitHub data (not re-run): no open PRs (`gh pr list` → []), no open issues (`gh issue list` → []), no workflow runs (`gh run list` → []); repo `TheHilarius/thesis2026`, default branch `hilarius`. From git: origin/hilarius = 6bf423f (2026-10-30 10:34:27 +0100), unchanged since the previous report — the local branch `hilarius` is 142 commits behind, working tree clean at session snapshot. `reports/overnight/` remains gitignored on the code branch (`.gitignore:119`), so this report exists only in the local working tree.

## Recommended Next Steps

1. **Decide the variant's construction of record (carried, blocking)**: pick the Python filter or the R restricted-pool re-sampling, delete or wire the other (one shared name in `run_pipeline.R:37` / `05_protein_features.R:13`, or the write at `04…R:805-807`), record the variant's row/label/protein counts somewhere citable, and state whether variant folds are row-matched to base.
2. **Consolidation commit (carried, still top of the code-debt list)**: one `src/investigation/delong_stats.py` (~40 lines: `delong_test`, `holm_adjust`, `star`) plus one small barplot helper, replacing 6 `delong_test` copies, 5 `holm_adjust`, ~9 star helpers, all 6 `spec_from_file_location` sites, and the fifth re-copy of the FALLBACK/COLORS/draw_figure skeleton.
3. **Provenance policy (carried)**: `git add -f` or a checksum manifest for `cv_results_*.json` and the variant CSVs; commit `runs/cnn_compare/results/cnn_compare_all.csv` when the CNN runs exist; refresh or untrack `results/figures/netmhcpan/confusion_matrix_9mer.png` and `affinity_bias_density_after.png` so figures match the restyled script.
4. **Docs pass (carried)**: document `THESIS_VARIANT` end-to-end (build → `01`/`01b` with env → `03` → `04`) and make run scripts symlink the variant folds CSV when set; in `src/investigation/README.md` remove the 7 dead `10_pad_subset_auc.py` references, index 16–28, disambiguate duplicated prefixes, and record the pooling-scope change (39659a3).
5. **One tidy commit (carried)**: delete dead `load_delong_pick` (`22_pool_barplot.py:100`) and the dead `esmif_` arm (`23_pool_delong.py:96`); replace the two tautological asserts with the check that matters (rank-bin balance of the variant negatives); drop the 4 zero-byte tracked logs, the `.gitignore` `runs/*/models/` triplicate, and `paired_tests_OLD.csv`.

Nothing carried forward from the 2026-10-31 report was resolved, as no commits landed: the consolidation, provenance policy, variant-construction decision, README staleness, and the 6bf423f leftovers are all unchanged at origin/hilarius = 6bf423f.

## Reviewed Through

6bf423f
