# Thesis Overnight — 2026-11-06

## Summary

origin/hilarius moved since the previous report: the reviewed range 76735ad..origin/hilarius is now **2 commits** (both 2026-11-05), tip 35ac9ca. The day's work is a CNN methodology change plus a homology audit: nested CNN CV was replaced by a plain 6-fold CV with a new k5/k7/k9 kernel sweep and a SLURM runner, a reusable peptide-homology cluster map and a fold-difficulty permutation audit were added, and seven legacy scripts (six embedding/inspection scripts plus the protein-level datasplit script) were deleted for −2,766 lines. No tracked numeric result table changed in the range, so every result number carried from earlier reports still stands; the only new results are two findings quoted in `src/investigation/README.md` whose output files are not tracked. The README was updated for the new and fixed scripts but still documents all six deleted scripts as current.

## Past 7 Days

Calendar window 2026-10-31..2026-11-06 contains 7 commits. Commits per day (count then date):

      5 2026-11-02
      2 2026-11-05

(10-31, 11-01, 11-03, 11-04 and today 11-06 have none; the most recent day with activity is 2026-11-05.) Theme level only — per-file detail stays under ## Changes:

- **ESM-IF arm consolidation to eos/bos-repeat (2026-11-02, four commits)** — ESM-IF investigation figures rerouted to the eos/bos-repeat key, ev-sweep figures recomputed under the corrected pad mode, eosrepeat removed from config and scripts, and the eos_bos EV ladder tracked while the eosrepeat ladder was dropped.
- **Figure-label pass (2026-11-02)** — y-axis renamed to "CV-AUC" in the pca-sweep and variance-optimum figure scripts (partial across the figure family; see ## Code Review).
- **CNN scheme reverted to simple 6-fold CV + kernel sweep (2026-11-05)** — nested 6×5 CV replaced by one model per bucket with a fixed seed, three new kernel-size model variants (k5/k7/k9), a dual-format learning-curve plotter, and a SLURM script that sweeps the kernels sequentially on one GPU.
- **Homology/leakage audit (2026-11-05)** — a reusable peptide → homology-cluster map built from the split script's own clustering code, and a fold-difficulty audit with a cluster-preserving permutation null wired into the datasplit analysis script; the datasplit script's import bootstrap was fixed in the same pass.
- **Legacy script purge (2026-11-05)** — four pad-probe embedding scripts, two embedding-inspection scripts, and the protein-level datasplit script deleted.

## Changes

Reviewed range 76735ad..origin/hilarius = **2 commits**: dc8cafb and 35ac9ca, both 2026-11-05 14:13/14:14 +0100. Nothing in this range was covered by previous reports.

**A. CNN CV scheme reverted from nested to simple 6-fold [dc8cafb]**
- `src/pipeline/python/05_cnn_modelling.py:4-24` — docstring rewritten: "NESTED CV … 30 models" → "SIMPLE 6-FOLD CV (experimentation): 6 buckets, 6 models. Same fixed seed for all 6 folds (no per-fold offset)".
- `:135` `train_one_inner` → `train_one_fold` (early-stops on val AUC, seed `args.seed` unchanged across folds); `:185-196` curves CSV flattened to `bucket,epoch,train_loss,val_auc`; `:267-306` the outer×inner loop replaced by one loop over the 6 buckets, models saved as `<tag>_val<vb>.pt` (30 → 6 files); `:344` JSON `mode` `cnn_nested_cv` → `cnn_cv`; `:341-343` config keys `n_outer_folds`/`n_inner_folds` → `n_folds`/`total_models`; `best_inner_per_outer` dropped from the JSON (`:355-356`).
- `src/pipeline/python/05_cnn_models.py:20-31` — three new registry entries `cnn_simple_k5/k7/k9` (identical to `cnn_simple` except kernel size).
- `src/pipeline/python/05_plotting.py:12-21,41-65,87-127` — `load_curves` now returns `(curves, nested)` and branches on CSV header (`outer`+`inner` vs `bucket`); labels changed "outer N" → "bucket N".

**B. Kernel-sweep SLURM runner [35ac9ca]**
- `src/run/run_cnn_kernel_slurm.sh` (new, 58 lines) — GPU partition, venv activation with hard fail, `torch.cuda.is_available()` assert, then a sequential loop over `cnn_simple_k5/k7/k9` calling `05_cnn_modelling.py --out-root $OUT` (`:34-47`), with three example submit lines in comments (`:53-58`).

**C. Homology cluster map + fold-difficulty audit [35ac9ca, dc8cafb]**
- `src/investigation/29_build_cluster_map.py` (new, 152 lines) — imports `cluster_by_hamming` from `01_datasplit` via `importlib.import_module` (`:87`, module name starts with a digit so stdlib importlib is required) rather than copying the clustering logic; refuses to build if any peptide spans folds (`:81-85`), reports homology clusters spanning folds (`:104-115`), writes `data/processed/cluster_map_peptide.csv` (`:146`).
- `src/investigation/02_datasplit_analysis.py:13-17` — bootstrap now inserts `src/pipeline/python`, fixing the previously documented `PYTHONPATH` workaround; `:33` output path constant; `:376-404` `load_cluster_map()` (skips with instructions if the map is missing); `:407-419` `_fold_stats_from_assignment()`; `:423-534` `report_fold_difficulty_audit()` — 1000-replicate cluster-preserving permutation null, seed 42, OUTLIER flags, writes `results/tables/fold_audit_cluster.csv`; `:599-601` wired into `__main__`.

**D. Deletions (−2,766 lines) [dc8cafb]**
- `src/investigation/embed_test_esmc.py` (83), `embed_test_esmc_pad.py` (146), `embed_test_esmif.py` (96), `embed_test_esmif_pad.py` (202), `inspect_embeddings.py` (630), `inspect_window_embeddings.py` (580), `src/pipeline/python/01b_datasplit_protein_level.py` (911). No remaining source reference to any of them except documentation (see ## Code Review 3).

**E. Small items [dc8cafb]**
- `src/investigation/10_dim_variance.py:166-172` — local `cum` renamed to `cumul` (3 lines, no logic change).
- `src/investigation/README.md` — script count 27 → 28 (`:5`); `02_datasplit_analysis.py` section rewritten and its ⚠ import-bug tag removed (`:68-74`); `29_build_cluster_map.py` section added (`:75-81`); config-import gotcha and cross-cutting gotcha updated to match the fixed bootstrap (`:13`, `:267`); run table gains `29` and drops the workaround for `02` (`:284-285`).
- `runs/pool_xgb/pool_naive_auc.png` — binary changed (114,971 → 87,906 bytes) with no change to its producer `src/investigation/22_pool_barplot.py` (last touched 39659a3, 2026-10-29) anywhere in the range.

## Experiments / Results

Repository-supported results only. The range changed no tracked numeric table (0 tracked `cv_results_*.json`, 0 tracked files under `data/`), so the ESM-IF/ESM-C variance-ladder numbers, the DeLong verdicts and the configured default rung reported previously are unchanged at 35ac9ca. New in this range:

- **Fold-difficulty audit verdict (README-quoted, file not tracked)** — `src/investigation/README.md:73`: "First run 2026-11-05: all CV-fold stats inside null 95% — no fold stands out on size/composition." Observation quoted from the README; `results/tables/fold_audit_cluster.csv` has 0 tracked copies, so the underlying rows are not in the repository.
- **Cluster-map build stats (README-quoted, file not tracked)** — `src/investigation/README.md:80`: "2026-11-05 build: 44,504 peptides → 43,164 clusters (97.4% singletons, largest 14 peptides), zero cross-fold clusters — saved split is homology-clean." `data/processed/cluster_map_peptide.csv` has 0 tracked copies.
- **No kernel-sweep results exist in the repository** — `runs/cnn_kernel/` and every `runs/cnn_*` path are gitignored (`.gitignore:149`), so the k5/k7/k9 sweep has produced no citable artifact yet; the script itself is the only tracked output.
- **Previously reported discrepancy still present** — `runs/ev_matched_topXvariance_sweep/results/variance_sweep_vs_best_delong.tsv:12-21` still reads `esmif_eosrepeat` with best ev85 = 0.648827 (`:21` `best_vs_raw` 0.648827 vs 0.649814) while the recomputed raw table says 0.648705.

## Code Review — Shortcomings & Issues

New findings first (redundancy / over-engineering / reinvented-stdlib priority), then carried items re-verified at 35ac9ca.

1. **Reported CNN validation AUC is selected on the same bucket it is scored on (new, risk)** — `src/pipeline/python/05_cnn_modelling.py:135` early-stops on val AUC with `--patience`, and `:296-298` scores `compute_metrics(y_va, probs)` on that same bucket's best-epoch output. In the deleted nested scheme the outer bucket was never used for selection, so `cnn_cv` numbers are not the same quantity as `cnn_nested_cv` numbers and are optimistic by up to one early-stopping choice. This is an observation of the diff, not a verdict — question for the authors: is simple CV intended as an experiment only, and which scheme will be quoted in the thesis? [dc8cafb]
2. **Two CV schemes can be mixed silently in one comparison figure (new, risk)** — `src/investigation/28_cnn_compare_barplot.py:118` docstring "Latest file wins per key" and `:133-141` pick by `(model_key, features_key)` + timestamp only; `mode` (`cnn_nested_cv` vs `cnn_cv`, `05_cnn_modelling.py:344`) is read but never used, and both JSON generations share `cv_summary.auc_roc`. The docstring at `:25` still says "(nested CNN, cv_summary.auc_roc)". Lazy fix: key or filter on the `mode` field the loader already parses. [dc8cafb]
3. **README still documents six deleted scripts as current (new, maintainability)** — `src/investigation/README.md:169,176,183,190,201,208` are full sections for `embed_test_esmc{,_pad}.py`, `embed_test_esmif{,_pad}.py`, `inspect_window_embeddings.py` and `inspect_embeddings.py`; `:270` states "current generation is `inspect_window_embeddings.py` + the 8 window H5s" (that file is deleted); the relationship map `:233-242` and the run table `:293-295` reference them too; and the count at `:5` went 27 → 28 in the commit that deleted 6 and added 1. `src/README.md:12` still lists the deleted `01b_datasplit_protein_level.py`. [dc8cafb]
4. **Dual-format plotter built for CSVs that do not exist in the repo (new, maintainability / YAGNI)** — `src/pipeline/python/05_plotting.py:41-65,92-127` plus `outer_mean` (`:68`) keep a nested-CSV branch alive; there are **0 tracked** `learning_curves` CSVs of either format and `runs/cnn_*/` is gitignored (`.gitignore:149`), so nothing in the repository exercises it. Dead line `:96` `_ = max_ep  # used per-bucket below`. Ceiling: delete the nested branch (and `outer_mean`) once no nested CSV needs re-plotting; keep it only while server-side nested runs still get re-rendered.
5. **Kernel registry is three copy-pasted dicts (new, style)** — `src/pipeline/python/05_cnn_models.py:20-31`: `cnn_simple_k5/k7/k9` differ only in `kernel`; `{f"cnn_simple_k{k}": {...} for k in (5,7,9)}` is one line. Explicit entries are readable, so this is a naming-only note. [dc8cafb]
6. **Dead condition in the new audit (new, style)** — `src/investigation/02_datasplit_analysis.py:438` `[f for f in range(N_CV_FOLDS) if f != HELD_OUT_INDEX]`: `HELD_OUT_INDEX = N_CV_FOLDS` (`config.py:561-562`), so the filter can never exclude anything. Harmless, but it implies a configuration that does not exist. [dc8cafb]
7. **SLURM log path does not follow the run override (new, risk, low)** — `src/run/run_cnn_kernel_slurm.sh:8-9` hardcode `runs/cnn_kernel/logs/slurm/%j.{out,err}` while `RUN` (`:34`) redirects every other output; submitting `RUN=runs/cnn_kernel_20261106` still writes scheduler logs into `runs/cnn_kernel/logs/slurm/`, and the submit comment `:53-58` tells you to `mkdir $OUT/logs/slurm` — the directory that must pre-exist for `sbatch` is the hardcoded one. Since `runs/cnn_*/` is gitignored, a fresh clone has neither. [35ac9ca]
8. **The purge stopped at files, not documentation or config (carried, re-verified)** — `28_cnn_compare_barplot.py:94,105,238` still name `emb_esmif_eosrepeat`; `src/investigation/README.md` still references the nonexistent `10_pad_subset_auc.py` on 7 lines (`:13,34,147,149,249,263,291` — 0 files of that name exist); 27 tracked files still mention `eosrepeat` (most are historical result matrices that correctly describe what ran); `runs/varfold_pilot/` still tracks the eosrepeat varfold record. [fd3a515, carried]
9. **Helper duplication unchanged (carried, re-counted at tip)** — `def star` in 10 files, `def delong_test` in 6, `def holm_adjust` in 5, `spec_from_file_location` in 8, identical to the previous report; the new `29_build_cluster_map.py` adds no copy (it imports `cluster_by_hamming` — the right ladder rung), so the debt is exactly where it was. Severity: maintainability (over-engineering by duplication). [fd3a515, carried]
10. **Result-table and figure inconsistencies unchanged (carried, re-verified)** — the `variance_sweep_vs_best_delong.tsv` / `esmif_eosrepeat` mismatch (see ## Experiments); no `dim_variance_esmif_eos_bos_repeat.csv` base ranking file (only `_ev30..ev95`/`_top21` JSONs); job-tag collision hardcoded across `20_dim21_barplot.py:90,93`, `21_dim21_delong.py:53-54`, `24_varfold_delong.py:65`, `25_varfold_barplot.py:44`; split y-axis conventions — "AUC-ROC (nested CV, mean +/- std)" in `20:205`, `22:172`, `25:140`, `26:226`, `28:165` vs "CV-AUC" in `27_variance_optimum.py:215` and `pca_sweep_AUC_delong_comparison.py:480,523`; `runs/esmif_eos_sweep/plots/pca_sweep_*.png` still last changed 2026-10-23 vs script 2026-11-02; dead `load_delong_pick` (`22_pool_barplot.py:100`, the copy at `20_dim21_barplot.py:107` is used) and the unreachable `esmif_` arm in `23_pool_delong.py:97` (`ARMS` at `:51-56` defines only `esmc_*` keys); 4 zero-byte tracked `.txt` logs, `.gitignore` `runs/*/models/` triplicate (`:125,128,160` plus `runs/**/models/` at `:150`) and `runs/padmode_choice/results/paired_tests_OLD.csv`.
11. **What passes the ladder (carried)** — net deletion of 2,766 lines with no new abstraction; `29_build_cluster_map.py` reuses `01_datasplit.cluster_by_hamming` instead of re-implementing Hamming clustering; `02_datasplit_analysis.py` reuses `config` symbols and its own existing report scaffolding rather than adding a second analysis framework; the audit's null is plain numpy (`bincount`, `default_rng`), no dependency added.

## Potential Issues

1. **Which CNN CV scheme is citable? (new, blocking for citation)** — `cnn_nested_cv` outputs and `cnn_cv` outputs have the same `model_key`/`features_key`/`cv_summary` shape, different `mode`, and are produced by the same script before/after dc8cafb. No tracked `cv_results_*.json` exists to compare, so git cannot show what changed numerically. Authors' call: is the simple 6-fold scheme an experiment to be superseded, or the new reported scheme — and do previously quoted CNN numbers refer to nested CV? (See ## Code Review 1-2.)
2. **New audit artifacts are README-only (new)** — `data/processed/cluster_map_peptide.csv` and `results/tables/fold_audit_cluster.csv` are both untracked, yet the README quotes their numbers as results. Same provenance class as the long-standing "0 tracked `cv_results_*.json` / 0 tracked `data/`" issue.
3. **Which dataset construction is authoritative? (carried, blocking)** — the R re-sampled variant writes `data/processed/df_combined_pos_and_neg_remove715negonly.csv` (`04_evaluate_netmhcpan_sensitivity.R:805-807`) that nothing consumes, while `run_pipeline.R:37` / `05_protein_features.R:13` hardcode the base CSV. Authors' call: is `make_variant_dataset.py` the definition of record?
4. **Is the eos_bos EV ladder recomputed or relabelled? (carried)** — all 10 renamed JSONs differ from their eosrepeat predecessors by one content line (`embedding_key`) while the variance arms themselves differ; git cannot decide it, and the base ranking CSV was deleted without an eos_bos replacement.
5. **Local checkout is stale (process, carried, re-verified)** — local `hilarius` at 26ec48d (2026-10-02), now **149** commits behind `origin/hilarius`, working tree clean (0 modified files). Both 2026-11-05 commits, the CNN rewrite and the cluster-map work are absent from the working tree; anything edited locally targets content the remote no longer has.
6. **Carried open questions, none answered** — variant folds regenerated rather than derived (`01_datasplit.py`/`01b`, the latter now deleted, so the protein-level split path has no script at all); ESM-IF pooling arms dropped from the 22/23 investigations; no ESM-C 6B embeddings in git; structure-audit CA-only vs complete-backbone residue semantics; the two restyled netmhcpan PNGs (`results/figures/netmhcpan/confusion_matrix_9mer.png`, `results/figures/netmhcpan/affinity_bias_density_after.png`) still last changed 2026-09-14 vs script 2026-10-30.

## GitHub

From the collected GitHub data (used verbatim, not re-run): no open PRs (`gh pr list` → []), no open issues (`gh issue list` → []), no workflow runs (`gh run list` → []); repository `TheHilarius/thesis2026`, default branch `hilarius`. From git: origin/hilarius = **35ac9ca** (2026-11-05 14:14:35 +0100), which moved during this report's window (previous anchor 76735ad, 2026-11-02 13:39:56 +0100). Local branch `hilarius` is 149 commits behind with a clean working tree. `reports/overnight/` is gitignored (`.gitignore:119`), so this report exists only in the local working tree.

## Recommended Next Steps

Carried items are marked; one new item addresses the day's methodology change. Prioritized, maximum five:

1. **Decide the variant's construction of record (carried, blocking)**: choose the Python filter or the R restricted-pool re-sampling, delete or wire the other (`04_evaluate_netmhcpan_sensitivity.R:805-807` vs `run_pipeline.R:37` / `05_protein_features.R:13`), record the variant's row/label/protein counts somewhere citable, and state whether variant folds are row-matched to base.
2. **Reconcile the ev-sweep result table trio (carried, top results-integrity item)**: recompute or drop `runs/ev_matched_topXvariance_sweep/results/variance_sweep_vs_best_delong.tsv:12-21` so the directory has one ESM-IF truth, and answer whether `results/tables/dim_variance_esmif_eos_bos_repeat_ev*.json` were recomputed or relabelled (add the missing `dim_variance_esmif_eos_bos_repeat.csv` either way).
3. **Settle the CNN CV scheme (new)**: state in `src/pipeline/python/05_cnn_modelling.py` docstring + README whether `cnn_cv` or `cnn_nested_cv` is the reported scheme, and make `28_cnn_compare_barplot.py:133-141` key on the `mode` field it already parses so old and new JSONs cannot mix in one figure.
4. **Docs and purge tail (carried, now larger)**: delete the six dead script sections (`src/investigation/README.md:169-212`), fix the "current generation" pointer at `:270` and the relationship map `:233-242`, correct the count at `:5` and `src/README.md:12`, drop the 7 `10_pad_subset_auc.py` references and `emb_esmif_eosrepeat` from `28_cnn_compare_barplot.py:94,105,238`, then the dead `load_delong_pick` (`22_pool_barplot.py:100`) and `esmif_` predicate arm (`23_pool_delong.py:97`).
5. **Provenance + helper consolidation (carried, merged)**: `git add` the new `data/processed/cluster_map_peptide.csv` and `results/tables/fold_audit_cluster.csv` (or state the server is the record), plus the tracked-worthy `runs/dim21_eosbos/results|plots` and `runs/varfold_pilot_eosbos/results|plots`, and land one `src/investigation/delong_stats.py` (~40 lines: `delong_test`, `holm_adjust`, `star`, `load_oof`) replacing the 6 + 5 + 10 copies and 8 `spec_from_file_location` sites.

Resolved since the previous report: the `02_datasplit_analysis.py` import-bootstrap workaround (fixed and README-updated at `:13`, `:267`) and the README's missing index for the newest investigation scripts (`29_build_cluster_map.py` now documented). Everything else carried remains open at 35ac9ca.

## Reviewed Through

35ac9ca
