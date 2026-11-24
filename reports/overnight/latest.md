# Thesis Overnight — 2026-11-24

## Summary

Range bb12e52..origin/hilarius now contains **3 commits, all 2026-11-23** (previous report's range was 0): the HLA-A*01:01 external-validation support moved out of a deleted `src/pipeline_a0101/` fork and into the main pipeline as an env switch (`THESIS_ALLELE`), with a new external-scoring script, a netMHCpan sbatch receipt wrapper, an updated run recipe, and a slurm script rename. The fork deletion removes ~4,365 lines of a duplicated second pipeline. No new numeric campaign results are tracked; the only new tracked result artifact is a differential-logo p-value table plus two PNGs.

## Past 7 Days

Calendar window 2026-11-18..2026-11-24: 18 commits. Commits per day (count then date):

     10 2026-11-18
      4 2026-11-19
      1 2026-11-22
      3 2026-11-23

0 commits on 2026-11-20, 11-21 and 11-24; the most recent day with activity is 2026-11-23 (counts re-derived from `git log --since/--until` over `origin/hilarius`; 18 total, matching the stated table). Theme level only — this section intentionally overlaps earlier reports:

- **Window-cutoff and split decision with evidence (11-18)** — `WINDOW_CUTOFF` → 6 with mechanistic rationale, the modelling default split moved onto the h6 file, the dedup A/B on a fixed split with paired DeLong, a duplicate-inclusive flank5 sensitivity sweep, fold-audit artifacts relocated under `runs/r19_rmdup/`, and logo-renderer fixes (flank-length-agnostic features, ggseqlogo overflow, orientation).
- **NetMHCpan bias reporting promoted to effect sizes (11-18)** — CLES headline plus delta-based equivalence for rank bias.
- **New encodings and the modelling-comparison campaign (11-19 → 11-22)** — BLOSUM45/BLOSUM90 added end to end with the follow-up fix for component gates that had silently dropped their columns, the Optuna TPE tuning campaign (tuner + sbatch + plot script), the XGB-vs-rf/LR model-comparison script, and the 11-22 hardening commit (per-arm `--features`, study-creation race retry, venv pip fix, `--smoke` trial-count fix).
- **HLA-A*01:01 external-validation allele wired into the main pipeline (11-23)** — fork deleted and superseded by a `THESIS_ALLELE` path switch, a new no-retraining external scoring script, per-allele arguments on the fetch/NSP3 tools, a netMHCpan sbatch wrapper with receipts, run-recipe and spec-doc updates, and the nsp3 slurm script rename.

## Changes

Only `bb12e52..origin/hilarius` = 3 commits (`902cfa2`, `d47275a`, `b10b74a`, all 2026-11-23); nothing already covered by previous reports is repeated.

**Allele switch in the shared config (`d47275a`)**

- `src/pipeline/python/config.py:49-54` — new `THESIS_ALLELE` env (`hla0201` default, unknown value raises), and for `hla_a0101` the env var `THESIS_VARIANT` is forced empty so the external allele never composes with dataset-variant suffixes.
- `src/pipeline/python/config.py:57-62` — `_allele()` prefixes per-dataset paths with the allele dir; applied to `RAW_DATA_PATH`/`SPLIT_DATA_PATH` (`:86-91`), the nine encoded/compositional CSVs (`:95-104`), `EMBEDDING_DIR`/`PREPARED_EMBEDDING_DIR` (`:109-110`). For `hla_a0101`, `SPLIT_DATA_PATH` aliases `RAW_DATA_PATH` (`:89-91`, no CV folds); `ALLELE_DATA_DIR` (`:78`) and `FIGURES_DIR` → `results/figures/models_hla_a0101/` (`:82-84`). `CONFIG_EDITION` marker at `:1262`; `validate_config` prints the edition at `:1359-1360`.
- `src/pipeline/python/03_prepare_embeddings.py:32, :687-688` — allele-tagged log filenames; `:731-735` adds a dummy `fold=0` column when the split file has none (schema compat for the fold-less external set).
- `src/pipeline/python/04_modelling.py:2111-2118` — comment update pointing the fold-artifact replay at the new script instead of the deleted fork path.

**New external scoring script (`d47275a`)**

- `src/pipeline/python/14_predict_external.py` (new, 370 lines): refuses to run unless `THESIS_ALLELE` is the external allele (`:69-75`); loads `04_modelling.py` through `spec_from_file_location` (`:80-84`); `discover_models` requires a matching `_artifacts.pkl` per model and raises with a regeneration message (`:122-157`); `prepare_external` (`:159-190`) rebuilds the feature matrix per fold's artifacts; `main` (`:199+`) averages inner-model probabilities and writes `predictions/<run_tag>_external_predictions.csv` + `_metrics.json` under `data/processed/<allele>/predictions/`.

**Fork deleted (`d47275a`)**

- `src/pipeline_a0101/` removed in full — 13 files, ~4,365 lines deleted (its README, fetch scripts, netMHCpan runners, Python `03`/`config`, R `02`/`04`/`05`/`06`/`07`/`08`/`functions.R`/`run_pipeline.R`).
- `src/pipeline/r/02_load_iedb_data.R:106-111` — the next-step hint for non-legacy alleles now prints a `fetch_uniprot_fastas.py` command instead of pointing at the deleted fork script.
- `src/fetch/prepare_fetch_list.sh:2-15, :93-104` and `src/fetch/fetch_structures.sh:1-55` — optional `list_tag`/`allele` arguments so per-allele fetch lists don't overwrite the shared ones (PDBs/logs stay shared).

**NSP3 / NetSurfP tools take an allele argument (`d47275a`)**

- `src/tools/netsurfp/nsp3_clean_fasta.sh:6-48`, `prepare_nsp3_batches.sh:1-58`, `run_netsurfp3.sh:4-30` — `[hla0201|hla_a0101]` (and for the runner, `batch_dir`/`output_base`), legacy defaults when omitted, input-existence guards, and the batch glob widened to `*batch_*.fasta` so per-allele prefixes resolve.

**netMHCpan sbatch receipt wrapper (`b10b74a`)**

- `src/run/run_netmhcpan_a0101_slurm.sh` (new, 91 lines): SBATCH directives (`:3-8`), netMHCpan PATH resolution with a fatal message if absent (`:40-49`), receipt header/footer (`:54-64`, `:70-78`), runner invocation (`:66-68`), and an output-count cross-check that exits 1 when csv/txt counts fall short of the batch count (`:73-91`) — the runner itself exits 0 on partial failure.
- `docs/hla_a0101_run_recipe.md:50-68` — step 4 rewritten around the wrapper, documenting receipts, resume behaviour and the 4.2e-vs-4.2b binary-version note.

**Rename (`902cfa2`)**

- `src/run/run_h100.sh` → `src/run/run_nsp3.sh`, pure rename (0 line changes).

**Docs and tracked figures (`d47275a`)**

- `docs/python_allele_support_plan.md:3-24` — gate replaced by "IMPLEMENTED", listing the three deviations from the original spec (variant forced off, split alias, log/figure routing) and a config-parity check.
- `src/README.md:46-57` — run order now carries `[hla0201|hla_a0101]` arguments and links the full recipe.
- `docs/hla_a0101_run_recipe.md` (+151 lines) — Stage A/B/C commands, the artifacts requirement, and gotchas; all 11 script paths referenced by it exist at the tip (checked with `git cat-file -e`).
- New tracked figures: `results/figures/logo/flank10_pre_remove/{differential_logo_29mer_difflogo.png, differential_logo_29mer_difflogo_js.png, difflogo_pvalues.tsv}` — output of `src/investigation/13_difflogo_significance.R` run with that script's own documented example output dir.

## Experiments / Results

Only repository-supported results; nothing was re-executed this run (no `Rscript` on this host; script execution blocked in unattended mode).

- **New tracked result artifact:** `results/figures/logo/flank10_pre_remove/difflogo_pvalues.tsv` — 30 lines (header + 29 positions) with columns `position, n_obs, js_divergence, max_abs_delta, p_raw, p_BH, significant_05`; `n_obs` runs 45,927–46,363; e.g. N4 `p_BH=0`, N5 `p_BH≈9.7e-6`, N8 `p_BH=0.52` (FALSE). The producing script's own header (`13_difflogo_significance.R:16-19`) states that with n≈20k+ the permutation null is sampling noise only and that the informative numbers are the effect-size columns, not the asterisks. The comparison encoded by the two logo groups is not documented in the tracked file itself.
- **No tracked a0101 results** — `.gitignore:19-20` ignores `data/processed/hla_*/`, so netMHCpan csvs, prepared embeddings and `predictions/*_metrics.json` for the external allele cannot appear in the repository.
- **The two earlier campaigns still produce nothing the repository can show (re-verified at the tip):** `runs/xgb_optuna` → 0 tracked files, `runs/modelcompare_29mer_46617` → 0 tracked files, `runs/is_encodings_helping_v2/results/..._delong_stats.tsv` → 0 `blosum45` rows. No AUC, DeLong table or tuned `best_params` exists in the tree.
- **Artifacts requirement is unmet in-repo:** `git ls-tree -r origin/hilarius | grep -c '_artifacts.pkl'` → **0**, and `/models/` is gitignored (`.gitignore:75`) — so the fold artifacts `14_predict_external.py` hard-requires exist only on the machine that trained the models.

## Code Review — Shortcomings & Issues

Severity legend: bug | risk | maintainability | style. New findings first (ponytail order: redundant / over-engineered / reinvented existing helpers first), then carried findings re-verified still open at `b10b74a`.

**New in this range**

1. **`prepare_external` re-implements `prepare_validation` from the module it already imports (maintainability — reinvented existing helper)** — `14_predict_external.py:159-190` hand-builds the CSV block, embedding block, concat and scaler step; `04_modelling.py:1207-1247` `prepare_validation()` does exactly that (it is imported as `M` 80 lines above). With the dummy `fold=0` column the same commit added (`03_prepare_embeddings.py:731-735`), `M.prepare_validation(df, csv_feature_cols, emb_data_dict, model_cfg, artifacts, 0)` selects all rows and returns the same `X, y, indices` — one assignment plus one call replaces ~30 lines that will otherwise drift from the training-side pipeline they must stay identical to. Cites: [d47275a].
2. **SBATCH output directory is created after the job starts (risk — submission failure)** — `run_netmhcpan_a0101_slurm.sh:7-8` writes to `logs/netmhcpan/a0101_nmpan_%j.out`, but `mkdir -p logs/netmhcpan` is at `:52`, i.e. inside the job; `logs/netmhcpan/` is untracked (`git ls-tree` → empty) and not gitignored, and `docs/hla_a0101_run_recipe.md:56` submits without a `mkdir`. Slurm cannot open the output file if the directory does not exist, so on a fresh checkout the job fails before the wrapper runs. One `mkdir -p logs/netmhcpan` before the `sbatch` line fixes it. Cites: [b10b74a].
3. **Legacy NSP3 default inputs changed while the comments say "legacy call preserved" (risk + doc drift — authors' decision needed)** — `nsp3_clean_fasta.sh:21-22` now defaults to `combined_positives_only.fasta` (was `combined_8_to_14mer.fasta` with a correspondingly different output name) and `prepare_nsp3_batches.sh:31` now defaults to `data/raw/fasta/combined_positives_only_nsp3_clean.fasta` (was `data/processed/nsp3_input_padded.fasta`, a file no tracked script produces or reads any more), yet both headers still claim the no-arg call is byte-identical (`nsp3_clean_fasta.sh:6-7`, `prepare_nsp3_batches.sh:5-7`). `src/pipeline/r/run_pipeline.R:46` still documents step 05's `IN: data/raw/fasta/combined_8_to_14mer.fasta`. The new wiring is self-consistent (clean → batches → `run_netsurfp3.sh`), and the old chain was not, but which FASTA is the input of record is a scientific-provenance question for the authors, not a code question. Cites: [d47275a].
4. **The dummy-fold fallback is allele-agnostic and masks a broken legacy split (risk, low)** — `03_prepare_embeddings.py:731-735` runs for any `THESIS_ALLELE`; a legacy split file missing `FOLD_COL` now silently becomes one bucket of fold 0 instead of failing at `sorted(df[FOLD_COL].unique())`. Guarding it with `ALLELE == "hla_a0101"` keeps the safety check. Cites: [d47275a].
5. **Partial rename (style)** — `run_nsp3.sh:6-7` still emits `nsp3_h100_%j.out/.err` and `:22` still says "Running NetSurfP-3.0 on H100..." after the script was renamed for clarity; the rename is the whole content of `902cfa2`. Cites: [902cfa2].
6. **Stale reference to the deleted fork (style)** — `.gitignore:19` comment still reads "(src/pipeline_a0101)" in the same commit that deleted that directory; the only other remaining mentions are the recipe's explicit "superseded" note (`docs/hla_a0101_run_recipe.md:9`). Cites: [d47275a].
7. **One commit carries a 4,365-line deletion, a new 370-line script and 1.6 MB of unrelated logo figures under "add: More changes to script to acompany new 0101 allele" (style — commit message accuracy)** — `git show --stat d47275a` → 30 files, +518/−4,365, including `results/figures/logo/flank10_pre_remove/*.png` (941 KB + 672 KB) which have nothing to do with the allele work. With no PRs or issues, `git log` is the only index a reader has. Cites: [d47275a].
8. **New script ships with no check (maintainability)** — `14_predict_external.py` is 370 lines of feature-matrix replay where the one thing that must be right is "identical to training"; the repo tracks **1** `test_*.py` overall, and none covers it. The cheapest honest check is a smoke run asserting the built feature count equals `{run_tag}_feature_names.json` (the comparison already exists at `14_predict_external.py:305-311` but only fires when the file happens to be present). Cites: [d47275a].
9. **Latent config inconsistency for the external edition (risk, low — no current reader)** — `config.py:690-691` derives `DEFAULT_SPLIT_PATH` from `SPLIT_DATA_PATH.with_name(f"{stem}_updated_h{WINDOW_CUTOFF}_pepcut1.csv")`; with `SPLIT_DATA_PATH` aliased to `df_all.csv` for `hla_a0101` (`:89-91`), that resolves to a folds-named file that cannot exist for an allele with no folds. `04_modelling.py` and `46_xgb_optuna_tune.py` are the only consumers, and Stage C scripts read `RAW_DATA_PATH`/`SPLIT_DATA_PATH` instead, so nothing breaks today — the one-line fix is `DEFAULT_SPLIT_PATH = SPLIT_DATA_PATH` in the external branch. Cites: [d47275a].

**Carried, re-verified still open at `b10b74a`**

10. **Arm-path mapping written down three times (maintainability)** — `46_xgb_optuna_tune.py:72-83` `arm_paths`, `47_xgb_optuna_plot.py:59-64` `arm_dir` (docstring says "Mirror 46_xgb_optuna_tune.arm_paths"), `run_xgb_optuna_slurm.sh:47-51` `blosum50` special case; one helper in `config.py` deletes two copies. Cites: [bb12e52].
11. **`delong_table` re-implements `holm_family` (maintainability)** — `46_model_compare_29mer.py:104-111` calls `d18.holm_adjust` and zips `p_adj` by hand while `18_encodings_sweep_delong.py:123-127` `holm_family(pairs, key)` does exactly that (and `d18.stars` is used one line below). Cites: [8147f53].
12. **21st `spec_from_file_location` bootstrap (maintainability)** — `git grep -l spec_from_file_location origin/hilarius -- src/` → **21** files (down from 22 only because the fork was deleted); `14_predict_external.py:80` is the new one. Python's native import system, or one shared `_bootstrap.py`, does this once. Cites: [d47275a].
13. **Encoding-name registry hand-copied across six files (maintainability — already caused a silent-corruption bug; a new instance found this range)** — hardcoded lists still at `04_modelling.py:324-325`, `extract_model_metrics.py:44-45`, `config.py:1153-1157` (`_AA_ENCODINGS`), `18_encodings_sweep_delong.py:205`, `08_aa_encoding.R:58-62`, `functions.R:1515`; `d4460e2` exists only because two Python copies were missed. New instance: the "cannot combine multiple AA encodings" gate at `config.py:1332-1335` lists `("sparse", "blosum50", "blosum62", "blosum80", "kidera", "atchley", "compositional")` and omits `blosum45`/`blosum90`, so that check silently does not apply to the two encodings added in `1da5d6a` (and the retirement message at `config.py:1269, :1296-1299` still says "explicit: blosum50 / blosum62 / blosum80"). Derive from `FEATURE_COMPONENTS`, or add the one `assert`. Cites: [1da5d6a, d4460e2].
14. **`THESIS_SPLIT` pin is a no-op (risk — a safety check that checks nothing)** — `46_model_compare_29mer.py:38-40` sets it, but `config.py:690-691` routes it into `DEFAULT_SPLIT_PATH` and this script reads `SPLIT_DATA_PATH` (`:200-201`); only `THESIS_VARIANT=none` (`:37`) pins anything, and the 46,617-row assert (`:202-204`) is the only real guard. Cites: [8147f53].
15. **Contradictory sweep footnote baked into a tracked figure (bug)** — `42_window_cutoff_sweep_plot.py:159-160` still emits "chosen setting c=5 (last point before any downward drift)" in the same caption as "Chosen setting c=6 …"; the tracked sweep PNG still contains it. One-line deletion + regenerate. Cites: [carried].
16. **Concurrent unpinned pip installs into one venv (risk)** — `run_xgb_optuna_slurm.sh:88` `|| pip install optuna` in jobs that start together; `optuna` still has 0 matches in `requirements*.txt`. Install once in `--submit` mode, workers `python -c "import optuna" || exit 1`, plus one pin. Cites: [bb12e52].
17. **Polish winner record: falsy-zero bug (bug, low)** — `46_xgb_optuna_tune.py:266-268` still uses `user_attrs.get("auc_folds") and float(np.std(...)) or None`, so a std of exactly `0.0` becomes `None`, and `fixed` still omits `auc_folds`/`logloss_folds`. Two plain assignments fix both. Cites: [b1dd369].
18. **`18_encodings_sweep_delong.py` cannot be re-run under default config (risk)** — `:477` reads `SPLIT_DATA_PATH` (45,278 rows under default `flank5_rmdup`) while the tracked campaign is 46,617 rows, and `:500-502` require the blosum45/90 arms → `Missing arms`. Which split of record should this script read? Cites: [1da5d6a].
19. **Doc drift (style, re-verified)** — `src/investigation/README.md:5` says "33 scripts (6 R, 27 Python)" while the tip tracks **74** (65 Python, 9 R) and its index table still has no rows for `42_`–`47_`; `src/README.md:12` still lists `01b_datasplit_protein_level.py` (absent from the tree); `04_modelling.py:1457-1461` `--split` help says "Default: config SPLIT_DATA_PATH" while the code resolves `DEFAULT_SPLIT_PATH`; `umap-learn` still absent from `requirements*.txt`; DiffLogo still absent from `renv.lock` although `13_difflogo_significance.R` requires it.
20. **Style leftovers (low, re-verified)** — hand-rolled TSV writing in `46_model_compare_29mer.py:176-197` with pandas already imported; `assert` for runtime preconditions at `46_xgb_optuna_tune.py:110, :247` (stripped under `-O`; `46_model_compare_29mer.py:202-204` does the same check with `sys.exit`); dead/tautological guards at `45_dedup_compare_plot.py:100-103`, `30_plot_fold_audit.py:41`, `30_ae_barplot.py:110-116`, `run_dimselect_sweep.sh:73,81,93`; unused `csv_cols` unpacks and `fold_vec` param in `46_model_compare_29mer.py`.
21. **What passes the ladder (good calls worth naming)** — the fork was deleted rather than maintained (the largest single win in this range); the allele switch reuses the existing `THESIS_VARIANT` guard pattern instead of introducing a config framework; the fetch/NSP3 tools gained optional arguments with legacy defaults instead of per-allele copies of whole scripts; the sbatch wrapper is plain bash with a count cross-check and no monitoring dependency; `47_xgb_optuna_plot.py` still uses Optuna's native fANOVA; the tuning refactor parameterises rather than abstracts.

## Potential Issues

1. **Neither the a0101 run nor the two earlier campaigns leave anything the repository can show (data/process, carried + new)** — `.gitignore:19-20` excludes `data/processed/hla_*/`, and `runs/xgb_optuna/`, `runs/modelcompare_29mer_46617/`, `runs/encodings_sweep`'s `cv_results_*.json` remain 0 tracked files with 0 `blosum45` rows in the DeLong stats. Question for the authors: are these outputs intended to be tracked (like `runs/r19_rmdup/`), or are the scripts cluster-only tools? Unchanged for a third consecutive report, now with a third invisible pipeline.
2. **Stage C of the recipe is blocked on retraining everything (process, new)** — the recipe's "Artifacts requirement" section says models trained before the artifacts change must be re-run with identical settings before `14_predict_external.py` can score them, and 0 `*_artifacts.pkl` are tracked while `/models/` is gitignored. Question for the authors: is that retraining run part of the intended scope, and where do its receipts live?
3. **NSP3 input of record (authors' question, new)** — see finding 3: two legacy defaults changed under "legacy call preserved" comments and `run_pipeline.R:46` still documents the old FASTA. The code now forms a coherent chain, but the provenance of the protein FASTA behind step 05/06 features should be stated deliberately, not implied by a silent default change.
4. **Selection-on-the-test-set question for the tuning campaign (authors' question, carried)** — `46_xgb_optuna_tune.py`'s objective is mean AUC over the 6 outer buckets "identical to the nested-CV outer evaluation", and `47_xgb_optuna_plot.py` names a same-split refit + DeLong as follow-up: selecting the best of ~100 trials on the fold partition that produces the reported CV number, then DeLong-testing that tuned refit on the same rows, is a question about what the reported figure estimates. Not decided here.
5. **The silent-corruption window is ~9 hours wide (observation — cluster check needed, carried)** — `1da5d6a` (14:27) introduced the gating omission, `d4460e2` (23:04) fixed it. No tracked artifact is affected, but any vardims blosum45/90 run completed in that window was trained embedding-only under the wrong label and should be discarded rather than compared.
6. **Two parallel writers on one SQLite study (process, known ceiling, carried)** — `run_xgb_optuna_slurm.sh` still fans two tune jobs onto one `study.db` per arm; the retry at `46_xgb_optuna_tune.py:163-188` covers only *schema creation*, not `database is locked` during `study.optimize`. Ceiling unchanged; upgrade path = one writer per study or PostgreSQL. Not needed until it errors.
7. **Git history as the only index (process, carried, now worse)** — no PRs, no issues, no CI, so `git log --oneline` is the whole index: a 30-file, +518/−4,365 commit is titled "add: More changes to script to acompany new 0101 allele", and a pure rename is its own commit. Commit messages are where a reader looks first.
8. **Verification limits (process)** — local branch `hilarius` at 26ec48d is **231** behind origin with an empty `git status --porcelain`, so every cite comes from `git show`/`git grep`/`git ls-tree` against the tip; `command -v Rscript` → none; script execution is blocked in unattended mode. `reports/overnight/` is gitignored (`.gitignore:134-135`, "published only to the `reports` branch"), so this report exists only in the local working tree. The 90 MB raw-data blob remains in history untouched.

## GitHub

From the collected GitHub data (used verbatim, not re-run): no open PRs (`gh pr list` → []), no open issues (`gh issue list` → []), no workflow runs (`gh run list` → []); repository `TheHilarius/thesis2026`, default branch `hilarius`. From git: origin/hilarius = **b10b74a**, i.e. **3 commits ahead of the previous report's anchor** (`git rev-list --count bb12e52..origin/hilarius` → 3), confirmed against the remote this run via `git ls-remote origin refs/heads/hilarius` → `b10b74a99f22596cccf3aca19f95afe1b9903b57` (read-only). Local branch `hilarius` at 26ec48d remains **231** behind with a clean working tree (an ancestor of the tip, no divergent local work). `reports/overnight/` is gitignored (`.gitignore:134-135`).

## Recommended Next Steps

Carried from the 2026-11-23 report (open unless marked resolved), prioritized, maximum five:

1. **Make the sbatch wrapper submit-safe**: add `mkdir -p logs/netmhcpan` before `docs/hla_a0101_run_recipe.md:56` (or drop the `#SBATCH --output` into an existing tracked dir) — one line, and it is the difference between the Stage A step 4 job starting and failing at submission.
2. **Decide the NSP3 input of record and say it in the code**: confirm whether `combined_positives_only.fasta` (new default) or `combined_8_to_14mer.fasta` (as `run_pipeline.R:46` still documents) feeds step 05, then make the two script headers and that comment agree — the "legacy call preserved" claims are currently false either way.
3. **Small dedup/guard batch**: replace `14_predict_external.py:159-190` with `df[FOLD_COL] = 0` + `M.prepare_validation(..., 0)`; scope the `03_prepare_embeddings.py:731` dummy-fold fallback to `ALLELE == "hla_a0101"`; delete the no-op `THESIS_SPLIT` pin (`46_model_compare_29mer.py:38-40`); two plain assignments for the falsy-zero polish record (`46_xgb_optuna_tune.py:266-268`).
4. **Finish the previously agreed one-liners and hardening**: delete the c=5 clause at `42_window_cutoff_sweep_plot.py:159-160` and regenerate `runs/r19_rmdup/plots/sweep_window_cutoff_auc.png`; fix the `--split` help at `04_modelling.py:1457-1461`; pin `optuna` in `requirements_esmc_working.txt` and move `run_xgb_optuna_slurm.sh:88`'s install into `--submit` mode; move `arm_paths` into `config.py`.
5. **Decide what is tracked**: a0101 outputs vs `data/processed/hla_*/` ignore, the model-compare/tuning outputs, and the artifacts requirement from step 2 — then check the cluster (`git ls-tree -r origin/hilarius -- runs/xgb_optuna runs/modelcompare_29mer_46617` and discard any vardims blosum45/90 run from the 11-19 14:27–23:04 window); carry the doc batch (`src/investigation/README.md:5` → "74 scripts (9 R, 65 Python)" + index rows for `42_`–`47_`, `src/README.md:12`'s `01b` line, `umap-learn` in requirements, DiffLogo in `renv.lock`, the shared logo renderer, `30_ae_barplot.py`'s `vals` guard, `run_dimselect_sweep.sh`'s three `:-29mer` echoes, one `flank_size()`-style helper for the 5 copied R stop-guards, the `iter_close_pairs` test at cutoffs 2..9, a shared `_bootstrap.py` for the 21 `spec_from_file_location` sites).

Resolved since the previous report: **the `src/pipeline_a0101/` fork duplication** (deleted in `d47275a`, ~4,365 lines, superseded by the `THESIS_ALLELE` switch), the stale next-step pointer to that fork (`02_load_iedb_data.R:106-111`), and the `run_h100.sh` script name (renamed `run_nsp3.sh` in `902cfa2`, though its log filenames still say `h100`). Previously resolved and confirmed still closed: the `--smoke` trial-count bug, the venv `pip install --user` failure, the R step-02 parse break, the tracked `remove_dup_windows` blosum50 A/B arm, the `DERIVED_MARKERS` denylist bypass, the A/B runner's dimselect switch, the trim5 slide logo producer, the ESM-IF base ranking CSV, tracked CNN sweep tables, the `02_datasplit_analysis.py` import bootstrap, the `29_build_cluster_map.py` README index, the flank5 CNN runner, `runs/r19_flank5/` tracked sweep outputs, the fold-audit relocation into the run tree, the dedup A/B holding the split rule fixed, and the duplicate Cliff's delta implementation.

## Reviewed Through

b10b74a
