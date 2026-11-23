# Thesis Overnight — 2026-11-23

## Summary

Range bb12e52..origin/hilarius contains **0 commits**: `origin/hilarius` is still `bb12e52` (confirmed against the remote this run via `git ls-remote`), i.e. identical to the previous report's anchor, and `git diff --stat` over the range is empty. Nothing changed in the repository since the 2026-11-22 report — no new source, no new tracked results. All carried findings were re-verified at the tip and remain open; the only state change is the calendar (the 7-day window slides to 2026-11-17..2026-11-23 with the same 28 commits). Local branch `hilarius` stays at 26ec48d, 228 behind origin, working tree clean.

## Past 7 Days

Calendar window 2026-11-17..2026-11-23: 28 commits. Commits per day (count then date):

     13 2026-11-17
     10 2026-11-18
      4 2026-11-19
      1 2026-11-22

0 commits on 2026-11-20, 11-21 and 11-23; the most recent day with activity is 2026-11-22 (counts re-derived from `git log --since/--until` over `origin/hilarius`; they match the stated table). Theme level only — this section intentionally overlaps earlier reports:

- **Dedup variant of record (11-17)** — the `flank5_rmdup` default dataset with a `none` opt-out, run scripts switched over, robust-cluster fold audits across window cutoffs c=1..8, random-window logo regimes, and the NetMHCpan affinity-bias report promoted to CLES / Cliff's Δ / bootstrap-CI effect sizes.
- **Window-cutoff and split decision with evidence figures (11-18)** — `WINDOW_CUTOFF` 5 → 6 with mechanistic rationale, the modelling default split moved onto the h6 file, three new evidence scripts (window-distance capture, split-rule figure, dedup A/B), the duplicate-inclusive flank5 sensitivity sweep, and ggseqlogo renderer fixes.
- **New encodings wired end to end (11-19)** — BLOSUM45/BLOSUM90 added to the R matrices, encoding registries and DeLong sweep arms, plus the follow-up fix for component gates that had silently dropped the two new encodings' CSV columns.
- **Modelling comparison + XGB tuning campaign (11-19 late → 11-22)** — an Optuna TPE tuner over 7 hyperparameters with its sbatch submitter and figure script, an XGBoost-vs-rf/LR model-comparison script on the 29mer split, and the 11-22 hardening commit (per-arm `--features` tuning, study-creation race retry, venv pip fix, `--smoke` trial-count fix). All of this landed on or before 11-22 and is covered in previous reports.

## Changes

**Range bb12e52..origin/hilarius = 0 commits, 0 files** (`git rev-list --count` → 0, `git diff --stat` → empty). There are no changes to report in this range, so nothing new appears here and nothing already covered by previous reports is repeated. The tip is byte-identical to the previous report's reviewed-through value.

## Experiments / Results

Only repository-supported results; nothing below was re-executed (no R interpreter on this host; script execution blocked in unattended mode). Re-verified at the tip this run:

- **No new tracked results** — the empty range means no file under `runs/`, `data/`, or `results/` changed since the 2026-11-22 report.
- **The two campaigns remain shipped-but-unrun in the repository (unchanged)** — `git ls-tree -r origin/hilarius runs/xgb_optuna` → **0** tracked files, `runs/modelcompare_29mer_46617` → **0** tracked files, `runs/encodings_sweep` → **0** tracked `cv_results_*.json`, `data/processed/` → **0** tracked files. `runs/is_encodings_helping_v2/results/is_encodings_helping_v2_delong_stats.tsv` still contains **0** `blosum45` rows. So no AUC, no DeLong table, no tuned `best_params` from either campaign exists in the repository.
- **Carried state re-checked (unchanged)** — repo-wide tracked `test_*.py` count is **1**; `src/investigation/README.md:5` still says "33 scripts (6 R, 27 Python)" while the tip tracks **74** (65 Python, 9 R); `spec_from_file_location` appears in **22** files under `src/`; optuna matches in `requirements*.txt` → **0**; exactly one `cliff_delta_sampled` definition remains (`src/pipeline/r/functions.R`).

## Code Review — Shortcomings & Issues

No new findings: the range is empty, so there is no diff to review. Everything below is a carried finding re-verified still open at `origin/hilarius` (= `bb12e52`) this run, ordered as ponytail demands — redundant / over-engineered / reinvented-stdlib-or-native first, then bugs, then style. Severity legend: bug | risk | maintainability | style.

**Redundancy and over-engineering (the class this repo keeps paying for):**

1. **Arm-path mapping written down three times** — `src/investigation/46_xgb_optuna_tune.py:72-83` (`arm_paths`), `src/investigation/47_xgb_optuna_plot.py:59-64` (`arm_dir`, the same rule inverted, docstring even says "Mirror 46_xgb_optuna_tune.arm_paths"), and `src/run/run_xgb_optuna_slurm.sh:48-52` (the `blosum50` special case). One helper in `config.py` (both Python files already import it) deletes two copies; the shell keeps one commented copy. Ceiling: the shell branch stays as a third source until the Python helper is exported or the arm list is derived from one config entry. Maintainability. Cites: [bb12e52].
2. **`delong_table` re-implements `holm_family` from the module it imports** — `46_model_compare_29mer.py:94-115` hand-rolls `holm_adjust` + zip + `p_adj`, while `18_encodings_sweep_delong.py:123-127` already defines `holm_family(pairs, key)` and `d18.stars` is already reused one line below. Two lines replace the loop. Maintainability — reinvented existing helper. Cites: [8147f53].
3. **22nd `spec_from_file_location` bootstrap** — `git grep -l spec_from_file_location origin/hilarius -- src/` → **22** files; `46_model_compare_29mer.py:41-45` is the newest. Python's native import system (a package dir, or one shared `_bootstrap.py`) does this once. Ceiling: converting `src/` to a package is bigger than any one script needs; the lazy step is the shared bootstrap module. Maintainability/style. Cites: [8147f53].
4. **Hand-rolled TSV writing with pandas already imported** — `46_model_compare_29mer.py:176-197` builds two TSVs from header strings and per-row f-strings; `pd.DataFrame(rows).to_csv(path, sep="\t")` is shorter and cannot drift from its own header. Style, low. Cites: [8147f53].
5. **Four module globals mutated in `main`, two derivable** — `47_xgb_optuna_plot.py:177-181` reassigns `RUN_DIR`, `PLOTS`, `STUDY_NAME`, `LABEL` via `global`, though `LABEL`/`RUN_DIR`/`PLOTS` are pure functions of `STUDY_NAME`; the search space is also written down twice (`46_xgb_optuna_tune.py:150-158` `suggest` vs `47_xgb_optuna_plot.py:55-56` `PARAMS`, with `t.params.get(p, np.nan)` at `:122` degrading silently to `nan` on a renamed parameter). One local passed through removes the state. Style, low. Cites: [bb12e52].

**Bugs and risks:**

6. **The `THESIS_SPLIT` pin is a no-op (risk — a safety check that checks nothing)** — `46_model_compare_29mer.py:38-40` sets `THESIS_SPLIT`, but `config.py:652-653` routes it into `DEFAULT_SPLIT_PATH` and no consumer in that process reads `DEFAULT_SPLIT_PATH`: the script reads `SPLIT_DATA_PATH` (`:201`), `18_encodings_sweep_delong.py` reads `SPLIT_DATA_PATH` (`:59`, `:477`). Only `THESIS_VARIANT=none` (`:37`) actually pins anything; the 46,617-row assert (`:202-204`) is the only real guard. Delete the line or make both readers use `DEFAULT_SPLIT_PATH`. Cites: [8147f53].
7. **Encoding-name registry hand-copied across six files (maintainability — already caused a silent-corruption bug)** — hardcoded sets still at `04_modelling.py:325`, `extract_model_metrics.py:44`, `config.py:1115`, `18_encodings_sweep_delong.py:205`, `08_aa_encoding.R:58-62`, `functions.R:1515`; `d4460e2` exists only because two Python copies were missed silently for vardims arms. Derive `AA_ENCODING_COMPONENTS` from `FEATURE_COMPONENTS`, or add the one `assert`. Cites: [1da5d6a, d4460e2].
8. **Concurrent unpinned pip installs into one venv (risk)** — `run_xgb_optuna_slurm.sh:88` runs `|| pip install optuna` in two jobs that start together; if optuna is absent on the node, two writes into one site-packages can corrupt it, and optuna is still unpinned (`requirements*.txt` → 0 matches). Lazy fix: install once in `--submit` mode, workers only `python -c "import optuna" || exit 1`, plus one pin. Cites: [bb12e52].
9. **Contradictory sweep footnote baked into a tracked figure (bug)** — `42_window_cutoff_sweep_plot.py:158` still emits "chosen setting c=5 (last point before any downward drift)" into the same caption as `:162` "Chosen setting c=6 …"; re-read at the tip, both clauses are still there and the tracked sweep PNG still contains it. One-line deletion + regenerate. Cites: [carried].
10. **`18_encodings_sweep_delong.py` cannot be re-run under default config (risk)** — `:477` reads `SPLIT_DATA_PATH` (45,278 rows under default `flank5_rmdup`) while the tracked campaign is 46,617 rows, and `:500-502` require the blosum45/90 arms → `Missing arms`. Authors' question: which split of record should this script read? Cites: [1da5d6a, carried].
11. **Polish winner record: falsy-zero bug (bug, low)** — `46_xgb_optuna_tune.py:266-268` still uses `user_attrs.get("auc_folds") and float(np.std(...)) or None`, so a std of exactly `0.0` becomes `None`; `fixed` still omits `auc_folds`/`logloss_folds`. Two plain assignments fix both. Cites: [b1dd369, carried].

**Style / docs drift (re-verified, all still open):**

12. **Copied-paste blocks that drift independently** — the `n_pad`/`c_pad` block re-typed in `46_xgb_optuna_tune.py:98-104` (verbatim from `04_modelling.py:1795-1801`) matters most, since the tuner's purpose is params that transfer to `04_modelling`; plus the FLANK_SIZE stop-guard copied in 5 R files, `newest_json`/`oof_aligned` copied across 3 figure scripts, the h6 split filename spelled out in 3 places instead of `DEFAULT_SPLIT_PATH`. Maintainability. Cites: [carried].
13. **Doc drift** — `run_xgb_optuna_slurm.sh:18` ("Results land in runs/xgb_optuna/") is true only for the default arm and `:16`'s `squeue -n xgb_opt` hint no longer matches suffixed job names; `04_modelling.py:1457-1461` `--split` help says "Default: config SPLIT_DATA_PATH" while the code uses `DEFAULT_SPLIT_PATH`; `src/investigation/README.md:5` "33 scripts (6 R, 27 Python)" vs 74 tracked and no index rows for `42_`–`47_`; `src/README.md:12` still lists deleted `01b_datasplit_protein_level.py`; `31_random_peptide_logos.R:195` cites non-existent commit `5ebb687`. Style, low. Cites: [carried].
14. **Asserts for runtime preconditions** — `46_xgb_optuna_tune.py:110`/`:247` use `assert` (stripped under `-O`); note `46_model_compare_29mer.py:202-204` does the same check correctly with `sys.exit`. Style, low. Also carried: unused `csv_cols` unpacks at `:199`/`:244`; unused `fold_vec` param in `plot_roc` (`46_model_compare_29mer.py:155`); discarded `_a`/`_b` at `:101-104`.
15. **Dead / tautological guards carried unchanged** — `45_dedup_compare_plot.py:100-103`, `30_plot_fold_audit.py:41`, `30_ae_barplot.py:110-116`, `run_dimselect_sweep.sh:73,81,93` `:-29mer` echoes, `pipeline_a0101/python/config.py:52` `pop()`-restores `THESIS_VARIANT`. Style, low.
16. **What passes the ladder** — nothing to add this range (empty diff). Last range's good calls still stand: the tuning refactor parameterises rather than abstracts, the race retry is a plain `for`/`time.sleep` loop rather than a retry library, the model-compare script reuses `d18.load_model`/`delong_full`/`_padmode_*` instead of copying DeLong or the plotting style, the submitter stays plain bash, and `47_` uses Optuna's native fANOVA importance.

## Potential Issues

1. **Two campaigns still produce nothing the repository can show (data/process, carried)** — `runs/xgb_optuna/`, `runs/modelcompare_29mer_46617/`, `runs/encodings_sweep`'s `cv_results_*.json` and `data/processed/` are all 0 tracked files, and the DeLong stats have 0 `blosum45` rows. A reader of this repo cannot see a single tuning or model-comparison number. Question for the authors: are these outputs intended to be tracked like `runs/r19_rmdup/`, or are the scripts laptop/cluster-only tools? (Unchanged for a second consecutive report.)
2. **Selection-on-the-test-set question for the tuning campaign (authors' question, carried)** — `46_xgb_optuna_tune.py`'s objective is mean AUC over the 6 outer buckets "identical to the nested-CV outer evaluation", and `47_xgb_optuna_plot.py`'s docstring names a same-split refit + DeLong as the follow-up: selecting the best of ~100 trials on the fold partition that produces the reported CV number, then DeLong-testing that tuned refit on the same rows, is a question about what the reported figure estimates. Not deciding it here.
3. **The silent-corruption window is ~9 hours wide (observation — cluster check needed, carried)** — `1da5d6a` (14:27) introduced the gating omission, `d4460e2` (23:04) fixed it. No tracked artifact is affected, but any vardims blosum45/90 run completed in that window was trained embedding-only under the wrong label and should be discarded rather than compared.
4. **Two parallel writers on one SQLite study (process, known ceiling)** — `run_xgb_optuna_slurm.sh` still fans two tune jobs onto one `study.db` per arm; the retry at `46_xgb_optuna_tune.py:163-188` covers only *schema creation*, not `database is locked` during `study.optimize`. Ceiling unchanged; upgrade path = one writer per study or PostgreSQL. Not needed until it errors.
5. **Git history as the only index (process)** — with no PRs, issues or CI, `git log --oneline` is the whole index, and it is misleading: `8147f53` is a 222-line new script committed as "style: model-compare bar title to storyline opener". Commit messages are where a reader looks first.
6. **Verification limits (process)** — local branch `hilarius` at 26ec48d is **228** behind origin with an empty `git status --porcelain`, so every cite comes from `git show`/`git grep`/`git ls-tree` against the tip; `command -v Rscript` → none; script execution is blocked in unattended mode. The 90 MB raw-data blob remains in history untouched.

## GitHub

From the collected GitHub data (used verbatim, not re-run): no open PRs (`gh pr list` → []), no open issues (`gh issue list` → []), no workflow runs (`gh run list` → []); repository `TheHilarius/thesis2026`, default branch `hilarius`. From git: origin/hilarius = **bb12e52** (2026-11-22), i.e. **0 commits ahead of the previous report's anchor** (`git rev-list --count bb12e52..origin/hilarius` → 0), confirmed against the remote this run via `git ls-remote origin refs/heads/hilarius` → `bb12e52add157beeba72ab94f0dcf8d789ac9a1a` (read-only). Local branch `hilarius` at 26ec48d remains **228** behind with a clean working tree (an ancestor of the tip, no divergent local work). `reports/overnight/` is gitignored (`.gitignore:115`), so this report exists only in the local working tree.

## Recommended Next Steps

Carried from the 2026-11-22 report (open unless marked resolved), prioritized, maximum five:

1. **Collapse the triplicated arm-path rule**: move `arm_paths` into `config.py`, import it in `47_xgb_optuna_plot.py:59-64`, keep the shell's one copy as a commented mirror; and either delete the no-op `THESIS_SPLIT` pin at `46_model_compare_29mer.py:38-40` or switch both readers to `DEFAULT_SPLIT_PATH` — first decide which split that script is supposed to load.
2. **Two one-liners already shipped wrong**: delete the c=5 clause at `42_window_cutoff_sweep_plot.py:158-161` and regenerate `runs/r19_rmdup/plots/sweep_window_cutoff_auc.png`; fix the `--split` help at `04_modelling.py:1457-1461` to say `DEFAULT_SPLIT_PATH`.
3. **Finish the dependency/pin hardening**: pin `optuna` in `requirements_esmc_working.txt` and move `run_xgb_optuna_slurm.sh:88`'s install into `--submit` mode (kills both the unpinned-install and the concurrent-pip-into-one-venv risks); derive `AA_ENCODING_COMPONENTS` from `FEATURE_COMPONENTS`; two plain assignments for the polish record (`46_xgb_optuna_tune.py:266-268`).
4. **Decide what is tracked from the two campaigns**: commit or document the `..._updated_h6_pepcut1.csv` split and the model-compare outputs, then check cluster output — `git ls-tree -r origin/hilarius -- runs/xgb_optuna runs/modelcompare_29mer_46617 runs/encodings_sweep | grep -c cv_results`; if any vardims blosum45/90 run completed in the 11-19 14:27–23:04 window, discard it before comparing.
5. **Carried batch (still open)**: `src/investigation/README.md:5` → "74 scripts (9 R, 65 Python)" plus index rows for `42_`–`47_`; `src/README.md:12`'s `01b` line; `umap-learn` in requirements; DiffLogo in `renv.lock`; the shared logo renderer for `13_`/`31_` with the y-limit fix; `30_ae_barplot.py`'s `vals` guard and invisible `axhline(0.5)`; `29_build_cluster_map.py`'s mode-derived default out-name; `run_dimselect_sweep.sh`'s three `:-29mer` echoes; the row/order assert for the (now default) split load; one `flank_size()`-style helper replacing the 5 copied R stop-guards; the `iter_close_pairs` brute-force test at cutoffs 2..9; a shared `_bootstrap.py` for the 22 `spec_from_file_location` sites.

Resolved since the previous report: **nothing** — the range is empty. Previously resolved and confirmed still closed: the `--smoke` trial-count bug (`46_xgb_optuna_tune.py:287` default `None`), the venv `pip install --user` failure (`run_xgb_optuna_slurm.sh:88`), the R step-02 parse break, the tracked `remove_dup_windows` blosum50 A/B arm, the `DERIVED_MARKERS` denylist bypass, the A/B runner's dimselect switch, the trim5 slide logo producer, the ESM-IF base ranking CSV, tracked CNN sweep tables, the `02_datasplit_analysis.py` import bootstrap, the `29_build_cluster_map.py` README index, the flank5 CNN runner, `runs/r19_flank5/` tracked sweep outputs, the fold-audit relocation into the run tree, the dedup A/B holding the split rule fixed, and the duplicate Cliff's delta implementation (only the chunked `outer()` version remains in `functions.R`).

## Reviewed Through

bb12e52
