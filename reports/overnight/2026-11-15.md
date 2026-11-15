# Thesis Overnight — 2026-11-15

## Summary

Range 79ee04d..origin/hilarius = **2 commits** (both 2026-11-14, linear, 0 merges): a parse-error fix in the R loader, a mode-dependent split cutoff default (`HAMMING_CUTOFF=1` for peptide mode, new `WINDOW_CUTOFF=5` for context mode), dynamic `FLANK_SIZE` in the NetSurfP/nsp3 feature extractors, and the `FLANK_SIZE` flag in the Assarsson prep (default now 19-mer / 399 features). The only tracked artifact to change is a regenerated sweep PNG whose footnote now records c=5 as the chosen setting; no new result tables, docs, or tests landed. Anchor advances 79ee04d → **5a3ca3d**; the local checkout is 200 commits behind with a clean tree. All carried items — dataset of record, two allele paths, README drift, the missing split-code tests, the 90 MB blob — are still open.

## Past 7 Days

Calendar window 2026-11-09..2026-11-15: 44 commits. Commits per day (count then date):

      4 2026-11-09
      2 2026-11-10
     22 2026-11-11
      1 2026-11-12
      7 2026-11-13
      8 2026-11-14

0 commits on 2026-11-15; the most recent day with activity is 2026-11-14. Theme level only (file:line detail stays under ## Changes); this section intentionally overlaps earlier reports:

- **CNN campaign tuning (11-09 → 11-10, 6 commits)** — kernel-size and channel sweeps, flank5 CNN reruns with per-fold seeds, 19-vs-29mer k5 test, AE no-compression control arm, leakage investigations.
- **Results-first analysis batch plus two external-validation efforts (11-11, 22 commits)** — paired significance on flank5 19mer-vs-29mer, AE capacity DeLong, supervised-dimselect screening at matched EV budgets, XGB-vs-simpler-models comparison, random-peptide logo null, ROC-crossing and flank duplicate-leakage audits; the Assarsson A02:01 external set with its tracked metrics, a parallel HLA-A*01:01 pipeline edition with a 90 MB raw IEDB export; supervised dimselect promoted to the default embedding arm, regenerated figures, gitignore entries for raw dumps/logs.
- **Dataset-variant tooling (11-12, 1 commit)** — the `remove_dup_windows` row-mask rule with drop-count guards, prepared-H5 subsetting, and the A/B runner's switch to the dimselect arm.
- **Allele restructure + robust split + plan docs (11-13, 7 commits)** — HLA-A*01:01 parameterized into the shared fetch/R scripts and the netMHCpan runner; `01_datasplit` rewritten around `--group-by context` (window ∪ peptide edges plus the flank-identity edge) with deterministic bucket ordering and marked output files; three handoff plan docs; the first tracked `remove_dup_windows` A/B result; two small sweep flags.
- **Flank flag, dedup runner, sweep results and this range's fixes (11-14, 8 commits)** — `FLANK_SIZE` (5 default → 19-mer) plus context-window dedup and multi-allele looping in the R pipeline; `--pep-cutoff {1,2}` and window Hamming cutoffs up to 9 via a growing-block pigeonhole index; the `flank5_rmdup` dataset variant; four tracked dedup/grouping/cutoff sweep tables; then the two fix commits reviewed here (R step-02 parse error, `WINDOW_CUTOFF=5` default, `FLANK_SIZE` in nsp3 features and in the Assarsson prep).

## Changes

Range 79ee04d..5a3ca3d = 2 commits, linear. Cites are `git show <sha>:<path>` / line numbers at `origin/hilarius` (the local checkout predates the range).

**Topic 1 — step-02 parse fix (ac839b1: `src/pipeline/r/02_load_iedb_data.R`, +4/−5)**

- `:16-26`: the three top-level `if/else` blocks now keep `else` on the same line as the closing value (`IEDB_RAW`, `OUT_POS`, `OUT_STAGES`). R treats a newline before `else` at top level as a parse error, so the allele-switch edit introduced earlier could not be parsed at all and blocked `run_pipeline` step 02. This is the smallest correct fix; the file's inline `commandArgs`/`match.arg` block (`:9-15`) that duplicates `functions.R`'s `parse_allele_arg()` is untouched (carried finding).

**Topic 2 — mode-dependent split cutoff (ac839b1: `config.py`, `01_datasplit.py`, +13/−6)**

- `config.py:627-631`: `HAMMING_CUTOFF = 1` stays as the peptide-mode default *and* the canonical-vs-marked naming discriminator (comment says never point it at window c); new `WINDOW_CUTOFF = 5` records the chosen context-mode cutoff with its rationale. `validate_config()` gains `assert WINDOW_CUTOFF in range(1, 10)` (`:1266`).
- `01_datasplit.py:878-882`: `--hamming-cutoff` default becomes `None` with prose documenting both modes; `:910-915` resolves it after parsing (peptide → `HAMMING_CUTOFF`, context → `WINDOW_CUTOFF`). The marked-output rule (`:948`) is unchanged, so context runs still never write the canonical `df_all_with_folds.csv`. Behavior change: a manual `--group-by context` run with no explicit cutoff now produces `updated_h5…` instead of `updated_h1…`; no shell runner in the repo invokes `--group-by context` (grep over `*.sh`/`*.R`/`*.py` finds none), so only interactive invocations are affected.
- `42_window_cutoff_sweep_plot.py:131-132`: footnote extended with "chosen setting c=5 (last point before any downward drift)"; `runs/r19_rmdup/plots/sweep_window_cutoff_auc.png` regenerated (147,782 → 155,750 bytes). The companion CSV is byte-unchanged.

**Topic 3 — `FLANK_SIZE` in the nsp3 extractors (ac839b1: `src/pipeline/r/functions.R`, +11/−10)**

- `extract_nsp3_windows` (`:645-651`, body `:674-679`) and `extract_nsp3_q8_features` (`:1166-1172`, body `:1183-1192`) take a `flank_size` argument defaulting to `as.integer(Sys.getenv("FLANK_SIZE", "5"))`, replacing hardcoded `target_len = 10` / `n_nflank = 10, n_cflank = 10`. Their only callers, `06_netsurfp_protein_features.R:89,111`, do not pass the argument and step 06 has no `FLANK_SIZE` handling of its own, so step 06 now follows the env/`functions.R` default. Commit message records that explicit `10` was verified byte-equal to the previous output.

**Topic 4 — `FLANK_SIZE` in the Assarsson prep (5a3ca3d: `src/pipeline/r/10_assarsson_preparation.R`, +17/−6)**

- `:18-30`: env read + `{5,10}` guard + banner, and `POSITION_COLS` derived as `N{FLANK_SIZE}..N1 / P1..P9 / C1..C{FLANK_SIZE}` (the nine `P` columns are unchanged); `:138-139` `N_FLANK_SIZE/C_FLANK_SIZE <- FLANK_SIZE`; `:165-166` the `full_context` width assert generalized from `{29,30}` to `{2*L+9, 2*L+10}`. Commit message records acceptance: `FLANK_SIZE=10` byte-identical to the previous outputs, default `5` emits 399 features "matching flank5 models".

No other files changed in the range: no new docs, no new tests, no new result tables.

## Experiments / Results

Only repository-supported results. The range itself adds **no new numbers** — one figure was regenerated from an unchanged CSV, and its footnote now names the chosen cutoff.

- **Window-cutoff sweep (tracked, unchanged CSV; footnote updated in ac839b1)** — `runs/r19_rmdup/plots/sweep_window_cutoff_auc.csv`: c=1,2,3 identical by construction (one run `pc1`, 0.770618 ± 0.006768); c=4 0.770925 ± 0.005440; c=5 0.770168 ± 0.004769; c=6 0.768816; c=7 0.769868; c=8 0.768497. Paired DeLong vs c=1: raw p = 0.748 (c4), 0.602 (c5), 0.0434 (c6), 0.399 (c7), 0.0197 (c8); Holm-adjusted 1.0 / 1.0 / 0.1735 / 1.0 / 0.0986 — no adjusted-significant difference up to c=8. Observation: the CSV's `json` column names the input `cv_results_xgb_blosum50_flat_*.json` files, none of which are tracked.
- **Chosen setting recorded, not re-measured** — c=5 appears as the choice in the figure footnote (`42_window_cutoff_sweep_plot.py:132`) and in `config.py:629-631`. Note that c=4 is the highest mean in the table and c=4 vs c=5 is not distinguishable (raw p = 0.602–0.748); "last point before any downward drift" is a reading of a noisy series, not a measurement.
- **Tracked external result that the new default no longer reproduces** — `results/external/assarsson_xgb_blosum50_flat_metrics.json` (tracked at c63db75, 2026-11-11): all rows n=104, AUROC 0.6667 / AUPRC 0.7607; 9-mer-only n=73, AUROC 0.7406 / AUPRC 0.8128; 30 pickles from `runs/encodings_sweep/models/flat`. It was produced by `10_assarsson_preparation.R` while that script still hardcoded 10-residue flanks (29-mer contexts); see Code Review 3.
- **Unchanged tracked tables re-verified** — `runs/rob_sweep/plots/grouping_rule_auc.csv` still ships the empty `canonical pep-ham≤1 only × flank5 19aa` cell; `runs/ab_dataset_test/results/ab_dataset_delong_remove_dup_windows.tsv` still Δ +0.001310, p_delong 0.1558, n_pairs 45,510.
- **Correction to the previous report** — it stated "0 JSON files under `runs/`"; there are **53** tracked JSONs under `runs/` at both 79ee04d and 5a3ca3d, all CNN `learning_curves_*_meta.json`. What remains untracked is every `cv_results_*.json` the XGB tables cite (`git ls-tree | grep cv_results` matches only two PNG filenames), so the provenance caveat itself stands.

## Code Review — Shortcomings & Issues

Range findings first (redundant / over-engineered / reinvented-helper first), then carried findings re-verified at 5a3ca3d.

1. **The `FLANK_SIZE` guard is now copied five times (maintainability — redundant)** — `05_protein_features.R:53-56`, `07_alpha_fold_features.R:8-11`, `08_aa_encoding.R:8-11`, `run_pipeline.R:24-27` and, added by this range, `10_assarsson_preparation.R:20-23` are verbatim copies of env read + `{5,10}` stop + banner. `functions.R` is sourced by all of them and already reads the same env in three default arguments (`:484`, `:650`, `:1171`) with *no* validation. One `flank_size()` helper next to the existing `parse_allele_arg()` removes five copies and closes a real gap: `FLANK_SIZE=7` stops steps 05/07/08/10 but is silently accepted by the three `functions.R` defaults, so a standalone step-06 run would impute to a width nothing else validates. [0d15e47, widened by 5a3ca3d]
2. **`ac839b1` silently changes the values in every previously built flank5 dataset (risk)** — `functions.R:674-679` and `:1183-1192` previously hardcoded a 10-residue flank target for RSA/disorder/Q8 imputation regardless of `FLANK_SIZE`; they now default to 5. Every flank5 artifact built before this commit (`df_all_flank5.csv`, `flank5_rmdup`, the tracked `runs/r19_rmdup/*` and `runs/rob_sweep/*` tables, the flank5 CNN runs) therefore contains nsp3 features computed over a 29-position window with 5 real + 5 imputed residues per side, while a rebuild now produces 19-position values. The commit message's "explicit 10 verified byte-equal to env" implies `FLANK_SIZE=5` does *not* reproduce the old output. Nothing records which side of this boundary a tracked table sits on. [ac839b1]
3. **The Assarsson prep default no longer reproduces the tracked external result (risk)** — `10_assarsson_preparation.R:20` now defaults to `FLANK_SIZE=5` (19-mer, 399 features), but `results/external/assarsson_xgb_blosum50_flat_metrics.json` was produced by the same script under hardcoded `N_FLANK_SIZE <- 10` / `C_FLANK_SIZE <- 10` (29-mer contexts) with the 30 `runs/encodings_sweep/models/flat` pickles. A default re-run of prep+eval either fails on feature-count mismatch or yields a different number than the tracked 0.6667 / 0.7406. The intent is stated in the commit message; it is not stated in any doc, and no README records which flank the tracked JSON used. [5a3ca3d]
4. **`WINDOW_CUTOFF` validates and documents itself twice (style, low)** — `config.py:1266` asserts `1..9` on the same value `01_datasplit.py:931-933` re-validates in argparse, and the help text at `:880-882` hardcodes "(1)" and "(5)" in prose because `%(default)s` became `None`; editing either config constant leaves the help wrong. Two lines of the change are redundant, the rest passes the ladder. [ac839b1]
5. **Carried findings re-verified at 5a3ca3d, all still open** — 90 MB `data/raw/mhc_ligand_table_export_HLA-A0101.csv` still tracked; `src/investigation/README.md:5` still "33 scripts (6 R, 27 Python)" vs **68** tracked `.py`/`.R` in that directory, table still 25 rows, `:331-332` still "Not run", no rows for `29_ab_dataset_compare.py` or the `41_`/`42_` scripts; 17 `spec_from_file_location` files (still 17 — `42_window_cutoff_sweep_plot.py:43` is the 17th, `:55` still selects its input by `os.path.getmtime` although the filenames carry sortable timestamps, `:118` still passes 5 colours for 8 bars); repo-wide `test_*.py` count still **1** (`src/tools/esm/test_embed_esmc6b.py`), so the growing-block pigeonhole index supporting cutoffs 2..9 still has no brute-force check; no 9-mer assert at `01_datasplit.py:304` (grep for `eq(9)` = 0 matches) while flank slicing assumes `peptide = 9`; the `--split` row/order contract in `04_modelling.py:1457-1463` is still enforced only on the fold column; `02_load_iedb_data.R:9-15` still hand-parses the allele argument instead of calling `parse_allele_arg()`; `.gitignore:32` still lacks a wildcard (the blob stays tracked) with `:35-36` duplicating `:31`/`:33`; `umap-learn` still absent from the requirements files and DiffLogo still absent from `renv.lock` (0 matches); `src/README.md:12` still lists the deleted `01b_datasplit_protein_level.py`.
6. **What passes the ladder (range)** — the `else` fix is a line move, not a refactor; the mode-dependent default is four lines instead of a new subcommand or a config object; `WINDOW_CUTOFF` is one documented constant; the nsp3 change threads nothing through step 06 — a default argument reusing the env convention already in `functions.R`; the Assarsson width assert reuses the file's existing `stopifnot` style. No new dependency, no new file, no abstraction with one implementation anywhere in the range.

## Potential Issues

1. **Which dataset is now the "base"? (carried, now with a third axis)** — `config.py:19` still documents "Raw embedding H5s stay base-only (N=46,617)" while R `07_alpha_fold_features.R:250-266` dedups unconditionally to 45,278 (flank5) / 45,510 (flank10). On top of the row-count question, `ac839b1` means two builds of the *same* flank5 row set now differ in nsp3 feature values. One file has to record the construction of record before anything is re-run.
2. **Two allele paths, no owner (carried)** — the shared tree gained the parse fix and `FLANK_SIZE` work; `src/pipeline_a0101/` was not touched by this range and still diverges (flank 10 vs 5, no context dedup), and its README's "legacy 02 untouched by design" line still contradicts the earlier shared-`02` edit.
3. **Tracked results straddle a feature-value change with no provenance note (new)** — items 2 and 3 above: the flank5 tables and the Assarsson metrics JSON each encode a specific `FLANK_SIZE`, and neither their run directories nor any README says which. Authors' call: pin `FLANK_SIZE` in the reproduction command, regenerate, or annotate — do not leave it implicit.
4. **Result tables without inputs or run READMEs (carried)** — `runs/r19_rmdup/` and `runs/rob_sweep/` still contain plots only (no run README naming arm/dataset), the cited `cv_results_*.json` are untracked, the `canonical × flank5 19aa` cell is still empty, and the dimselect arm of `run_ab_dataset_test.sh` plus `run_rmdupwin_ab.sh`'s embedding arm still have no tracked output.
5. **90 MB blob in history (carried, unchanged)** — keep → repair the dead `.gitignore:32` and the duplicate lines; drop → `git rm --cached` plus a history rewrite that rewrites every commit SHA.
6. **c=5 as the chosen setting rests on a footnote (observation → authors' question)** — the sweep shows c=4 highest and c=4 vs c=5 indistinguishable, with all Holm-adjusted p ≥ 0.099; whether c=5 or c=4 is the pre-registered choice, and whether the choice belongs in a figure footnote or in the plan docs, is the authors' call.
7. **Process / verification limits** — 2 commits in the daily range (both 2026-11-14, none on 2026-11-15); local checkout at 26ec48d is **200** commits behind with a clean tree (0 modified files), so every cite above comes from `git show`/`git grep` against `origin/hilarius`. No R interpreter exists on this host, so the R files could not be parse-checked here — all **101** `.py` files at `origin/hilarius` compile (one pre-existing `SyntaxWarning` at `src/investigation/06_feature_importance.py:11`, invalid escape sequence, not in range).

## GitHub

From the collected GitHub data (used verbatim, not re-run): no open PRs (`gh pr list` → []), no open issues (`gh issue list` → []), no workflow runs (`gh run list` → []); repository `TheHilarius/thesis2026`, default branch `hilarius`. From git: origin/hilarius = **5a3ca3d** (2026-11-14), 2 commits ahead of the previous anchor 79ee04d in a linear chain (0 merges); local branch `hilarius` at 26ec48d is 200 commits behind with a clean working tree (0 modified files). `reports/overnight/` is gitignored (`.gitignore:134-135` "Overnight reports — published only to the `reports` branch"), so this report exists only in the local working tree.

## Recommended Next Steps

Carried from the 2026-11-14 report (open unless marked resolved), prioritized, maximum five:

1. **Decide the dataset of record (carried, now blocking more than row counts)**: settle whether `df_all.csv` is the 46,617-row base (Python `make_variant_dataset.py` view) or the deduped 45,510/45,278-row output of R `07:250-266`, whether `FLANK_SIZE=5` is the default of record, and now also which side of `ac839b1` existing flank5 files were built on. One file (`docs/` or `config.py`) records all three.
2. **Pin flank provenance for the tracked artifacts (new)**: record `FLANK_SIZE` next to `results/external/assarsson_xgb_blosum50_flat_metrics.json` and the `runs/r19_rmdup/` + `runs/rob_sweep/` tables (run README or reproduction command), and decide regenerate-vs-annotate for anything built before `ac839b1`.
3. **One `flank_size()` helper plus the missing checks (carried, widened)**: collapse the five guard copies (`05:53`, `07:8`, `08:8`, `run_pipeline:24`, `10_assarsson:20`) into `functions.R` beside `parse_allele_arg()`, then add the `iter_close_pairs` brute-force test at cutoffs 2..9, the 9-mer assert at `01_datasplit.py:304`, and the row/order check for `04_modelling --split`.
4. **One README/gitignore batch (carried)**: `src/investigation/README.md:5` → 68 scripts, add rows for the undocumented scripts and the `41_`/`42_` scripts, flip `:331-332` "Not run"; delete `src/README.md:12`'s `01b` line; add `umap-learn` to the requirements files; decide DiffLogo vs `renv.lock`; de-duplicate and repair `.gitignore:5-6,31-36`.
5. **Run and track the remaining A/B arms (carried)**: the dimselect arm of `run_ab_dataset_test.sh` (with a separate `--run-dir` so the tracked default `ab_dataset_delong.tsv` survives) and `run_rmdupwin_ab.sh`'s embedding arm, each with a run README naming arm and dataset, plus the missing `canonical × flank5 19aa` cell.

Resolved since the previous report: the R step-02 parse break introduced by the earlier allele edit is fixed by `ac839b1` (new item, closed within this range). Corrected: the "0 JSON under `runs/`" claim — 53 CNN meta JSONs are tracked, only the `cv_results_*.json` inputs are not. Still resolved from earlier reports: the tracked `remove_dup_windows` blosum50 A/B arm, the `DERIVED_MARKERS` denylist bypass via `--source-tag`, the A/B runner's dimselect switch, the trim5 slide logo producer, the ESM-IF base ranking CSV, tracked CNN sweep tables, the `02_datasplit_analysis.py` import bootstrap, the `29_build_cluster_map.py` README index, tracked fold-audit tables, a flank5 CNN runner.

## Reviewed Through

5a3ca3d
