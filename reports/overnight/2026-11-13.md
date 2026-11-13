# Thesis Overnight — 2026-11-13

## Summary

Range 78d683e..origin/hilarius = **1 commit** (3ca725b, 2026-11-12; 6 source files, +345/−32): a `remove_dup_windows` dataset variant with a rule registry and drop-count guards, a new prepared-H5 subsetter that documents the `03_prepare` row-dropped misalignment hazard, and the A/B runner finally switched to `--pca-mode dimselect --top-n default` — resolving the carried runner-vs-commit-message mismatch. The range adds no tracked results (nothing under `runs/` or `results/` changed). The anchor advances 78d683e → **3ca725b**; the local checkout is 185 commits behind with a clean tree. Carried items otherwise unchanged: the 90 MB blob, the README drift, and the 16 dynamic-import loaders all remain open.

## Past 7 Days

Calendar window 2026-11-07..2026-11-13: 30 commits. Commits per day (count then date):

      1 2026-11-08
      4 2026-11-09
      2 2026-11-10
     22 2026-11-11
      1 2026-11-12

2026-11-07 and 2026-11-13 have **0** commits; the most recent day with activity is 2026-11-12. Theme level only (file:line detail stays under ## Changes):

- **CNN campaign tuning and sweep setup (11-08 → 11-10, 7 commits)** — AE no-compression control-arm fix, kernel-sweep parameter fix, CNN flank5 reruns with per-fold random seeds, 19-vs-29mer k5 test, leakage investigations, channel and two-layer kernel sweeps.
- **Results-first analysis batch (11-11)** — flank5 19mer-vs-29mer paired significance, AE capacity DeLong, supervised-dimselect screening at matched EV budgets, XGB-vs-simpler-models comparison, A/B dataset outputs, random-peptide logo null model, ROC-crossing screen, flank duplicate-leakage audit.
- **Two external-validation efforts (11-11)** — the Assarsson A0201 set and a parallel HLA-A*01:01 pipeline edition with its 90 MB raw IEDB export.
- **Pipeline default and figure/tooling refresh (11-11)** — supervised dimselect promoted to the default embedding arm, `--campaign`/`--variant` sweep flags, gitignore entries for raw dumps/logs, regenerated PCA/variance figures, a slide logo, an MPNN booktabs table.
- **Dataset-variant tooling (11-12)** — the `remove_dup_windows` row-mask rule, a prepared-H5 subsetting script, and the A/B runner's switch to the dimselect arm.

## Changes

Range 78d683e..3ca725b = 1 commit, linear (0 merges). All cites are `git show <sha>:<path>` (the local checkout predates the range).

**Topic 1 — remove_dup_windows dataset variant (3ca725b: `src/pipeline/python/make_variant_dataset.py`, `config.py`)**

- `make_variant_dataset.py:43-64`: the mask logic becomes two named rules — `orphan_mask` (previous behavior) and new `dup_window_mask` (one row per `full_context` = n_flank|peptide|c_flank; a label-conflicting window keeps its positive row; same peptide with different flanks is kept).
- `:67-70` `RULES = {"remove715negonly": (orphan_mask, 2271), "remove_dup_windows": (dup_window_mask, 1107)}`; `--expect-orphan` is replaced by `--expect-dropped` (`:78-80`) defaulting per rule, and the drift guard becomes `assert n_dropped == expect` (`:93`).
- Per-rule post-checks `:99-119`: orphan-free for remove715negonly; for remove_dup_windows, no duplicate `full_context`, `nunique == len(out)`, no peptide lost, plus a printed breakdown. Commit-stated guard numbers: 1,107 rows dropped (361 duplicate positive windows, 746 negatives = 352 label-conflicting + 394 same-label) and 304 label-conflicting windows — these are runtime assertions; `df_all.csv` is untracked, so the repo cannot corroborate them.
- `:126-128` writes `variant_mask_<variant>.csv` (kept base row indices) as the handoff to the H5 subsetter.
- `config.py:16-23`: documents both variants and the hazard — `03_prepare_embeddings`' `align_window_by_row_indices` maps old row indices into the shorter variant df and "silently misaligns"; raw embedding H5s stay base-only (N=46,617).

**Topic 2 — prepared-H5 subsetting script (3ca725b: new `src/pipeline/python/prepare_variant_h5.py`, 207 lines)**

- Builds variant prepared H5s by positional subset of the base prepared files instead of re-running `03_prepare`; the docstring (`:3-17`) records the misalignment reason and the input/output contract.
- `:73-96` rewrites `folds`/`labels`/`peptide_seqs`/`uniprot_ids`/`starts`/`df_row_indices` from the variant fold CSV; every other dataset is copied in chunks (`:121-127`) because window tensors are GB-scale.
- Self-checks: per-column identity vs the variant CSV (`:140-151`) and sampled window bit-identity vs the source file (`:154-166`).
- Base-file selection: `glob("*_prepared.h5")` minus `DERIVED_MARKERS = ("remove715", "flank5", "first21", "dim21")` (`:50`, `:186-189`).
- `variant_name()` (`:56-59`) mirrors `config._variant()` (`config.py:28-31`) to name outputs.

**Topic 3 — A/B tooling (3ca725b: `29_ab_dataset_compare.py`, `run_ab_dataset_test.sh`, new `run_rmdupwin_ab.sh`)**

- `src/investigation/29_ab_dataset_compare.py`: new `--variant` flag (`:163-165`, default `remove715negonly`, so defaults are unchanged); module-global `REDUCED_MARKER` reassigned in `main` (`:61`, `:168-169`); suffixes applied to the reduced split CSV, the DeLong TSV, and the plot prefix (`:170-173`, `:220-221`, `:357`); caption text branches per variant (`:370-382`).
- `src/run/run_ab_dataset_test.sh:13-14`: `--vardims default` → `--pca-mode dimselect --top-n default` — **resolves** the carried finding that commit 74351e3's message claimed this switch while no diff contained it.
- New `src/run/run_rmdupwin_ab.sh` (16 lines): sbatch job with `THESIS_VARIANT=remove_dup_windows`, blosum50 arm only (comment: embedding arm appended later); `:7` hardcodes the server repo path — the same carried pattern as the other runners.

## Experiments / Results

Only repository-supported results; observations, not conclusions.

- **No new tracked results in the range** — 3ca725b touches only 6 source files; nothing under `runs/` or `results/` changed, and no `remove_dup_windows` artifact is tracked (`data/processed/` has 0 tracked files at the tip).
- **The tracked A/B table is unchanged and still evidences the vardims arm**: `runs/ab_dataset_test/results/ab_dataset_delong.tsv` — `emb_esmc300m_win_padtoken_vardims_ev80_var643` 0.698220 vs 0.698765 (Δ +0.000545, p_delong 0.8376, ns), blosum50 0.765693 vs 0.766280 (p 0.5402, ns), n_pairs 44,346.
- The variant guard numbers above (1,107 / 304 / 361 / 746) are asserted in code at run time on an untracked input; the repository supports them only as a documented guard, not as reproducible data.
- Pending, not yet in the repo: outputs of `run_rmdupwin_ab.sh` and of the dimselect A/B re-run now configured in `run_ab_dataset_test.sh`.

## Code Review — Shortcomings & Issues

Range findings first (redundant / over-engineered / reinvented-helper first), then carried findings re-verified at 3ca725b.

1. **`variant_name` re-implements `config._variant` (maintainability — redundant)** — `prepare_variant_h5.py:56-59` is a hand copy of `config.py:28-31` (its own docstring says "Mirror config.py _variant()"). `config` is importable; if the two naming rules ever diverge, the subsetter writes files the config redirect never looks for. One import — or a `variant_path(path, variant=None)` helper in config — removes the copy. [3ca725b]
2. **Base-file selection is a hardcoded substring denylist with a silent wrong-source ceiling (risk)** — `prepare_variant_h5.py:50,186-189` treats any `*_prepared.h5` not containing one of four markers as a base file. A derived prepared file whose marker is not in the list and whose row count equals base N would be subset as if it were base; the "identity" checks (`:154-166`) compare the output against the same file it copied from, so they validate the copy, not the source choice → silently wrong tensors. The shape-mismatch path (`:116`) only prints and does not fail, so a wrong-source file with a different row count yields an H5 missing its window tensors with exit 0. Nothing is broken today (variant outputs are named `<stem>_<variant>.h5` and never match the glob). ponytail: substring marker list — upgrade path: whitelist the exact base filenames from config or a small manifest. [3ca725b]
3. **Dead return value (style, low)** — `subset_file` collects `checks` (`:136`) and returns it (`:167`); `main` ignores the result. [3ca725b]
4. **Marker plumbing via mutable module global, plus a stale comment (style, low)** — `29_ab_dataset_compare.py:168-169` reassigns module-global `REDUCED_MARKER` instead of passing `args.variant` to the reader at `:102`; the `:61` comment ("substring `remove715` still matches old runs") no longer describes the marker value (`remove715negonly`). Matching still works because old run JSONs carry full variant paths, but the comment now misleads. [3ca725b]
5. **Half-configured runner (style, low)** — `run_rmdupwin_ab.sh` runs only the blosum50 arm ("embedding arm appended later", `:11-13`) and hardcodes `cd /home/projects/…` at `:7`, the carried runner pattern. [3ca725b]
6. **Carried findings re-verified at 3ca725b, all still open** — the 90 MB blob plus the dead `.gitignore:32` line (no wildcard; `git check-ignore` still exits 1); README drift (`src/investigation/README.md:5` still "33 scripts" vs 67 tracked `.py`/`.R`, only 25 table rows, `:331-332` still "Not run", and `29_ab_dataset_compare.py` — which gained a flag in this range — has no row at all); 16 `spec_from_file_location` loaders and 10 `def delong_test` copies (0 new in range); `predict_with_averaging` still re-derived in `06_external_eval.py`/`14_predict_external.py`; 11 hardcoded timestamps at `30_ae_delong.py` (11 `2026…` filenames); dated globs at `37:236-240`/`38:257-266`; repo-wide `test_*.py` count still 1 (`src/tools/esm/test_embed_esmc6b.py`); `iter_close_pairs` still without a brute-force check; `import glob` still unused at `35_cnn_kernel_performance.py:49`; `umap-learn` still absent from both requirements files (only `README:108` records the pip install) and DiffLogo still absent from `renv.lock` (0 matches); `src/README.md:12` still lists the deleted `01b_datasplit_protein_level.py`; `05_cnn_modelling.py:9` prose unchanged; `28_cnn_compare_barplot.py` still ignores `seed_mode`. [re-verified at 3ca725b]
7. **What passes the ladder (range)** — the variant builder uses a rule registry with per-rule expected-drop guards instead of growing `if/else` sprawl, and keeps its read-only-on-inputs discipline; `29`'s defaults are unchanged so existing invocations still work; the `03_prepare` hazard is documented in `config.py` where the next person will read it; the subsetter reuses h5py/numpy/pandas already declared and chunks GB tensors instead of loading them; no new dependency.

## Potential Issues

1. **90 MB blob in history (carried)** — untouched this range; decide keep-or-drop before the repo travels. Keep → fix or remove the dead `.gitignore:32` line; drop → `git rm --cached` plus a history rewrite, which rewrites every commit SHA.
2. **All variant artifacts are untracked** — `df_all_remove_dup_windows.csv`, `variant_mask_remove_dup_windows.csv`, `df_all_with_folds_remove_dup_windows.csv`, and the subsetted prepared H5s exist only in `data/processed/` (0 tracked files). The 1,107/304/361/746 guard numbers are reproducible only by re-running the script against an untracked `df_all.csv`. Authors' question: should the mask CSV and the drop-count report be tracked as the variant's provenance record?
3. **A/B runner and tracked table now point in opposite directions** — `run_ab_dataset_test.sh:13-14` produces dimselect results while the tracked `ab_dataset_delong.tsv` is a vardims result, and `29` writes no arm suffix for the default variant (`:357`), so the next run overwrites the tracked table in place. Decide whether the vardims table remains the citable record (e.g. re-run into a different `--run-dir`) before executing.
4. **The rmdupwin A/B run has not started (observation)** — the runner exists, no outputs are tracked, and the embedding arm is explicitly deferred in its comment.
5. **Carried open items, not otherwise resolved**: which CNN number family is citable (0.681498 vs 0.684720 vs the 0.6674/0.6536 prose) and the missing `cv_results_*.json` files; variant construction of record (`make_variant_dataset.py` vs `r/04_evaluate_netmhcpan_sensitivity.R`); the ev-sweep recomputed-vs-relabelled question; h2 split inputs still untracked while h2 result tables are tracked; the screening-cache race documented but not prevented (`run_dimselect_sweep.sh:20` — deliberate simplification; a lockfile or a cache key including the variant once multiple submitters matter); NetMHCpan prediction dumps redistributed in-repo.
6. **Process**: 1 commit in the daily range (2026-11-12), 0 on 11-07 and 11-13; local checkout at 26ec48d is **185** commits behind with a clean tree (0 modified files), so every cite above comes from `git show <sha>:<path>`.

## GitHub

From the collected GitHub data (used verbatim, not re-run): no open PRs (`gh pr list` → []), no open issues (`gh issue list` → []), no workflow runs (`gh run list` → []); repository `TheHilarius/thesis2026`, default branch `hilarius`. From git: origin/hilarius = **3ca725b** (2026-11-12), exactly 1 commit ahead of the previous anchor 78d683e in a linear chain (0 merges). Local branch `hilarius` is 185 commits behind with a clean working tree (0 modified files). `reports/overnight/` is gitignored (`.gitignore:134-135`, "Overnight reports — published only to the `reports` branch"), so this report exists only in the local working tree.

## Recommended Next Steps

Carried from the 2026-11-12 report (open unless marked resolved), prioritized, maximum five:

1. **Decide the 90 MB A0101 export (carried, top hygiene item)**: keep it and fix/remove the dead `.gitignore:32` line, or `git rm --cached` + history rewrite while the range is young; record the raw-export policy in `.gitignore`'s "Large data files" section either way.
2. **Run and track the two A/B arms (new; the runner half of the old item 2 is resolved)**: decide the overwrite policy for the tracked vardims `ab_dataset_delong.tsv` (keep via a separate `--run-dir`), then execute the updated `run_ab_dataset_test.sh` (dimselect arms) and `run_rmdupwin_ab.sh` and track their outputs with a run README naming the arm each table evidences.
3. **One README batch (carried)**: fix `:5` (67 tracked `.py`/`.R`), add rows/sections for the 9 undocumented scripts plus `29_ab_dataset_compare.py`'s new `--variant` flag, flip the `:331-332` "Not run" rows, delete `src/README.md:12`'s `01b` line, add `umap-learn` to `requirements_esmc_working.txt`, decide DiffLogo vs `renv.lock`.
4. **Collapse duplication before the next campaign (carried, plus one new copy)**: repair the import chain in `15_padmode_significance.py` (or move DeLong/Holm to one importable module) and delete the 16 `spec_from_file_location` bootstraps; import `predict_with_averaging` in `06_external_eval.py`/`14_predict_external.py`; replace `prepare_variant_h5.variant_name` with an import of `config._variant`; unpin the dated globs at `37:236-240`/`38:257-266` and the 11 filenames at `30_ae_delong.py:52-63`; whitelist base prepared filenames instead of the `DERIVED_MARKERS` substring list; add the `iter_close_pairs` brute-force check (cutoffs 2 and 3, a few hundred random 9-mers).
5. **One provenance pass on the citable numbers (carried)**: track the winning `cv_results_*.json` files (the two k17 C32 runs behind 0.681498/0.684720, the dimselect n643 JSON behind 0.706062) or state that server `runs/` is the record; reconcile `05_cnn_modelling.py:9` with the tracked flank5 table; decide the variant's construction of record and answer the ev-sweep recomputed-vs-relabelled question.

Resolved since the previous report: the A/B runner now matches the stated dimselect intent (`run_ab_dataset_test.sh:13-14`), closing the 74351e3 message-vs-diff mismatch. Still resolved from earlier reports: the trim5 slide logo producer, tracked A/B artifacts, the ESM-IF base ranking CSV, tracked CNN sweep tables, the `02_datasplit_analysis.py` import bootstrap, the `29_build_cluster_map.py` README index, tracked fold-audit tables, a flank5 CNN runner.

## Reviewed Through

3ca725b
