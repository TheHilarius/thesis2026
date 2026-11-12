# Thesis Overnight — 2026-11-12

## Summary

Range 25769e6..origin/hilarius = **22 commits, all on 2026-11-11** (550 files, +317,517/−303; of that, 42 source files +8,835/−181 and 508 tracked run/data files +308,682/−122). This is the first range that ships tracked results alongside code: CNN sweep tables, flank5 / AE-capacity / dimselect / model-compare / A-B / ROC-crossing outputs, and an Assarsson external-validation metrics JSON. It also adds a full HLA-A*01:01 pipeline edition with a 90 MB raw IEDB export, and promotes supervised dimselect to the default embedding arm (`--top-n default`). The anchor advances 25769e6 → **78d683e**; the local checkout is 184 commits behind with a clean tree. Three carried items are resolved (trim5 logo producer, tracked A/B artifacts, missing ESM-IF base ranking CSV); the README-drift and duplication findings got worse.

## Past 7 Days

Calendar window 2026-11-06..2026-11-12: 35 commits. Commits per day (count then date):

      6 2026-11-06
      1 2026-11-08
      4 2026-11-09
      2 2026-11-10
     22 2026-11-11

2026-11-07 and 2026-11-12 have **0** commits; the most recent day with activity is 2026-11-11. Theme level only (file:line detail stays under ## Changes):

- **CNN campaigns move from runners to tracked output (11-10 → 11-11)** — kernel / k2 / channel / 2-layer-channel summary, DeLong and overfit tables plus learning-curve files land under `runs/cnn/` (previously 0 tracked files).
- **Results-first analysis batch (11-11)** — flank5 19mer-vs-29mer paired significance, AE capacity DeLong, supervised-dimselect screening at matched EV budgets, XGB-vs-simpler-models comparison, A/B dataset outputs, random-peptide logo null model, ROC-crossing screen, flank duplicate-leakage audit.
- **Two external-validation efforts (11-11)** — the Assarsson A0201 set (FASTA fetch, feature prep, evaluation, tracked metrics) and a parallel HLA-A*01:01 pipeline edition (raw IEDB export, FASTA fetch, NetMHCpan runners, R chain, embedding prep, external prediction).
- **Pipeline default change (11-11)** — supervised dimselect becomes the default embedding-selection arm, with `'default'` top-N resolved from config per embedding.
- **Sweep tooling and figure refresh (11-11)** — `--campaign` flags on the encodings/handcrafted sweeps, `--variant` flag on the dimselect sweep runner, gitignore entries for raw dumps/logs, regenerated PCA/variance figures, a trim5 slide logo, and an MPNN booktabs table.

## Changes

Range 25769e6..78d683e = 22 commits, linear (0 merges). All cites are `git show <sha>:<path>` (the local checkout predates the range).

**Topic 1 — supervised dimselect as default embedding arm (74351e3)**

- `src/pipeline/python/config.py:69-77`: new `DEFAULT_DIMSELECT_N = {"esmc300m_win_padtoken": 643, "esmif_eos_bos_repeat": 250}` with the supporting numbers in the comment (0.7061 vs 0.7018, DeLong Holm p<0.001; 0.6504 vs 0.6468, p<0.01).
- `src/pipeline/python/04_modelling.py:46` imports it; `:1524-1531` changes `--top-n` from `type=int` to `type=str` accepting `'default'`; `:1603-1619` resolves `'default'` via the single embedding component of the feature set (raises on ambiguity or unknown budget), else `int()`. Existing invocations are unaffected (default stays `None`).
- The commit message says `run_ab_dataset_test.sh` "switches its two arms … to `--pca-mode dimselect --top-n default`"; the commit touches only the two files above, and the tip still reads `--vardims default` (`src/run/run_ab_dataset_test.sh:13-14`).

**Topic 2 — Assarsson A0201 external validation (c63db75)**

- New `src/util/fetch_assarsson_fastas.py` (86 lines): UniProt REST fetch keyed on VACWR locus codes; writes `assarsson_orf_accession.csv`.
- New `src/pipeline/r/10_assarsson_preparation.R` (291): builds `df_assarsson.csv` + `df_blosum50_assarsson.csv`; label decision (Dominant/Subdominant/Cryptic → positive) recorded at `:15`.
- New `src/pipeline/python/06_external_eval.py` (198): loads run pickles and averages `predict_proba` (docstring `:4` states it mirrors `04_modelling.predict_with_averaging`); fixed `FEATURES_CSV`/`META_CSV` paths (`:36-37`), threshold 0.5, refuses scaled/PCA feature sets (`:8-9`).
- Tracked I/O: 7 files under `data/raw/external/assarsson/` (~122k lines; `NetMHCpan_BAmode_assarsson_alltext.txt` 4.25 MB) plus `results/external/assarsson_xgb_blosum50_flat_metrics.json` and `_predictions.csv`.

**Topic 3 — HLA-A*01:01 pipeline edition (c16d8ba, 9ee1a04)**

- New `src/pipeline_a0101/` (12 files, ~3.6k lines): `README.md` (edition contract — legacy pipeline untouched, outputs under `data/processed/hla_a0101/`), `fetch/fetch_a0101_fastas.py` (130), `netmhcpan/run_netmhcpan.sh` (82) + `run_netmhcpan_parallel.sh` (211, "edition of legacy `src/tools/netmhcpan/…`", `:4`), `python/config.py` (123-line trimmed fork of the 1,272-line pipeline config), `python/03_prepare_embeddings.py` (1,002), `python/14_predict_external.py` (350), R chain `02/04/05/06/07/08` + `functions.R` (78) + `run_pipeline.R` (109).
- Fork distance (`git diff --no-index`): `03_prepare_embeddings.py` vs legacy = +9/−20 lines out of ~1,000; `r/04_evaluate_netmhcpan_sensitivity.R` (1,207) vs legacy (1,189) = +60/−42.
- New tracked raw input `data/raw/mhc_ligand_table_export_HLA-A0101.csv` (55,184 lines; blob 90,263,530 bytes). `.gitignore:19` adds `data/processed/hla_*/`; `.gitignore:32` (9ee1a04) adds `data/raw/mhc_ligand_table_export_HLA-A0101` — no wildcard, so it does not match the `.csv` (`git check-ignore` exits 1) and could not untrack the already-committed file anyway.

**Topic 4 — seven new analysis scripts, one plot script, one README section (222c86c, a70684e, 1351bfb, b152039, 24e009c, f916b65, 953bb40, 78d683e, 03ded4c, b5bcba9)**

- `31_model_compare.py` (409 lines, +96 in 78d683e): XGB vs rf/lr_l2/lr_elasticnet on blosum50 and handcrafted_blosum50; DeLong/Holm loaded from `19_handcrafted_sweep_delong.py` via `spec_from_file_location` (`:55-60`; docstring `:31` calls it "repo convention"); AUC labels on CV bars (78d683e); bar scale/bracket fix (03ded4c, 25-line diff).
- `40_roc_crossing_check.py` (543): screens every DeLong-producing run for ROC crossings; reuses each script's ARMS/FAMILIES and `15`'s `delong_test` (`:124`, `:275`); writes `runs/<tag>/results/roc_crossings.tsv` into ~15 run directories.
- `31_random_peptide_logo.py` (261) + `31_random_peptide_logos.R` (248): score 1M random 29-mers with the baseline XGB's 30 inner pickles; the 1M-row score dumps are gitignored as reproducible (`.gitignore:89-93`), PPMs/logos tracked.
- `32_flank5_significance.py` (260) + `plot_flank5_compare.py` (188): paired DeLong / bootstrap / t-test family for 19mer vs 29mer (loads `15` dynamically, `:55`) plus bar+ROC figure; new `runs/flank5_compare/README.md` (90 lines) records derivation, experiment spec, and the caveat that structural flank features were not re-derived for 19mer.
- `30_ae_delong.py` (156) + `30_ae_barplot.py` (192): 11 AE capacity arms vs var643; the arm list hardcodes 11 timestamped JSON filenames (`:52-63`).
- `30_dimselect_ev_bars.py` (452): dimselect vs topX-variance bars at matched EV budgets with paired-DeLong brackets; sources span three run trees (docstring `:24-27`).
- `41_flank_duplicate_leakage_audit.py` (355): read-only duplicate/cross-fold flank audit — the only new script that got a README section (`src/investigation/README.md:89-94`).
- Numbering collisions widened: three 30s, three 31s, two 32s (README `:5` documents the "two 09s" chronological convention).

**Topic 5 — sweep parameterization and figure work (f1b9604, 0e32ee3, 8329e7e, 44452ac)**

- `18_encodings_sweep_delong.py`: `--campaign` reroutes all outputs into `runs/<campaign>/` (`:20-30`), `read_mem` takes the campaign, `REDUCTION_NOTES` keyed by feature count gives truthful legend notes (`:223-232`), flat figure cut to 13 ESM-C-only bars, y-label → "CV-AUC".
- `19_handcrafted_sweep_delong.py`: `--campaign` / `--ref-campaign` (`:567-574`), scans both trees, `read_mem` falls back across campaigns (`:197-207`).
- `src/run/run_dimselect_sweep.sh` (+20/−8): `--variant` flag defaulting to flank5; header warns the screening-cache filename carries no variant and that "SUBMISSION ORDER MATTERS (no lock around the cache)" (`:19-21`).
- `12_sequence_logos.R:104-121`: generates `differential_logo_29mer_trim5.png` — the previously orphaned slide logo now has a producer.
- `16_mpnn_padmode_plot.py:199-242`: `build_padmode_tex` emits a booktabs `.tex` for the padmode campaign.
- 44452ac: regenerates PCA/variance-sweep figures and `runs/pca_sweep_delong/*.csv`.

**Topic 6 — tracked run outputs and new runners (0cc8dbb, ca13187, 12fea2e, b152039, 5db3620)**

- `runs/cnn/` gains summary/DeLong/overfit TSVs and learning-curve CSVs/PNGs for the kernel, k2, channel and 2-layer-channel campaigns; `runs/dimselect_matched/` (66 screening CSVs of 961/513 rows plus curve/DeLong TSVs), `runs/flank5_compare/`, `runs/autoencoder_pilot/`, `runs/ab_dataset_test/` (previously 0 tracked files), `runs/investigate_diff_logo_pattern_w_simpler_models/`, `runs/random_peptide_logo{,_a02}/`, and `runs/ev_matched_topXvariance_sweep/results/roc_crossings.tsv` (1,486 rows) are all new.
- New runners: `run_ab_dataset_test.sh`, `run_flank5_compare.sh`, `run_flank5_esmc.sh`, `run_flank5_quick.sh`, `run_helping_v2.sh` — each hardcodes `cd /home/projects/thesis_s204692_s204581/thesis2026` (e.g. `run_ab_dataset_test.sh:7`).
- `.gitignore` +16 (5db3620, c16d8ba, 9ee1a04): raw IEDB dumps, `runs/*.log`, random-peptide score dumps, `data/processed/hla_*/`, and the non-matching A0101 line.
- `results/tables/dim_variance_esmif_eos_bos_repeat.csv` (513 rows) added — the base ranking CSV missing at 25769e6.

## Experiments / Results

Only repository-supported results, read directly from tracked files at 78d683e; observations, not conclusions.

- **flank5 (19mer) vs 29mer — all 4 pairs favor flank5**: pooled deltas +0.00681 (BLOSUM50), +0.00502 (handcrafted), +0.00573 (ESM-C EV80), +0.01450 (CNN k5); DeLong Holm p from 1.35e-14 to 7.55e-32, all `***` (`runs/flank5_compare/results/paired_tests.csv`); outer-fold means in `runs/flank5_compare/README.md` (e.g. BLOSUM50 0.7633 → 0.7701). The same README records that structural flank features were not re-derived for 19mer — authors' call whether that qualifies the handcrafted pair.
- **dimselect vs topX-variance at matched budgets**: ESM-C n643 0.706062 vs 0.701727 (Δ +0.004335, Holm 2.86e-05 `***`), n793 ns; ESM-IF n122 and n260 `***`, n429 ns; dimselect also beats flat PCA at every tested budget, and both embedding arms trail blosum50 by 0.057/0.113 (`runs/dimselect_matched/results/dimselect_delong_pairs.tsv`). This is the table behind the `config.py:71-76` comment.
- **EV-ladder dimselect vs the EV80 variance arm**: only EV80 (n643) and EV85 significantly beat var643; EV30/EV40 significantly worse; EV50–EV75 and EV90 ns (`runs/dimselect_matched/results/dimselect_vs_vardims_ev80.tsv`).
- **AE capacity never reaches var643**: all 11 arms 0.6550–0.6936 vs 0.701727, every Holm p < 1e-8 (`runs/autoencoder_pilot/results/ae_delong_pairs.tsv`).
- **XGB vs simpler models**: xgb 0.763254 / 0.766513 (handcrafted) vs rf 0.713908 / 0.715752 vs lr_l2 0.704515 / 0.709575 vs elasticnet ≈ lr_l2; every contrast `***`/`**`, including simple-vs-simple (rf > lr) and a tiny but significant lr_l2-vs-elasticnet gap (`runs/investigate_diff_logo_pattern_w_simpler_models/results/model_compare_delong.tsv`, `_metrics.tsv`).
- **A/B dataset, full vs reduced variant: ns on both arms** — blosum50 0.765693 vs 0.766280 (p 0.54), vardims 0.698220 vs 0.698765 (p 0.84) (`runs/ab_dataset_test/results/ab_dataset_delong.tsv`).
- **CNN sweep tables now tracked**: kernel curve k3→k19, best k17 = 0.681498; k2 curve best k19 = 0.679415 with a 1-layer C32 reference row of 0.684720; channel curve c8→c128, best c32 = 0.684720 (`runs/cnn/*/results/*_summary.csv`). Observation: two different pooled AUCs exist for the same model key `cnn_simple_k17` c32 — 0.681498 (run 20261109_224446) and 0.684720 (run 20261110_115910, used as the k2 reference). Authors' question: which run is the reference? The `cv_results_*.json` files named in the `file` column remain untracked (0 in repo).
- **Assarsson A0201 external validation**: AUROC 0.6667 (n=104, 65 pos / 39 neg), 0.7406 on the 9-mer subset (n=73); accuracy 0.5385 / 0.5753; mean probability lowest on Negatives (0.2385) and highest on Dominant (0.5066) (`results/external/assarsson_xgb_blosum50_flat_metrics.json`). Small n, one model, no CI in the file — not a validated performance claim.
- **ROC-crossing screen**: of the tracked `roc_crossings.tsv` files only `runs/padmode_choice` contains `critical` pairs — 2, both subset_ESM-C flat-vs-sparse (p_delong 0.0366 and 0.00239, crossing flagged); every other tracked run is clean/weak/noise only.
- **Flank duplicate-leakage audit** (README `:94` + tracked CSVs): 1,690 duplicate 9-mers, all fold-contained (0 cross-fold); 476 mixed-label groups, 260 with identical input context (581 rows = 1.25%); 20 cross-fold identical-flank pairs (0.09% of rows); 5 held-out rows share a train flank pair; 0 Hamming ≤1 violations.

## Code Review — Shortcomings & Issues

Range findings first (redundant / over-engineered / reinvented-stdlib first), then carried findings re-verified at 78d683e.

1. **90 MB raw dump committed, then an ignore line that cannot work (bug — dead config + permanent bloat)** — `data/raw/mhc_ligand_table_export_HLA-A0101.csv` (blob 90,263,530 bytes) added in c16d8ba; 9ee1a04 then adds `.gitignore:32` `data/raw/mhc_ligand_table_export_HLA-A0101`, a pattern with no wildcard that does not match the `.csv` (`git check-ignore` exits 1) and cannot untrack an already-tracked file regardless. Both halves of the stated intent ("Ignoreing new large HLA0101 file") fail; the blob stays in history absent a rewrite. [c16d8ba, 9ee1a04]
2. **~1,000-line and ~1,200-line verbatim forks of pipeline files (maintainability — redundant)** — `src/pipeline_a0101/python/03_prepare_embeddings.py` (1,002 lines) differs from `src/pipeline/python/03_prepare_embeddings.py` (991) by +9/−20 lines; `src/pipeline_a0101/r/04_evaluate_netmhcpan_sensitivity.R` (1,207) differs from legacy (1,189) by +60/−42 (`git diff --no-index`). Path redirection is already this codebase's pattern (the a0101 README itself cites config's `THESIS_VARIANT` path mechanism), so these two needed a paths argument rather than a fork; every legacy fix now needs a manual mirror. `python/config.py` (123 vs 1,272 lines) is a trimmed copy of the same class. [c16d8ba]
3. **`predict_with_averaging` re-derived twice more instead of imported (maintainability — redundant)** — `src/pipeline/python/06_external_eval.py:4` and `src/pipeline_a0101/python/14_predict_external.py:11` both say they "mirror `04_modelling.predict_with_averaging`" and re-code the average-over-pickles loop. `04_modelling.py` is importable (main guarded); re-deriving shared logic at the external-validation trust boundary is where drift turns into a wrong number. [c63db75, c16d8ba]
4. **Dynamic-import bootstrap grew from 9 to 16 files (maintainability — carried, worse)** — `git grep -l spec_from_file_location origin/hilarius -- src/` now returns 16 files; the range adds loaders in `30_ae_delong.py:36-45`, `30_dimselect_ev_bars.py:80`, `31_model_compare.py:55`, `32_flank5_significance.py:55`, `40_roc_crossing_check.py:124` to reach `15_padmode_significance.py` / `19_handcrafted_sweep_delong.py`, because numbered filenames are not importable module names. `def delong_test` still exists in 10 files, and `15_padmode_significance.py` was never touched in the range — the named root cause (repair that import chain once, or give the stats one importable module) is still undone, and every new analysis pays the bootstrap tax. [222c86c, a70684e, 1351bfb, b152039, 24e009c, f916b65, 953bb40]
5. **README drift widened (risk — carried, worse)** — `src/investigation/README.md:5` still says "33 scripts (6 R, 27 Python)" while the directory tracks **67** `.py`/`.R`; the only new section is `41_` (`:89-94`), so 9 of the 10 new scripts (`30_ae_barplot`, `30_ae_delong`, `30_dimselect_ev_bars`, `31_model_compare`, `31_random_peptide_logo{,.R}`, `32_flank5_significance`, `40_roc_crossing_check`, `plot_flank5_compare`) have no section and no reproducibility row; `:331-332` still mark 33/34 "Not run" although their tables are tracked. A reader trusting the README would report this entire range's analysis work as unwritten. [README touched only by 222c86c in the range]
6. **Commit message claims a runner change that is not in any diff (risk)** — 74351e3's message says `run_ab_dataset_test.sh` "switches its two arms from `--vardims default` to `--pca-mode dimselect --top-n default`", but the commit touches only `04_modelling.py` + `config.py`, no commit in the range touches that runner, and the tip still reads `--vardims default` (`src/run/run_ab_dataset_test.sh:13-14`) — consistent with the tracked A/B table, whose embedding arm is `..._vardims_ev80_var643`. Either the switch was forgotten or the message overstates it; as committed, "dimselect is the go-to arm" (`config.py:71-72`) and "the A/B test runs vardims" disagree. [74351e3]
7. **Hardcoded timestamped arm filenames (risk — extends a carried pattern)** — `30_ae_delong.py:52-63` pins 11 AE JSON filenames with literal timestamps; any new capacity run requires editing the script. Same class as the carried dated auto-find globs at `37:236-240` / `38:257-266`, which the range left untouched. [24e009c]
8. **`--top-n` made stringly-typed (style)** — `04_modelling.py:1524` changes `type=int` → `type=str` so `'default'` passes through, and validation/`int()` moves into `__main__` (`:1603-1619`), so argparse no longer rejects a malformed number at parse time (it fails later with `ValueError`). A `type=` callable (`'default'` → sentinel, else `int`) keeps parse-time validation in one line. [74351e3]
9. **Misspelled campaign directory baked into committed paths (style, low)** — `runs/cnn/cnn_2layer_kernal2_curve/…` ("kernal") is now the path of tracked outputs and appears in `37`'s usage text; renaming later breaks every committed reference. [c16d8ba]
10. **Carried findings re-verified at 78d683e, all still open** — `import glob` unused at `35:49`; dated reference globs at `37:236-240`/`38:257-266`; `runs/*/models/` ignore line triplicated (`.gitignore:141`, `:144`, `:176`); `iter_close_pairs` (`src/pipeline/python/01_datasplit.py:99`) still has no brute-force check — repo-wide test count still 1 (`src/tools/esm/test_embed_esmc6b.py`); `src/README.md:12` still lists the deleted `01b_datasplit_protein_level.py`; `umap-learn` absent from both requirements files (README `:108` records only a pip install); DiffLogo still absent from `renv.lock`; `significant_05` raw-p flag beside `p_BH` (`13_difflogo_significance.R:211`); dead `load_delong_pick` (`22_pool_barplot.py:100`); `variant_path` still mirrors `config._variant` (`reduce_29mer_to_19mer.py:88`); `05_cnn_modelling.py:9` prose (0.6674/0.6536) unchanged; `28_cnn_compare_barplot.py` still ignores `seed_mode`.
11. **What passes the ladder (range)** — DeLong/Holm is loaded, not re-pasted, in all five new stats scripts (the copy count did not grow); `18`/`19` gained `--campaign`/`--ref-campaign` parameters instead of forked copies; the random-peptide script gitignores its 1M-row dumps as reproducible; `40` reuses each analysis script's own ARMS/FAMILIES rather than re-listing pairs; no new Python dependency (numpy/pandas/scipy/sklearn/matplotlib/h5py already declared); `32` runs three test families and applies Holm to each rather than picking one; the flank5 run README records the one real caveat instead of hiding it.

## Potential Issues

1. **90 MB blob in history (observation → authors' decision)**: decide keep-or-drop before the repo travels. If kept, delete or correct the dead `.gitignore:32` line (e.g. `data/raw/mhc_ligand_table_export_HLA-A0101*.csv` — useful for future exports, moot for this one); if dropped, `git rm --cached` plus a history rewrite is the only fix, and that rewrites every commit SHA.
2. **External-validation provenance is half-tracked**: Assarsson raw inputs and the metrics JSON are tracked, but the intermediates `data/processed/df_assarsson.csv` / `df_blosum50_assarsson.csv` consumed by `06_external_eval.py:36-37` are not (`data/processed/` has 0 tracked files), and the 30 pickles averaged by that JSON live at an absolute path from another machine (`run_dir: /home/hilarius/…` in `assarsson_xgb_blosum50_flat_metrics.json`). Authors' question: is rerunning `10_assarsson_preparation.R` the documented path to the tracked metrics?
3. **Three CNN number families, one prose claim**: `cnn_2layer_k2_summary.csv` reference row 0.684720 vs `cnn_kernel_summary.csv` k17 0.681498 for the same model key, while `05_cnn_modelling.py:9` still justifies the protocol with 0.6674/0.6536 and the tracked flank5 README reports 0.6688/0.6543 (pooled). Which number is the thesis claim, and where is its `cv_results` JSON (0 tracked)?
4. **Runner/commit-message disagreement on the A/B arms** (review 6): until `run_ab_dataset_test.sh:13-14` matches the stated intent, the tracked A/B table is a vardims result and cannot evidence a dimselect claim.
5. **Screening-cache race is documented, not prevented**: `run_dimselect_sweep.sh:20` warns "SUBMISSION ORDER MATTERS (no lock around the cache)" — fine while one person submits sequentially; concurrent submits can mix rankings. Deliberate simplification with a known ceiling (a lockfile, or a cache key that includes the variant, once multiple submitters matter).
6. **Carried open items, not otherwise resolved**: variant construction of record (`make_variant_dataset.py` vs `r/04_evaluate_netmhcpan_sensitivity.R`, both untouched); the ev-sweep truth question (`variance_sweep_vs_best_delong.tsv:21` still `best_vs_raw … 0.648827 vs 0.649814 … ns` — the missing base ranking CSV is now tracked, the recomputed-vs-relabelled answer is not); which CNN protocol axis is citable (`mode`, `seed_mode`, channels, depth); h2 split inputs still untracked while h2 result tables are tracked; the "Not run" README rows for 33/34 unchanged.
7. **Process**: all 22 commits landed on 2026-11-11; local checkout `hilarius` at 26ec48d is **184** commits behind with a clean working tree (0 modified files), so every cite above comes from `git show <sha>:<path>`. NetMHCpan prediction dumps (4 files, ~122k rows, up to 4.25 MB) are now redistributed in-repo — authors' call whether that is intended.

## GitHub

From the collected GitHub data (used verbatim, not re-run): no open PRs (`gh pr list` → []), no open issues (`gh issue list` → []), no workflow runs (`gh run list` → []); repository `TheHilarius/thesis2026`, default branch `hilarius`. From git: origin/hilarius = **78d683e** (2026-11-11), 22 commits ahead of the previous anchor 25769e6 in a linear chain (0 merges) 25769e6 → … → 78d683e. Local branch `hilarius` is 184 commits behind with a clean working tree (0 modified files). `reports/overnight/` is gitignored (`.gitignore:134-135`, "Overnight reports — published only to the `reports` branch"), so this report exists only in the local working tree.

## Recommended Next Steps

Carried from the 2026-11-11 report (all open unless marked resolved), prioritized, maximum five:

1. **Decide the 90 MB A0101 export (new, top hygiene item)**: keep it and fix/remove the dead `.gitignore:32` line, or `git rm --cached` + history rewrite while the range is young; record the raw-export policy in `.gitignore`'s "Large data files" section either way.
2. **Make the A/B runner match the stated dimselect intent (new, blocks an interpretation)**: change `run_ab_dataset_test.sh:13-14` to `--pca-mode dimselect --top-n default` and re-run the two embedding arms, or correct the 74351e3 claim — and state in a run README which arm the tracked A/B table evidences.
3. **One README batch (carried, now covering 10 new scripts)**: fix `:5` (67 tracked `.py`/`.R`), add sections + reproducibility rows for the 9 undocumented scripts, flip the `:331-332` "Not run" rows, delete `src/README.md:12`'s `01b` line, add `umap-learn` to `requirements_esmc_working.txt`, decide DiffLogo vs `renv.lock`.
4. **Collapse duplication before the next campaign (carried, now 16 loaders)**: repair the import chain in `15_padmode_significance.py` (or move DeLong/Holm to one importable module) and delete the 16 `spec_from_file_location` bootstraps; import `predict_with_averaging` in `06_external_eval.py`/`14_predict_external.py`; unpin the dated globs at `37:236-240`/`38:257-266` and the 11 filenames at `30_ae_delong.py:52-63`; add the `iter_close_pairs` brute-force check (cutoffs 2 and 3, a few hundred random 9-mers) in `01_datasplit.py`.
5. **One provenance pass on the citable numbers (carried, extended)**: track the winning `cv_results_*.json` files (the two k17 C32 runs behind 0.681498/0.684720, the dimselect n643 JSON behind 0.706062) or state that server `runs/` is the record; reconcile `05_cnn_modelling.py:9` with the tracked flank5 table; decide the variant's construction of record and answer the ev-sweep recomputed-vs-relabelled question.

Resolved since the previous report: the trim5 slide logo now has a producer (`12_sequence_logos.R:104-121`); A/B artifacts are tracked (`runs/ab_dataset_test/`, previously 0 files); the missing ESM-IF base ranking CSV exists (`results/tables/dim_variance_esmif_eos_bos_repeat.csv`); CNN sweep outputs are now tracked as tables and figures (their `cv_results` JSONs still are not). Still resolved from earlier reports: the `02_datasplit_analysis.py` import bootstrap, the `29_build_cluster_map.py` README index, tracked fold-audit tables, a flank5 CNN runner.

## Reviewed Through

78d683e
