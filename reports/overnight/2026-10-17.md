# Thesis Overnight — 2026-10-17

## Summary

origin/hilarius has not moved: `git ls-remote` shows the branch tip is still a4071c7 (2026-10-15), the same commit the previous report reviewed through, so the range a4071c7..origin/hilarius is empty and no code, data, or results changed. Every carried item was re-verified against the origin blobs and is unchanged, including the 15↔09 import break (now thirteenth report), the zero-byte ESM-C k=17909 log, and the untracked `cv_results_*.json` provenance gap. Two carried questions are now answerable from repository evidence: the three missing `handcrafted*` rows in `padmode_data_compare_runtime.csv` are explained by the plot script's default `--layout padmode` (not silent data loss), and the 2026-10-16 report was published to the `reports` branch (790ff00), confirming the storage convention works.

## Changes

No changes: `git log --oneline a4071c7..origin/hilarius` is empty and `git ls-remote origin hilarius` returns a4071c78d3b1626c40e732a180c3eb64205f5c39 — the remote tip equals the previous report's reviewed-through commit. The local working tree (branch `hilarius` at 26ec48d, 62 commits behind, clean) is also unchanged since the previous report.

## Experiments / Results

No new repository-supported results. Re-verified status at a4071c7:

- ESM-C k=17909: log still zero bytes (`git cat-file -s` on `runs/esmc300m_pca_sweep/logs/flat/04_modelling_xgb_emb_esmc300m_win_padtoken_pca17909_flat_log_20261013_113657.txt` = 0) and still absent from `runs/pca_sweep_delong/pca_sweep_results.csv` (0 of 24 lines match; the CSV now carries a `campaign` column with values `esmc300m_pca_sweep`/`esmif_eos_sweep`, but `pca_sweep_delong.csv` — 319 lines, header `family,k1,k2,...` — still has no campaign column).
- `runs/encodings_sweep/` still contains only its `data` symlink; no MPNN embeddings, runs, audit TSV, or validator output exist in git (`runs/` tree at origin has no mpnn paths).
- `runs/padmode_choice/results/runtime.tsv` has 13 feature rows but `plots/padmode_data_compare_runtime.csv` has 10: the missing rows are exactly `handcrafted`, `handcrafted_sparse`, `handcrafted_blosum`, i.e. everything outside the plot script's default `--layout padmode` = `[sparse, blosum, 8 embedding modes]` (`src/investigation/plot_padmode_data_compare.py:86-97`, `:306`). This resolves the previous report's open question — the drop is a layout selection, and all 13 rows remain in the TSV.
- All AUC/p-values quoted in previous reports stand unchanged.

## Code Review — Shortcomings & Issues

No new code entered in this range, so no new findings; all items below were re-confirmed present at a4071c7 and are ordered per ponytail (redundancy and reinvented wheels first).

1. **DeLong/Holm helpers duplicated verbatim across two files** — `src/investigation/16_esmc300_vs_600.py:52-67` and `src/investigation/pca_sweep_AUC_delong_comparison.py:144-174` each define `delong_structure`/`delong_test` (plus `holm_adjust`/`holm_family` only in the latter, :174/:262). One shared module imported by both deletes ~50 lines and keeps the statistical core single-sourced. Severity: maintainability. [carried]
2. **Peptide locator/window convention copied verbatim a third time** — `src/tools/proteinmpnn/windowing.py:120-180` copies `find_peptide_in_structure` and helpers from `src/tools/esm/embed_peptides_esmif.py`; `src/tools/esm/extract_window_plddt.py:80-102` re-implements the chain-choice rule; `windowing.py:5-21` documents a third window convention. The env constraint (env_mpnn must not install fair-esm) is real, but a dependency-free shared module satisfies it and deletes the copies. Severity: maintainability. [carried]
3. **Validator re-implements the embedder it validates** — `src/tools/proteinmpnn/validate_embeddings.py:44-66` duplicates `load_model`/`tied_featurize` plumbing from `embed_peptides_mpnn.py:117-172` while already importing `encode_protein` from it (:138); importing the loaders too removes ~30 lines and keeps the two in lockstep. Severity: maintainability. [carried]
4. **Four hand-expanded 21-line config dicts** — `src/pipeline/python/config.py:313-391` adds ~84 lines whose entries differ only in checkpoint tag, pad_mode, two paths, and a display name; the same file already generates keys programmatically (`_AA_EMB_PAIRS`, :703). A ~6-line loop over (ckpt, mode) deletes ~70 lines. Severity: maintainability. [carried]
5. **15↔09 import break** — `src/investigation/15_padmode_significance.py:81-83` does `_mod.PADMASK_FILES` / `_mod.PADTYPE_MAP` / `_mod.load_padded_masks` on `importlib.import_module("09_padmode_subset_auc")`, but 09 defines only `load_pad_metadata` (`09_padmode_subset_auc.py:59`) — AttributeError at import, which also breaks the reproduce block in `runs/padmode_choice/README.md:118`. Severity: bug. [carried, thirteenth report]
6. **`esmc6b_pca` label hardcodes n_features==781** — `src/pipeline/python/extract_model_metrics.py:69` (and `extract_pca_results.py:44-47`'s esmc6b-first branch): a 6B run with a different k is silently labelled `esmc6b_full`. Severity: risk. [carried]
7. **Structure audit's guarantee weaker than its docstring** — `src/tools/proteinmpnn/audit_structures.py:50-51` keeps any residue with a CA while the encoder-aligned selection in `embed_peptides_mpnn.py:146` requires N,CA,C,O, yet the docstring (:10-11) calls `atom_match` "what the encoder actually sees"; `classify()` (:79) also takes an unused `seqres_match` param even though residue numbering is a stated positional-mapping requirement. The embedder's own sequence+length check remains the real gate, so this is a decision-surface risk, not silent corruption. Severity: risk. [carried]
8. **Two doc inaccuracies** — (a) `vendor/ProteinMPNN_encoder.patch:16` claims h_V is "post-LayerNorm"; the pinned upstream commit 8907e66's `protein_mpnn_utils.py` contains no `layer_norm` anywhere; (b) `config.py:316` says the 13 isoform-mismatch proteins "are status=3 (zero-filled)", but the embedder assigns status=1 string-search when the peptide is found (`embed_peptides_mpnn.py:293-310`); status=3 is reserved for genuinely absent peptides. Severity: style (interpretation-affecting). [carried]
9. **16 divides a fold-mean AUC difference by a pooled DeLong SD and emits a no-op Holm column** — `src/investigation/16_esmc300_vs_600.py:195-197` (`z = (a3["auc"] - a6["auc"]) / np.sqrt(var_d)` over fold-mean AUCs; `auc_padj = auc_p  # Holm over a 1-pair family`). Severity: risk. [carried]
10. **Carried minor debt, unchanged** — `embed_peptides_mpnn.py:253` assigns `stats = {}` that :345 overwrites; `audit_structures.py:142` binds an unused `e` while parse failures are recorded as no_atoms despite the "note it" comment; `windowing.py:49` `pad_mask_for` is never imported by any consumer; `results/tables/slot_vs_flat_leaderboard.tsv` (:2-5, :27) and `slot_vs_flat_results.tsv` (:39, :44, :49, :54, :125) still carry 5 references each to the renamed-away `runs/padfull_flat_20261001_1645`; `.gitignore` still duplicates `runs/*/models/` (:123/:126) and subsumes `runs/slot_legacy/**/*.json` + `runs/slot_legacy/*/*.json` (:141-142) under `runs/slot_legacy/*/` (:144), and `runs/padmode_choice/results/paired_tests_OLD.csv` is still tracked; `src/run/run_pca_sweep.sh:2` still documents the nonexistent `run_pca_k.sh`; `src/investigation/README.md` still documents the deleted `10_pad_subset_auc.py` (7 references, :13/:34/:140/:142/:242/:256/:283 — the file is absent from `src/investigation/`), and neither `pca_sweep_AUC_delong_comparison.py` nor `run_encodings_sweep.sh` nor the `proteinmpnn/` tools appear in any README. Severity: maintainability/style. [carried]

## Potential Issues

1. **Provenance chain for every quoted number (blocking, carried)** — 0 `cv_results_*.json` files are tracked; `.gitignore:124` ignores `runs/*/models/`, and the sweep CSVs derive from those JSONs. `data/` remains untracked, `runs/encodings_sweep/` still contains only its symlink, and the ProteinMPNN feature still has zero committed evidence of execution.
2. **MPNN consumption path (question for the authors, carried)** — the four config sources are wired but nothing consumes them: no FEATURE_SETS entry, no run script, no analysis (grep for `mpnn` outside `config.py`/`run_embeddings_mpnn_slurm.sh`/`vendor/` returns nothing). Is an MPNN modelling arm planned, and which comparison does the v_48_002-vs-v_48_020 sensitivity arm serve? The two-checkpoint design doubles embedding compute for a comparison no committed script defines.
3. **Operating-point decision (carried, sharpened)** — unchanged: ESM-C best and optimum-rule coincide at k=2278 (80 % EV) with a knife-edge margin over k=1403 (Holm p_adj=0.044); ESM-IF optimum rule flags k=632 while best is k=850, mutually indistinguishable with k=1143. Interpretation remains the authors'.
4. **Non-monotonic fold-seconds (carried) still unexplained**, and ESM-C k=17909 (99 % EV) is still pending — zero-byte log, absent from the results CSV, job id known only from a source comment.
5. **ESM-C 6B scope (carried)** — wired into the variance analysis and both metric extractors, but no 6B H5, embeddings, or runs exist in git; the variance script's exists-filter means the new window set silently disappears when the file is absent (no warning). What k will 6B use — `extract_model_metrics.py:69` assumes 781.
6. **Structure-audit semantics (question, carried)** — should the audit's CA-only residue rule be aligned with the encoder's complete-backbone rule before its TSV is cited as evidence that positional mapping is safe?
7. ~~Carried: confirm the silent drop of three `handcrafted*` rows from `padmode_data_compare_runtime.csv`~~ — **resolved** (see Experiments): default `--layout padmode` selection, data intact in `runtime.tsv`.
8. **Cluster paths hardcoded in run scripts (carried)** — `src/run/run_embeddings_mpnn_slurm.sh:26` sets `PROJECT_DIR="/home/projects1/thesis_s204692_s204581/thesis2026"`; same pattern in the other run scripts.

## GitHub

From the collected GitHub data (not re-run):

- **PRs:** none open (`gh pr list` returned []).
- **Issues:** none open (`gh issue list` returned []).
- **CI:** no workflow runs (`gh run list` returned []); no workflows configured.
- **Branch/state:** `TheHilarius/thesis2026`, default branch `hilarius`. From git: origin/hilarius = a4071c7, unchanged since 2026-10-15 17:19 (verified via `git ls-remote`); local branch HEAD 26ec48d, 62 commits behind, working tree clean. origin/reports advanced to 790ff00 "docs: overnight thesis report 2026-10-16" (2026-10-16 05:10), so the 2026-10-16 report is published and `reports/overnight/latest.md` exists on that branch; `reports/overnight/` remains gitignored on the code branch (`.gitignore:115`).

## Recommended Next Steps

1. Repair the 15↔09 break and consolidate once: move `PADMASK_FILES`/`PADTYPE_MAP`/`load_padded_masks` plus the DeLong/Holm helpers into one small module imported by 09/15/16 and `pca_sweep_AUC_delong_comparison.py`, and put the peptide locator/window convention in one dependency-free module used by the ESM-IF, ESM-C, and ProteinMPNN paths (Code Review 1-2, 5); then fix `runs/padmode_choice/README.md:118`. [carried, top priority]
2. Smoke-test the ESM-IF fix: `python src/tools/esm/embed_peptides_esmif.py --dry-run`, then a small `--n-debug` run on the server — 72f2984 restored the call (re-verified at :517) but no successful run output exists in git. [carried]
3. Decide the MPNN consumption path before writing more tooling (Potential Issue 2): if an arm is planned, add the FEATURE_SETS/`--combos` entry and one run script; in the same commit fix the two doc inaccuracies (patch "post-LayerNorm", config status=3) and the `n_features==781` magic. [carried]
4. Make the numbers quotable: commit the sweep and padmode_choice `cv_results_*.json` files (`git add -f`, the dirs are ignored) or record the rsync; sed the dead `padfull_flat_20261001_1645` paths in the two `slot_vs_flat` TSVs; add a `campaign` column to `pca_sweep_delong.csv` (its sibling `pca_sweep_results.csv` already has one). [carried, refined]
5. One housekeeping commit: fix `run_pca_sweep.sh:2` header filename and the hardcoded cluster paths; dedupe `.gitignore` (:123/:126, :141-144); delete `paired_tests_OLD.csv`; index `pca_sweep_AUC_delong_comparison.py`, `run_encodings_sweep.sh`, and the `proteinmpnn/` tools in the READMEs and delete the `10_pad_subset_auc.py` sections; dedupe the validator's copy of the embedder loaders; align or soften the audit's residue-selection claim; when k=17909 lands, record its result or delete its zero-byte log. [carried]

Resolved since the previous report: the `padmode_data_compare_runtime.csv` row drop is explained (default padmode layout, TSV intact); publication of the 2026-10-16 report to origin/reports (790ff00) confirms the report-storage convention works end to end. Nothing else carried was fixed, and no new commits landed.

## Reviewed Through

a4071c7
