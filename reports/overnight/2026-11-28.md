# Thesis Overnight — 2026-11-28

## Summary

The reviewed range `b0d3908..origin/hilarius` is **empty**: `git ls-remote origin hilarius` and `git rev-parse --short origin/hilarius` both give `b0d3908`, the same tip the 2026-11-27 report anchored on, so no commits landed since and `## Changes` has nothing to list. The seven-day window is likewise unchanged — 19 commits, all on 11-22/23/24/26, none on 11-27 or 11-28. Every open finding from the previous report was re-verified against the tip and still reproduces, with one exception: the reported fetch-resume / shared-log mismatch (previous finding 18) does **not** reproduce — the resume loop reads both logs. No new results, no CI, no PRs or issues.

## Past 7 Days

Calendar window 2026-11-22..2026-11-28: 19 commits. Commits per day (count then date):

      1 2026-11-22
      3 2026-11-23
      5 2026-11-24
     10 2026-11-26

0 commits on 11-25, 11-27 and 11-28; most recent day with activity is 2026-11-26 (re-derived from `git log --format=%ad --date=short origin/hilarius | sort | uniq -c`, total 19 — matches the stated counts). This is the same commit set as the previous report's window (11-21..11-27); 11-28 added nothing. Theme level only, no file:line detail (that belongs under ## Changes):

- **Optuna tuning-campaign hardening and a second search design (11-22 → 11-26)** — per-arm `--features`, study-creation race retry and a venv install fix for the XGB TPE scripts (11-22), then a `--full-space` mode that puts `learning_rate`/`n_estimators` inside the study instead of the two-phase polish design (11-26).
- **HLA-A*01:01 external-validation wiring (11-23)** — renamed NetSurfP slurm script, external-data script extended for the new allele, netMHCpan sbatch receipt wrapper.
- **Run-format and run-environment hardening for the a0101 rerun (11-24)** — forced column types + fail-loud problem guard for the netMHCpan 4.2f preamble, CLES-based bin-count selection, NetSurfP sbatch receipt wrapper, per-allele fetch workspace; the external scoring script grew a `--dataset assarsson` mode with prerequisite pre-flights.
- **Neural-net family day (11-26)** — an FFNN family added beside the CNN family, stages renamed to lettered files, default CV switched to nested on a rewritten fold assignment, and Optuna tuners/plotter/launchers built for both neural nets.
- **Thesis-figure and result hygiene (11-26)** — caption sidecar files instead of in-figure footnotes, stable dataset colors, a ggseqlogo clipping fix, and the first tracked model-comparison result tables for the 46,617-row split.

## Changes

**None — the range is empty.** `git log --oneline b0d3908..origin/hilarius` → no output; `git rev-parse --short origin/hilarius` → `b0d3908`; `git ls-remote origin hilarius` → `b0d3908b40154b7f50a3e984911342e7fbd93a94` (the remote has not moved). The 10 commits that made up the previous report's range are all dated 2026-11-26 and are already covered there; there is nothing new to list and nothing to repeat.

## Experiments / Results

Range empty, so no new results; nothing was re-executed this run (`command -v Rscript` → none; script execution blocked in unattended mode). Repository-supported artifacts are byte-identical to the previous report's state:

- Still the only tracked numeric results: `runs/modelcompare_29mer_46617/results/model_compare_29mer_metrics.tsv` (8 rows = 4 models × 2 encodings) and `..._delong.tsv` (6 DeLong rows), plus 3 plots; and `runs/xgb_tuning_final/plots/{tuning_delong_comparison.png,caption.txt}` (its `results/tuning_delong_pairs.tsv` and all four input JSONs remain untracked).
- The nested-CV-by-default protocol statement (`05a_cnn_modelling.py:4-14`: `THESIS_VARIANT=flank5_rmdup_updated_h6_pepcut1`, 45,278 rows, folds-only rewrite, pre-11-26 numbers not comparable) is unchanged, and no tracked artifact in-repo backs any of those numbers (`data/processed/hla_*/` and `runs/*/models/` gitignored).

## Code Review — Shortcomings & Issues

No new commits means no new findings; everything below was **re-verified against `b0d3908` this run** with the line numbers shown. Severity legend: bug | risk | maintainability | style. Redundant / over-engineered / reinvented-stdlib-or-existing-code first, then defects.

**Redundancy, over-engineering, reinvention**

1. **The FFNN family is a near-verbatim copy of the CNN family (maintainability — still the largest standing cost)** — re-measured at the tip: `05a_cnn_modelling.py` (496 lines) vs `06a_ffnn_modelling.py` (501) = 231 diff lines; `05e_cnn_optuna_tune.py` (466) vs `06e_ffnn_optuna_tune.py` (459) = 274 diff lines; every fix lands twice. The repo already contains the lazier pattern one file over: `05d_plotting.py` is shared by both modelling scripts and `05f_cnn_optuna_plot.py` is shared by both campaigns via `--model`. Cites [66c2de5, b0d3908], unchanged this run.
2. **`open_study` race-retry body copied three times (maintainability)** — `git grep 'def open_study' b0d3908 -- src/` → `46_xgb_optuna_tune.py:181`, `05e_cnn_optuna_tune.py`, `06e_ffnn_optuna_tune.py`. One helper in a shared module covers all three; the retry policy is exactly what belongs in one place. Unchanged.
3. **`39_ffnn_width_performance.py` re-inlines the DeLong/Holm/stats stack (maintainability)** — `39:45-47` states "Stats code copied verbatim from `18_encodings_sweep_delong.py` (which itself inlined it from `15_padmode_significance.py` — its import chain is broken upstream)"; `git grep 'def holm_adjust' b0d3908 -- src/` → 10 files. The same repo shows the lazy version: `48_xgb_optuna_delong.py:50-58` `importlib`-loads `18_` (and `19_`) instead of copying, deleting ~150 lines. Unchanged.
4. **`--study-db study_smoke.db` points at a file the NN tuners never create (maintainability)** — claims at `05f_cnn_optuna_plot.py:42`, `06e_ffnn_optuna_tune.py:21`, `run_cnn_optuna_slurm.sh:63`, `run_ffnn_optuna_slurm.sh:62`; both tuners hardcode `study.db` (`05e:263`, `06e:254`) and vary only the study name. Note the XGB side is fine — `46_xgb_optuna_tune.py:216` really does write `study_smoke{sfx}.db`; the mismatch is NN-only. Fix = write smoke to `study_smoke.db` or delete the three NN claims. Unchanged.
5. **Four sweep scripts still document the protocol the same range inverted (maintainability)** — flat-CV notes at `35_cnn_kernel_performance.py:30`/`:542`, `36_cnn_channel_performance.py:30`/`:545`, `37_cnn_2layer_k2_performance.py:35`/`:691`, `38_cnn_2layer_chan_performance.py:35`/`:724` ("no nested split … inner-val best and outer-test AUC are the same numbers by construction"), while `05a:235` defaults `--cv-mode nested` and every launcher exports `CV_MODE=nested`. New runs of these sweeps produce 5-model-mean outer scores labelled as val==test. Unchanged.

**Defects and risks**

6. **BUG — `--refit` selects the checkpoint on the outer-test bucket it then scores** — `05e_cnn_optuna_tune.py:372-387`: `va_idx = folds == outer` → `vl` → `train_fixed_budget(net, tl, vl, …)` which keeps the highest-AUC epoch on that bucket (`best_state` `:196`, reloaded `:197`) → `roc_auc_score(y_ou, probs)` on the same bucket; identical at `06e_ffnn_optuna_tune.py:365-380`. So `nested_summary.json`'s "honest nested estimate" and every `winner_stability.csv` row are max-over-epochs scores on the test bucket. Lazy fix: one flag in `train_fixed_budget` to use the final-epoch state in refit only. Unchanged. Cite [b0d3908].
7. **BUG (sbatch output-directory defect, 3 instances) — no fresh checkout can start these jobs** — `run_netmhcpan_fake_slurm.sh:23` `--output=runs/fake_protein_netmhcpan/slurm/%x_%a.out` vs the only `mkdir` at `:33` (`.../netmhcpan`), tree gitignored (`.gitignore:188`); the a0101 pair: `run_netmhcpan_a0101_slurm.sh:7` vs `:53`, `run_nsp3_a0101_slurm.sh:7` vs `:36` — an in-script `mkdir` runs only after slurm has already tried to open the output file, and `docs/hla_a0101_run_recipe.md:56`/`:93` submit bare. One `mkdir -p` before each `sbatch` fixes all three. Unchanged.
8. **RISK — array size and batch count are two independent numbers** — `run_netmhcpan_fake_slurm.sh:19` hardcodes `--array=0-153%20` while batch count is `ceil(N / 500)` from `--generate N` (`32_fake_protein_netmhcpan_logo.py:98-111`); too few tasks leave batches unscored (`collect()` only warns before capping, `:210-211`), too many die on an unbound element, and the ">2 lines = complete" resume skip treats a truncated CSV as done. Unchanged.
9. **RISK — split/vintage is not recorded in the artifacts** — `05a`/`06a` write `run_tag` with no `variant`/`cv_mode` key, while `39_ffnn_width_performance.py:324-325` pools every `runs/ffnn_width_*/` dir under a caption label defaulting to `"flank5-rmdup-19mer flat (L*D)"`; since [694ad3c] says pre-11-26 numbers are on old folds, one figure can mix fold assignments under a hardcoded label. One `variant` key plus an assert in the loader closes it. Unchanged.
10. **RISK — `optuna` is in no requirements file and unpinned, and six writers share one SQLite DB per arm** — `git grep optuna b0d3908 -- 'requirements*' '*.txt' 'environment*'` → empty; all three launchers fall back to unpinned installs (`run_xgb_optuna_slurm.sh:95`, `run_cnn_optuna_slurm.sh:110`, `run_ffnn_optuna_slurm.sh:109`). Both NN campaigns are `--array=0-5` (`run_cnn_optuna_slurm.sh:30`, `run_ffnn_optuna_slurm.sh:29`) on one `study.db` per arm (`05e:263`, `06e:254`), and the retry loop covers study creation only, not `optimize`. Unchanged.
11. **MAINTAINABILITY — docs drifted** — `src/investigation/README.md:5` still says "33 scripts (6 R, 27 Python)" while `git ls-tree` at the tip counts **68 `.py` + 9 `.R` = 77**; `src/README.md:14-18` lists `05a–05d` and `06a–06c` but omits the new `05e`/`05f`/`06e`. Unchanged.
12. **STYLE — path typo and asymmetric ignore rules** — `32_fake_protein_netmhcpan_logo.py:30` sends the logo step to `runs/fake_peptide_netmhcpan` but the script writes `runs/fake_protein_netmhcpan` (`:61`); `.gitignore:165` `runs/cnn_*/` makes the CNN Optuna output tree uncommittable while `runs/ffnn_*` has no rule, so the sibling campaign can be committed including `study.db`. Unchanged.

**Passed the ladder (still true)** — `48_xgb_optuna_delong.py:50-58` imports proven stats instead of copying them; `05f` is shared by both campaigns rather than forked; both optuna launchers `mkdir -p "$OUT_ROOT/logs/slurm"` **before** `sbatch` (`run_cnn_optuna_slurm.sh:60`, `run_ffnn_optuna_slurm.sh:59`), i.e. they are not instances of finding 7.

**Not reproduced this run**

13. **Previous finding 18 (fetch resume never reads the shared pool) does not reproduce at the tip** — `src/fetch/fetch_structures.sh:66` sets `LOG_READ="${OUT_DIR}/logs/fetch_log.tsv"` where `OUT_DIR` is the shared structures directory passed as arg 1 (`:3`, `:27`, usage `:16`), and `:100` loops `for _resume_log in "${LOG_READ}" "${LOG}"` reading both the shared legacy log and the per-allele tagged log into `ALREADY_DONE` (`:101-109`). The documented design (`:60-65`) matches the code. Treat as resolved/withdrawn.

## Potential Issues

1. **The go-forward fold assignment has no in-repo generator (data/process, authors' question)** — [694ad3c] documents a one-time folds-only H5 rewrite run on hub1 and `.gitignore` excludes the tool "kept locally for reference only". Where is the procedure recorded, and how does a git-only reader reproduce or audit the h6+pepcut1 split?
2. **Which protocol/vintage each published figure is on (authors' decision)** — findings 5, 9 and 12 interact: the 35–38 sweeps and the FFNN width curve can mix flat/old-fold and nested/new-fold runs, and neither the JSON nor the caption carries the fold assignment. A `variant`/`cv_mode` field per result dict plus one sentence per caption settles it.
3. **Compute shape of the nested default (observation)** — 30 model fits for every CNN/FFNN run including sweeps (was 6), and one Optuna arm is 6 studies × 25 trials × 5 inner fits ≈ 750 trainings before `--refit`'s 6 × 3 more. Sizing fact for whoever submits the arrays, not a recommendation.
4. **Carried, still unresolved (process)** — the three sbatch output-directory instances (finding 7), the one-sided CLES rule at `04_evaluate_netmhcpan_sensitivity.R:849` (`filter(cles <= CUTOFF_CLES)` with `CUTOFF_CLES <- 0.531` at `:681` — re-verified this run) and the provenance of that 53.1% anchor, the bare `source "$NSP3_CONDA_SH"` at `src/tools/netsurfp/run_netsurfp3.sh:26` (re-verified), and the Stage C retraining receipts.
5. **Carried — most campaigns still leave nothing the repository can show (data)** — `runs/modelcompare_29mer_46617/` tracks results+plots (resolved earlier), but `runs/xgb_tuning_final/` tracks only a PNG + caption while `.gitignore:148` (`runs/xgb_*`) makes its `results/*.tsv` uncommittable without `-f`, and none of `48_`'s four input JSONs is tracked — the tuning figure cannot be regenerated from a clean clone.
6. **Verification limits (process)** — local branch `hilarius` sits at `26ec48d`, **246** behind `origin/hilarius`, with an empty `git status --porcelain`, so every cite above comes from `git show` / `git grep` / `git ls-tree` against the tip (plus `git ls-remote` to confirm the remote had not moved); `command -v Rscript` → none; script execution blocked in unattended mode. `reports/overnight/` is gitignored (`.gitignore:115` in the worktree, "published only to the `reports` branch"), so this report exists only in the local working tree.

## GitHub

From the collected GitHub data (used verbatim, not re-run): no open PRs (`gh pr list` → []), no open issues (`gh issue list` → []), no workflow runs (`gh run list` → []); repository `TheHilarius/thesis2026`, default branch `hilarius`. From git: `origin/hilarius` = **b0d3908**, i.e. **0 commits ahead of the previous report's anchor** (`git rev-list --count b0d3908..origin/hilarius` → 0), confirmed by `git ls-remote origin hilarius` returning the same SHA. Local branch `hilarius` at `26ec48d` is **246** behind with a clean working tree (an ancestor of the tip, no divergent local work). No CI is configured, so `git log --oneline` remains the whole index.

## Recommended Next Steps

Carried from the 2026-11-27 report (nothing new this run; fetch-resume item resolved), prioritized, maximum five:

1. **Close the refit leakage (finding 6)** — in `05e_cnn_optuna_tune.py:372-387` and `06e_ffnn_optuna_tune.py:365-380`, score the outer bucket from the final-epoch state instead of the state selected on it (one flag in `train_fixed_budget`, `05e:166`), then re-run `--refit` before trusting `nested_summary.json`.
2. **Make the sbatch jobs submittable on a fresh checkout (finding 7)** — `mkdir -p runs/fake_protein_netmhcpan/slurm` before `sbatch src/run/run_netmhcpan_fake_slurm.sh`, and `mkdir -p logs/{netmhcpan,netsurfp}` before the two `sbatch` lines in `docs/hla_a0101_run_recipe.md:56`/`:93` (three `mkdir -p` lines total); while there, derive `--array=0-$((n_batches-1))` from `--generate N` instead of hardcoding `0-153` (finding 8).
3. **Record the split/vintage in the artifacts (findings 5, 9; Potential Issue 2)** — add `variant` + `cv_mode` to the `05a`/`06a` result dicts, have `39_`'s loader assert a single variant across pooled run dirs, and correct the flat-CV notes at `35:30`, `36:30`, `37:35`, `38:35` and their in-code twins.
4. **Dedup / guard batch (findings 1–4, 10)** — load `18_encodings_sweep_delong` from `39_ffnn_width_performance.py` the way `48_xgb_optuna_delong.py:50-58` does (delete ~150 lines); one shared `open_study` helper for `46`/`05e`/`06e`; either write NN smoke trials to `study_smoke.db` or delete the three claims; pin `optuna` (+ `sqlalchemy`) in the requirements file and drop the `|| pip install optuna` fallbacks at `run_xgb_optuna_slurm.sh:95`, `run_cnn_optuna_slurm.sh:110`, `run_ffnn_optuna_slurm.sh:109`.
5. **The standing one-liner / doc batch (findings 4, 11, 12; Potential Issue 4)** — `abs(cles - 0.531)` at `04:849` plus the `:732`/`:752` tiers, with the 53.1% provenance answered; `source "$NSP3_CONDA_SH" || exit 1` at `run_netsurfp3.sh:26`; correct `32:30`'s `fake_peptide` → `fake_protein`; `src/investigation/README.md:5` → "77 scripts (9 R, 68 Python)" plus index rows for `39_`/`48_`; list `05e`/`05f`/`06e` in `src/README.md`; pick one ignore policy for `runs/cnn_*/` vs `runs/ffnn_*`; the remaining `14_predict_external.py` dedup and the falsy-zero `or None` at `46_xgb_optuna_tune.py:297`.

Resolved since previous reports and still closed (re-confirmed this run): the fetch resume now reads the shared legacy pool (`fetch_structures.sh:66,:100-109` — previous finding 18 withdrawn); `runs/modelcompare_29mer_46617/` tracks results and plots; `src/README.md` no longer lists `01b_datasplit_protein_level.py`; the stage rename is documented in one place with no stale `06_external_eval`/`05_cnn_modelling` references; the two optuna launchers `mkdir -p` their log dirs before `sbatch`; the `src/pipeline_a0101/` fork, the `--smoke` trial-count bug, the venv `pip install --user` failure, the R step-02 parse break, and the earlier figure/doc items listed in the 2026-11-26 report remain fixed.

## Reviewed Through

b0d3908
