# padmode_choice — pad-mode / data-comparison batch

XGBoost nested-CV batch comparing AA encodings, engineered features alone,
and embeddings-only representations under 4 pad-fill modes per toolkit.
Identified from server auto-run folders (`runs/xgb_*_flat_*` on hub1) by
JSON content (`features_key`, `pca_mode=flat`, `pca_total`), not timestamps.
Source of truth for the grid: `src/run/run_padmode_data_compare_slurm.sh`
(+ `run_padmode_data_compare_hc_slurm.sh` for handcrafted-only).

## Experiment spec

- **Learner**: XGBoost, nested CV 6 outer × 5 inner folds, `random_state=42`
- **PCA @ 90% EV** (flat mode, mode-invariant spectra):
  ESM-C k=6389 · ESM-IF k=4308 · baselines: none
- **Pad modes** (code tokens):
  - ESM-C: `zeropad`, `padtoken`, `impute_bos_eos`, `impute_boundary`
  - ESM-IF: `win` (zero), `pad`, `boundary`, `eos_bos_repeat`
- **12 runs** (see `models/flat/cv_results_xgb_*.json`):
  handcrafted · sparse · blosum · handcrafted_blosum
  + 4× emb_esmc_win_* @6389 + 4× emb_esmif_* @4308

## Held-out AUC-ROC (outer-fold mean)

| feature_set | AUC | runtime (h) |
|---|---:|---:|
| handcrafted_blosum | 0.7648 | 0.19 |
| blosum | 0.7633 | 0.13 |
| sparse | 0.7409 | 0.06 |
| emb_esmc_win_impute_bos_eos | 0.6882 | 15.21 |
| emb_esmc_win_padtoken | 0.6880 | 24.99 |
| emb_esmc_win_zeropad | 0.6878 | 19.71 |
| emb_esmc_win_impute_boundary | 0.6870 | 15.50 |
| emb_esmif_win | 0.6335 | 12.79 |
| emb_esmif_boundary | 0.6333 | 10.51 |
| emb_esmif_eos_bos_repeat | 0.6328 | 8.26 |
| emb_esmif_pad | 0.6325 | 7.69 |
| handcrafted | 0.5658 | 0.03 |

CSV-only baselines beat embeddings-only by a wide margin — embeddings here
carry no handcrafted/sparse features. Full tables: `results/model_matrix_padmode.tsv`,
`results/runtime.tsv`, figure `plots/padmode_data_compare_auc_roc.png`.

## Padded-subset AUC (pad-mode discrimination)

Overall Val AUC cannot separate pad modes (real-slot signal identical). Only
the informative padded rows carry mode signal (~1443/toolkit: ≥1 pad AND ≥1
real slot). AUC recomputed on those rows from saved OOF predictions
(`results/padmode_subset_auc.csv`, script `09_padmode_subset_auc.py`):

| toolkit | best padmode (padded AUC) | worst | n_padded |
|---|---|---|---:|
| ESM-C @6389 | zero 0.6990 (pad_token 0.6969) | eos_bos_repeat 0.6862 | 1443 |
| ESM-IF @4308 | eos_bos_repeat 0.6687 (pad 0.6631) | boundary 0.6548 | 1442 |

Non-padded control AUC is ~identical across modes (mode-invariant reals) —
as expected.

## Significance (`results/paired_tests.csv`)

Primary test — paired DeLong (correlated ROC curves, Sun & Xu 2014
midrank) on pooled out-of-fold predictions, Holm-corrected within each
family. Effect sizes reported as ΔAUC with DeLong 95% CIs
(`ci95_lo/hi = Δ ± 1.96·SE`). Robustness checks in the same CSV: paired
row-level bootstrap (B=10,000, per-pair RNG seeded from a hash of the
pair name; `p_boot = 2·min(P(Δ*≤0), P(Δ*≥0))` with +1 smoothing, floor
≈1e-4) and fold-paired t-test on the 6 outer-fold AUCs, with a
Nadeau–Bengio sensitivity variant (`t_nb = t/√2.2`, df=5 — heuristic,
derived for repeated subsampling). Script: `15_padmode_significance.py`.

Methods: "Differences in AUC between models were tested on identical
evaluation rows using the DeLong test for correlated ROC curves, applied
to pooled out-of-fold predictions, with Holm correction within each
family of comparisons. Effect sizes are reported as delta-AUC with
DeLong 95% confidence intervals. A paired row-level bootstrap
(B = 10,000) and a fold-paired t-test on the six outer-fold AUCs (with a
Nadeau–Bengio-corrected variant) are reported as robustness checks.
Because CV folds share training data, the fold-level test is
approximate, and all inference is conditional on a single CV partition
(random_state = 42)."

Interpretation:

- **Padmode pairwise on padded subset: no detectable difference.** All
  12 pairs (6 per toolkit) have DeLong CIs spanning 0; no pair is
  significant after Holm. Non-significant results indicate no detectable
  difference, not equivalence — equivalence was not formally tested.
- **Padded-subset CI extent:** ESM-C DeLong 95% CIs lie within ±0.030;
  ESM-IF within ±0.032. A bound of this size cannot exclude padmode
  effects as large as the sparse-vs-blosum effect (−0.022) *on the padded
  rows*. Padmode changes ~3% of rows, so the effect on the full data
  would be much smaller (full-data padmode DeLong CIs: within ±0.004).
- **sparse vs blosum (full data): significant.** ΔAUC = −0.0223, DeLong
  CI [−0.0247, −0.0200], z = −19.0, p ≈ 4e-80 (`***`) — BLOSUM50 beats
  one-hot sparse. Nadeau–Bengio-corrected fold t-test agrees
  (p ≈ 4e-4, ≈ 7e-4 after Holm over 3 baseline pairs).
- **sparse vs blosum (padded subset): no detectable difference.**
  ΔAUC = +0.0053 (sign flip vs full data), p = 0.51.

Caveats: all inference is conditional on a single CV partition
(random_state = 42) — no seed-variance estimate (XGB is deterministic
here: subsample=1.0, colsample_bytree=1.0, fixed seed). Row bootstrap
and DeLong assume i.i.d. rows; homologous/near-duplicate peptide
sequences would make CIs too narrow (no sequence clustering was applied).
The paired design is robust to shared bias for model-vs-model contrasts.

## Reproduce

```bash
env_esmc/bin/python src/pipeline/python/extract_model_metrics.py \
  --results "runs/padmode_choice/models/flat/cv_results_*.json" \
  --out runs/padmode_choice/results/model_matrix_padmode.tsv
env_esmc/bin/python src/investigation/plot_padmode_data_compare.py \
  --tsv runs/padmode_choice/results/model_matrix_padmode.tsv \
  --out runs/padmode_choice/plots/
env_esmc/bin/python src/investigation/09_padmode_subset_auc.py \
  --models-dir runs/padmode_choice/models/flat \
  --out runs/padmode_choice/results/padmode_subset_auc.csv
env_esmc/bin/python src/investigation/15_padmode_significance.py
```
