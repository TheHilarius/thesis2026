# flank5_compare — 19mer (5-aa flank) vs 29mer (10-aa flank)

Motivation: the differential sequence logo showed weak signal at the
terminal positions furthest from the peptide. This campaign tests whether
trimming the flanks from 10 to 5 residues (29mer → 19mer) helps.

## How the flank5 data was made

`src/pipeline/python/reduce_29mer_to_19mer.py` derives every flank5
artifact from the existing 29mer outputs by exact slicing (flanks are
alignment-padded: N-flank X-padded left, C-flank X-padded right) — no
re-embedding, no structure re-fetch:

- CSVs: `data/processed/*_flank5.csv` (split data with folds byte-identical,
  encoded position columns N6–N10/C6–C10 dropped, compositional recomputed
  over 19 positions, `n_pad_len`/`c_pad_len`/flank coords updated)
- HDF5: `esmc300m_win_padtoken_prepared_flank5.h5` — `windows[:, 5:24, :]`,
  `pad_counts = clip(old−5, 0)`, `attrs window=19`
- Runs pick the files up via `THESIS_VARIANT=flank5` (config.py path redirect)

Known caveat: structural flank aggregates in the handcrafted feature set
(`mean_rsa_nflank`, `frac_q8_*_nflank`, `mean_plddt_nflank`, …) were
computed over the original 10-residue windows and are kept as-is — the
"handcrafted" half of these runs is not a pure 19mer.

## Experiment spec

- **Learner**: XGBoost, nested CV 6 outer × 5 inner folds, `random_state=42`
- **flank5 runs** (this folder): `xgb × {blosum50, handcrafted_blosum50}`
  (flat, jobs 222183→222185) and `xgb × handcrafted_esmc300m_win_padtoken`
  with `--vardims` EV80 = 645 dims × 19 slots (job 222174; dim ranking
  recomputed on the 19-wide H5 → `results/tables_flank5/`)
- **CNN pair** (`cnn/{full,flank5}/`): `cnn_simple_k5 ×
  emb_esmc300m_win_padtoken`, mode `cnn_cv`, 6 outer folds — rerun batch
  `cnn_flank_rerun_20261109` (both arms same batch); 29mer CNN = 0.6553,
  flank5 = 0.6704
- **29mer baselines** (other campaigns, same split/folds, different run
  batches): `encodings_sweep` (blosum50), `handcrafted_sweep`
  (handcrafted_blosum50), `is_handcrafted_helping_v2` (esmc EV80 var643)

## Held-out AUC-ROC (outer-fold mean ± std)

| feature set | 29mer | 19mer | Δ |
|---|---:|---:|---:|
| BLOSUM50 | 0.7633 ± 0.0044 | **0.7701 ± 0.0038** | +0.0068 |
| BLOSUM50 + engineered | 0.7665 ± 0.0064 | **0.7716 ± 0.0047** | +0.0051 |
| ESM-C EV80 + engineered | 0.7018 ± 0.0048 | **0.7076 ± 0.0052** | +0.0058 |
| CNN k5 (ESM-C windows) | 0.6553 ± 0.0058 | **0.6704 ± 0.0049** | +0.0145 |

## Significance (`results/paired_tests.csv`)

Paired DeLong (pooled OOF, Sun & Xu midrank) + paired row bootstrap
(B=10,000, per-pair seeded RNG) + fold-paired t-test with
Nadeau–Bengio correction; Holm across the 4 pairs.
Script: `src/investigation/32_flank5_significance.py`.

| pair | ΔAUC | DeLong p | Holm p | bootstrap p | fold-t p | |
|---|---:|---:|---:|---:|---:|:--:|
| blosum50 | +0.0068 | 4.5e-15 | 1.4e-14 | 1e-4 | 0.000 | *** |
| handcrafted_blosum50 | +0.0050 | 1.5e-10 | 3.0e-10 | 1e-4 | 0.004 | *** |
| esmc EV80 | +0.0057 | 9.0e-10 | 9.0e-10 | 1e-4 | 0.001 | *** |
| cnn_simple_k5 | +0.0145 | 1.9e-32 | 7.6e-32 | 1e-4 | 0.000 | *** |

**All four pairs favour the 19mer trim, all significant after Holm** —
consistent with the logo hypothesis that the far terminal positions were
noise. XGB effect sizes (+0.005…+0.007) are ~2× the outer-fold SD; the
CNN effect (+0.0145) is by far the largest — its convolution over window
slots is the most sensitive to per-slot noise, so removing the far
positions helps most there. The CNN pair is also the cleanest: both arms
from the same rerun batch (`cnn_flank_rerun_20261109`, moved to
`cnn/{full,flank5}/`).

Caveats: paired DeLong/bootstrap assume i.i.d. rows (no sequence
clustering; homologous peptides would narrow CIs). All inference is
conditional on one CV partition (random_state=42). The 29mer baselines
come from three different campaign folders — same split and folds, but
separate run batches. Structural flank features remain 10-wide (above).

Figure: `plots/flank5_vs_29mer_auc_roc.png` (brackets = stars + Δ).

## Reproduce

```bash
python src/pipeline/python/reduce_29mer_to_19mer.py   # derive flank5 data
# sbatch src/run/run_flank5_quick.sh ; sbatch src/run/run_flank5_esmc.sh
# CNN: sbatch (05_cnn_modelling, cnn_simple_k5) x {full, flank5} — see
#   cnn/*/logs/cnn/*.txt for the exact invocations
python src/investigation/32_flank5_significance.py
python src/investigation/plot_flank5_compare.py
```
