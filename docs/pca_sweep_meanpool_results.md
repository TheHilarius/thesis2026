# PCA Sweep Results: Handcrafted Sparse + Mean-Pooled Context Window Embeddings

## Dataset

- **Peptides:** 46,617 HLA-A\*02:01 binding predictions
- **Task:** Binary classification (binder vs non-binder)
- **Cross-validation:** Nested CV — 5 inner folds × 6 outer folds (30 models per combo)
- **Embeddings:** Mean-pooled context window embeddings (ESM-C 600M, ESM-IF1)
- **Feature sets:**
  - `handcrafted_sparse_esmc`: Structural features + one-hot AA encoding + ESM-C context embeddings (631–878 features after PCA)
  - `handcrafted_sparse_esmif`: Structural features + one-hot AA encoding + ESM-IF context embeddings (631–878 features after PCA)
- **PCA sweep:** 6 values per embedding type
  - ESM-C: 1, 13, 26, 66, 218, 718 components (50%–99% variance)
  - ESM-IF: 9, 64, 95, 148, 248, 420 components (50%–99% variance)

---

## Best Model per Feature Set

| Model | Embedding | Best PCA | CV AUC-ROC | ± std | MCC | F1 | Accuracy |
|-------|-----------|----------|-----------|-------|-----|-----|----------|
| **xgb** | **esmc** | **66** | **0.7584** | ±0.0059 | 0.3783 | 0.6629 | 0.6913 |
| **xgb** | **esmif** | **9** | **0.7465** | ±0.0050 | 0.3645 | 0.6583 | 0.6841 |
| lr_l2 | esmc | 218 | 0.7401 | ±0.0056 | 0.3492 | 0.6593 | 0.6747 |
| lr_elasticnet | esmc | 218 | 0.7400 | ±0.0057 | 0.3492 | 0.6594 | 0.6747 |
| lr_l2 | esmif | 248 | 0.7258 | ±0.0068 | 0.3268 | 0.6475 | 0.6636 |
| lr_elasticnet | esmif | 248 | 0.7257 | ±0.0068 | 0.3295 | 0.6494 | 0.6648 |
| rf | esmc | 1 | 0.7122 | ±0.0050 | 0.3057 | 0.6151 | 0.6561 |
| rf | esmif | 9 | 0.7090 | ±0.0041 | 0.3012 | 0.6123 | 0.6539 |

> **Top performer:** XGBoost with ESM-C PCA=66 achieves CV AUC-ROC of 0.7584 (±0.0059).

---

## Full PCA Sweep by Model

### XGBoost

#### ESM-C

| PCA | CV AUC-ROC | ± std | MCC | F1 | Accuracy |
|-----|-----------|-------|-----|-----|----------|
| 1 | 0.7465 | ±0.0047 | 0.3605 | 0.6576 | 0.6819 |
| 13 | 0.7568 | ±0.0040 | 0.3795 | 0.6662 | 0.6915 |
| 26 | 0.7579 | ±0.0038 | 0.3766 | 0.6636 | 0.6902 |
| **66** | **0.7584** | **±0.0059** | **0.3783** | **0.6629** | **0.6913** |
| 218 | 0.7575 | ±0.0053 | 0.3786 | 0.6610 | 0.6917 |
| 718 | 0.7508 | ±0.0052 | 0.3654 | 0.6515 | 0.6853 |

> XGBoost + ESM-C peaks at PCA=66 (AUC 0.7584). Performance plateaus between PCA 13–218, drops at 718.

#### ESM-IF

| PCA | CV AUC-ROC | ± std | MCC | F1 | Accuracy |
|-----|-----------|-------|-----|-----|----------|
| **9** | **0.7465** | **±0.0050** | **0.3645** | **0.6583** | **0.6841** |
| 64 | 0.7456 | ±0.0068 | 0.3570 | 0.6521 | 0.6806 |
| 95 | 0.7455 | ±0.0070 | 0.3577 | 0.6514 | 0.6810 |
| 148 | 0.7463 | ±0.0076 | 0.3586 | 0.6508 | 0.6816 |
| 248 | 0.7456 | ±0.0074 | 0.3597 | 0.6497 | 0.6823 |
| 420 | 0.7412 | ±0.0056 | 0.3542 | 0.6463 | 0.6797 |

> XGBoost + ESM-IF peaks at PCA=9 (AUC 0.7465). Performance essentially flat across all PCA values — minimal PCA needed.
>
> NOTE: Interesting to test without embeddings and compare.

---

### Logistic Regression (L2)

#### ESM-C

| PCA | CV AUC-ROC | ± std | MCC | F1 | Accuracy |
|-----|-----------|-------|-----|-----|----------|
| 1 | 0.7116 | ±0.0057 | 0.3052 | 0.6381 | 0.6524 |
| 13 | 0.7228 | ±0.0044 | 0.3230 | 0.6464 | 0.6615 |
| 26 | 0.7284 | ±0.0040 | 0.3338 | 0.6517 | 0.6669 |
| 66 | 0.7356 | ±0.0051 | 0.3421 | 0.6562 | 0.6711 |
| **218** | **0.7401** | **±0.0056** | **0.3492** | **0.6593** | **0.6747** |
| 718 | 0.7380 | ±0.0052 | 0.3470 | 0.6579 | 0.6737 |

> LR-L2 + ESM-C benefits from more PCA components, peaking at 218. Slight drop at 718.

#### ESM-IF

| PCA | CV AUC-ROC | ± std | MCC | F1 | Accuracy |
|-----|-----------|-------|-----|-----|----------|
| 9 | 0.7154 | ±0.0059 | 0.3071 | 0.6387 | 0.6534 |
| 64 | 0.7222 | ±0.0061 | 0.3185 | 0.6443 | 0.6592 |
| 95 | 0.7235 | ±0.0061 | 0.3213 | 0.6456 | 0.6606 |
| 148 | 0.7249 | ±0.0065 | 0.3244 | 0.6469 | 0.6622 |
| **248** | **0.7258** | **±0.0068** | **0.3268** | **0.6475** | **0.6636** |
| 420 | 0.7248 | ±0.0058 | 0.3255 | 0.6466 | 0.6630 |

> LR-L2 + ESM-IF peaks at PCA=248. Similar pattern to ESM-C — benefits from more components.

---

### Logistic Regression (ElasticNet)

#### ESM-C

| PCA | CV AUC-ROC | ± std | MCC | F1 | Accuracy |
|-----|-----------|-------|-----|-----|----------|
| 1 | 0.7116 | ±0.0058 | 0.3051 | 0.6381 | 0.6523 |
| 13 | 0.7228 | ±0.0045 | 0.3226 | 0.6462 | 0.6613 |
| 26 | 0.7283 | ±0.0040 | 0.3332 | 0.6514 | 0.6667 |
| 66 | 0.7355 | ±0.0051 | 0.3418 | 0.6561 | 0.6709 |
| **218** | **0.7400** | **±0.0057** | **0.3492** | **0.6594** | **0.6747** |
| 718 | 0.7380 | ±0.0052 | 0.3469 | 0.6578 | 0.6737 |

> ElasticNet + ESM-C nearly identical to LR-L2. Peaks at PCA=218.

#### ESM-IF

| PCA | CV AUC-ROC | ± std | MCC | F1 | Accuracy |
|-----|-----------|-------|-----|-----|----------|
| 9 | 0.7154 | ±0.0060 | 0.3070 | 0.6386 | 0.6534 |
| 64 | 0.7221 | ±0.0062 | 0.3183 | 0.6443 | 0.6591 |
| 95 | 0.7234 | ±0.0061 | 0.3199 | 0.6450 | 0.6599 |
| 148 | 0.7249 | ±0.0066 | 0.3241 | 0.6468 | 0.6621 |
| **248** | **0.7257** | **±0.0068** | **0.3295** | **0.6494** | **0.6648** |
| 420 | 0.7248 | ±0.0058 | 0.3254 | 0.6469 | 0.6629 |

> ElasticNet + ESM-IF peaks at PCA=248 — matches LR-L2. Same "more components helps" pattern.

---

### Random Forest

#### ESM-C

| PCA | CV AUC-ROC | ± std | MCC | F1 | Accuracy |
|-----|-----------|-------|-----|-----|----------|
| **1** | **0.7122** | **±0.0050** | **0.3057** | **0.6151** | **0.6561** |
| 13 | 0.7120 | ±0.0049 | 0.3105 | 0.6158 | 0.6586 |
| 26 | 0.7095 | ±0.0044 | 0.3064 | 0.6145 | 0.6565 |
| 66 | 0.7050 | ±0.0039 | 0.2925 | 0.6041 | 0.6498 |
| 218 | 0.6990 | ±0.0043 | 0.2873 | 0.5921 | 0.6477 |
| 718 | 0.6886 | ±0.0061 | 0.2714 | 0.5651 | 0.6402 |

> RF + ESM-C peaks at PCA=1. Performance degrades monotonically with more PCA — tree models prefer raw, untransformed features.
>
> NOTE: Interesting to test without embeddings and compare.

#### ESM-IF

| PCA | CV AUC-ROC | ± std | MCC | F1 | Accuracy |
|-----|-----------|-------|-----|-----|----------|
| **9** | **0.7090** | **±0.0041** | **0.3012** | **0.6123** | **0.6539** |
| 64 | 0.6954 | ±0.0046 | 0.2779 | 0.5954 | 0.6426 |
| 95 | 0.6939 | ±0.0047 | 0.2733 | 0.5913 | 0.6405 |
| 148 | 0.6915 | ±0.0043 | 0.2730 | 0.5874 | 0.6405 |
| 248 | 0.6892 | ±0.0051 | 0.2708 | 0.5819 | 0.6396 |
| 420 | 0.6864 | ±0.0045 | 0.2644 | 0.5704 | 0.6367 |

> RF + ESM-IF peaks at PCA=9. Same monotonic degradation pattern.
>
> NOTE: Interesting to test without embeddings and compare.

---

## Key Findings

1. **XGBoost dominates.** Best model across all feature sets, achieving AUC-ROC 0.746–0.758. The 1.2–4.6% gap over LR/RF is consistent across PCA values.

2. **ESM-C > ESM-IF.** ESM-C context embeddings outperform ESM-IF structure embeddings for every model. Best ESM-C (0.7584) vs best ESM-IF (0.7465) — a 1.2% AUC gap. ESM-C captures sequence context that correlates better with MHC-I binding.

3. **PCA saturation differs by model family:**
   - **XGBoost:** Peaks at PCA 66 (ESM-C) / PCA 9 (ESM-IF). More PCA hurts — gradient boosting benefits from lower-dimensional, less noisy representations.
   - **LR (L2/ElasticNet):** Peaks at PCA 218 (ESM-C) / PCA 248 (ESM-IF). Linear models benefit from more variance — the additional components provide useful signal even with some noise.
   - **RF:** Peaks at PCA 1 (ESM-C) / PCA 9 (ESM-IF). Tree models are invariant to feature scaling and benefit from raw, untransformed features. PCA hurts RF because it destroys the feature-space structure that split decisions rely on.

4. **LR-L2 ≈ LR-ElasticNet.** Nearly identical performance across all PCA values (AUC difference < 0.001). Regularization type doesn't matter — the bottleneck is the linear model family itself, not the penalty.

5. **Low PCA is surprisingly competitive.** PCA=1 (50% variance) already captures most of the signal. The gap between PCA=1 and best PCA is only 1.2% for XGBoost and 2.8% for LR. This suggests the embedding space is highly low-dimensional for the binding prediction task.

6. **Three optima sit at minimum PCA (≈50% variance).** XGBoost+ESM-IF (PCA 9), RF+ESM-C (PCA 1), and RF+ESM-IF (PCA 9) all peak at their smallest sweep value. The embeddings add little here — **interesting to test without embeddings and compare** (handcrafted + sparse only).

---

## Files

| File | Description |
|------|-------------|
| `results/tables/pca_sweep_clean.tsv` | All 48 combos (48 rows) |
| `results/tables/pca_sweep_best_by_model_feature.tsv` | Best PCA per model + feature set (8 rows) |
| `src/pipeline/python/extract_pca_results.py` | Script that extracts results from `cv_results_*.json` |

---

*Generated by `extract_pca_results.py`. Data: mean-pooled context window embeddings on 46,617 HLA-A\*02:01 peptides. Nested CV (5 inner × 6 outer folds).*
