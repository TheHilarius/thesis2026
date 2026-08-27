# src/ — Codebase Layout

MHC-I antigen-processing prediction pipeline. Scripts grouped by functionality/purpose.
All scripts assume they run from the **repo root** unless noted otherwise.

## Directory layout

```
src/
├── pipeline/          # Core numbered analysis scripts (the actual pipeline)
│   ├── python/        # ML pipeline: data split → embeddings → modelling → analysis
│   │                 # 01_datasplit.py, 01b_datasplit_protein_level.py,
│   │                 # 02_datasplit_analysis.py, 03_prepare_embeddings.py,
│   │                 # 04_modelling.py, 05_model_analysis.py, 06_feature_importance.py,
│   │                 # 07_compare_heldout_roc.py, baseline_model.py,
│   │                 # extract_model_metrics.py, config.py
│   └── r/            # Feature engineering + analysis (R)
│                     # functions.R (shared helpers), run_pipeline.R (runner),
│                     # 02_load_iedb_data.R, 04_data_overview_figures.R,
│                     # 04_evaluate_netmhcpan_sensitivity.R, 05_protein_features.R,
│                     # 06_netsurfp_protein_features.R, 07_alpha_fold_features.R,
│                     # 08_aa_encoding.R, 09_numeric_feature_analysis.R,
│                     # 10_categorical_residue_analysis.R, 11_structural_preference_logo.R
├── tools/             # Wrappers around external tools
│   ├── netmhcpan/     # run_netmhcpan.sh, run_netmhcpan_test.sh
│   ├── netsurfp/      # run_netsurfp3.sh, nsp3_clean_fasta.sh, prepare_nsp3_batches.sh
│   ├── alphafold/     # af_run_chunks.sh, af_split_predict.py, af_stitch_plddt.py,
│   │                 # af_version_compare.R
│   └── esm/           # embed_peptides_w_esm.py, embed_structures_esmif_gpu.py,
│                     # embed_structures_w_esm_if.py, inspect_embeddings.py,
│                     # summarize_embeddings.py
├── fetch/             # Structure fetching + coverage
│                     # fetch_structures.sh, fetch_structures_hybrid.sh,
│                     # audit_hybrid_structures.sh, prepare_fetch_list.sh,
│                     # report_structure_coverage.py
├── util/              # FASTA / data-prep utilities
│                     # batch_fasta.py, build_combined_fasta.py,
│                     # export_fasta_files_from_iedb.py
├── run/               # Orchestration scripts (run from repo root)
│                     # run_all_models.sh, run_all_models_and_analyze.sh,
│                     # run_all_feature_importance.sh
├── archive/           # Obsolete/superseded scripts (kept for reference, delete later)
└── logo_comparison.qmd  # Logo comparison report source (renders to results/)
```

## Run order

1. **Data prep** — `util/` + `pipeline/python/01*.py` + `pipeline/r/02_load_iedb_data.R`
2. **External tool features** — `tools/` (NetMHCpan, NetSurfP, AlphaFold, ESM)
3. **Feature engineering** — `pipeline/r/run_pipeline.R` (steps 04–07)
4. **Modelling** — `pipeline/python/04_modelling.py` (or `run/run_all_models.sh`)
5. **Analysis** — `pipeline/python/05_model_analysis.py`, `06_feature_importance.py`, `07_compare_heldout_roc.py`, `pipeline/r/08–11`

## Notes

- Python pipeline scripts import `config.py` via a `sys.path` bootstrap — runnable from any cwd.
- R scripts `source("src/pipeline/r/functions.R")` — run from repo root.
- `run/` scripts `cd` to repo root internally; invoke as `bash src/run/<script>.sh`.
- Rendered outputs go to `results/` (e.g. `logo_comparison.html`).
