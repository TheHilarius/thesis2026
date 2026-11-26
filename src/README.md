# src/ — Codebase Layout

MHC-I antigen-processing prediction pipeline. Scripts grouped by functionality/purpose.
All scripts assume they run from the **repo root** unless noted otherwise.

## Directory layout

```
src/
├── pipeline/          # Core numbered analysis scripts (the actual pipeline)
│   ├── python/        # ML pipeline: data split → embeddings → modelling → analysis
│   │                 # 01_datasplit.py, 03_prepare_embeddings.py, 03b_run_matrix.py,
│   │                 # 04_modelling.py, 04b_run_matrix.py,
│   │                 # 05a_cnn_modelling.py, 05b_run_cnn_matrix.py,
│   │                 # 05c_cnn_models.py, 05d_plotting.py (shared),
│   │                 # 06a_ffnn_modelling.py, 06c_ffnn_models.py,
│   │                 # 07_external_eval.py, 11–14 (window reduction/dim
│   │                 # screening/autoencoder/external predict),
│   │                 # baseline_model.py, extract_*.py, config.py
│   │                 # Number = pipeline stage, letter = function within
│   │                 # the stage: a = main training script, b = sweep
│   │                 # matrix runner, c = model zoo, d = plotting.
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
│   └── esm/           # embed_peptides_esmc.py, embed_peptides_esmif.py,
│                     # legacy/embed_structures_esmif_gpu.py (flat, pre-pad-modes)
├── fetch/             # Structure fetching
│                     # fetch_uniprot_fastas.py, prepare_fetch_list.sh,
│                     # fetch_structures.sh, find_correct_af_model.py
├── util/              # FASTA / data-prep utilities
│                     # batch_fasta.py, build_combined_fasta.py,
│                     # export_fasta_files_from_iedb.py
├── run/               # Orchestration scripts (run from repo root)
│                     # run_all_models.sh, run_all_models_and_analyze.sh,
│                     # run_all_feature_importance.sh
├── investigation/     # One-off exploration/diagnosis scripts (ESM pad probes,
│                     # data/structure audits, embedding QA, post-modelling
│                     # analysis) — see investigation/README.md
├── archive/           # Obsolete/superseded scripts (kept for reference, delete later)
└── logo_comparison.qmd  # Logo comparison report source (renders to results/)
```

## Run order

1. **Data prep** — `util/` + `pipeline/python/01*.py` + `pipeline/r/02_load_iedb_data.R [hla0201|hla_a0101]`
2. **External tool features** — `tools/` (NetMHCpan, NetSurfP, AlphaFold, ESM); allele args where per-allele inputs apply
3. **Feature engineering** — `pipeline/r/run_pipeline.R [hla0201|hla_a0101]` (steps 04–08; NSP3/AlphaFold tool runs sit between 05/06 and 06/07)
4. **Modelling** — `pipeline/python/04_modelling.py` (flat/sklearn), `05a_cnn_modelling.py` (CNN on window embeddings), `06a_ffnn_modelling.py` (FFNN on flattened windows); external-allele scoring: `THESIS_ALLELE=hla_a0101` + `pipeline/python/14_predict_external.py`
5. **Analysis** — `pipeline/r/08–11`, `src/investigation/` (CNN kernel/channel/2-layer performance 35–38, FFNN width 39)

Full per-allele recipe (incl. server commands): `docs/hla_a0101_run_recipe.md`.

## Notes

- Python pipeline scripts import `config.py` via a `sys.path` bootstrap — runnable from any cwd.
- Nov 2026 rename: `05_cnn_modelling/05_cnn_models/05_plotting` → `05a/05c/05d`, `06_external_eval` → `07_external_eval`. Old `runs/` logs and READMEs keep the old names (history untouched).
- Go-forward data (2026-11-26): all new CNN/FFNN runs default `THESIS_VARIANT=flank5_rmdup_updated_h6_pepcut1` (robust homology split, folds-only change). Pre-2026-11-26 numbers are old folds, not comparable. ESM-C 300M windows only; esmif/esmc6b/mpnn dropped. `rewrite_h5_folds.py` was a one-time migration tool (gitignored).
- R scripts `source("src/pipeline/r/functions.R")` — run from repo root.
- `run/` scripts `cd` to repo root internally; invoke as `bash src/run/<script>.sh`.
- Rendered outputs go to `results/` (e.g. `logo_comparison.html`).
