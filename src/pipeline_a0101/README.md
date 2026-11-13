# HLA-A\*01:01 External-Validation Edition (`src/pipeline_a0101/`)

Parallel pipeline edition that scores a **second allele** (HLA-A\*01:01) with
models **trained on the legacy HLA-A\*02:01 pipeline — no retraining**.
The legacy pipeline (`src/pipeline/`) and all its outputs in
`data/processed/` are **untouched** and continue to document the previous
dataset. Everything this edition produces goes to `data/processed/hla_a0101/`.

```
src/pipeline_a0101/
├── r/                     R chain (editions of legacy 02/04/05/06/07/08 + runner)
├── python/                a0101 config + embedding prep + external prediction
├── netmhcpan/             netMHCpan runners (HLA-A01:01, 9-mers only)
└── fetch/                 UniProt FASTA fetch for a0101-specific proteins
```

## Verified input

`data/raw/mhc_ligand_table_export_HLA-A0101.csv` — checked 2026-11-11:
55,183 rows, 112 columns (same IEDB schema as the legacy export),
comma-delimited, **100% HLA-A\*01:01**, **100% ligand presentation**,
**100% linear peptide** — exactly mirrors the filters of the original
A0201 export. 9-mers: 22,394; 10-mers: 11,109; 8-mers: 1,850.
(The 97 MB xlsx in `~/Downloads` is A\*02:01-filtered — do NOT use it.)

## Shared vs per-allele resources

| Resource | Location | Why |
|---|---|---|
| AlphaFold PDBs + pLDDT | `data/processed/structures/` (shared) | per protein, allele-independent |
| NetSurfP-3 output | `data/processed/nsp3/` (shared) | per protein |
| Trained models | `models/` (shared, read-only) | loaded by 14_predict_external |
| FASTA database | `data/raw/fasta/combined_positives_only.fasta` (legacy) **+** `hla_a0101_positives_only.fasta` (new) | union DB; 04/05 read both |
| All a0101 data | `data/processed/hla_a0101/` | per-allele |
| a0101 figures | `results/figures/netmhcpan_hla_a0101/` | per-allele |

## Run recipe (in order)

```bash
# 0. (server) netMHCpan 9-mer predictions for HLA-A*01:01 — needs batch
#    FASTAs of the proteins covering the a0101 positives (see notes below):
bash src/pipeline_a0101/netmhcpan/run_netmhcpan_parallel.sh \
     <fasta_batch_dir> data/processed/hla_a0101/netmhcpan hla_a0101 10

# 1. R chain — steps 02→08 (positives → combined pos+neg → features →
#    encodings). After step 02 the runner pauses effectively: run the
#    FASTA fetch for the ~2000 a0101 proteins outside the legacy DB:
Rscript -e 'source("src/pipeline_a0101/r/02_load_iedb_data.R")'
python src/pipeline_a0101/fetch/fetch_a0101_fastas.py
Rscript src/pipeline_a0101/r/run_pipeline.R

# 2. (optional, GPU) embeddings — only if embedding feature sets are tested:
#    raw H5 for the a0101 peptides via the legacy tools with redirected IO:
python src/tools/esm/embed_peptides_esmc.py \
    --csv data/processed/hla_a0101/df_all.csv \
    --out-dir data/processed/hla_a0101/embeddings
#    then align to the a0101 dataset:
python src/pipeline_a0101/python/03_prepare_embeddings.py --embedding esmc

# 3. Score with trained models (no retraining):
python src/pipeline_a0101/python/14_predict_external.py \
    --run-tag xgb_blosum50 --model xgb --features blosum50
```

## Regeneration cost per representation

| Representation | Cost | How |
|---|---|---|
| Encodings (blosum50/62/80, sparse, kidera, atchley, compositional) | cheap, deterministic | step 08 of the a0101 R chain writes `hla_a0101/df_<enc>.csv` |
| Embeddings (esmc, esmif, mpnn, ...) | GPU recompute | legacy `src/tools/esm|proteinmpnn/embed_peptides_*.py` with `--csv hla_a0101/df_all.csv --out-dir hla_a0101/embeddings`, then a0101 `03_prepare_embeddings.py` |
| Models | none | saved `.pkl` inner-CV models + artifacts (see below) |

## ⚠️ Artifacts requirement for 14_predict_external

`04_modelling.py` historically saved only the model per inner fold — the
preprocessing objects (scaler, fold PCA, column medians, dimselect dims)
lived only in memory, so a loaded `.pkl` alone was **unreplayable**.
The legacy `04_modelling.py` now **additively** saves
`{run_tag}_outer{i}_inner{j}_artifacts.pkl` next to each model — existing
behaviour is unchanged.

Consequence: models trained **before** this change have no artifacts file.
For each config you want to externally validate, re-run
`src/pipeline/python/04_modelling.py` once with identical settings —
same data, folds and seed, so the models regenerate deterministically and
now come with artifacts.

## Notes / gotchas

- **FASTA coverage**: ~2000 of ~7000 a0101 proteins are NOT in the legacy
  FASTA DB. Without `fetch_a0101_fastas.py` those peptides are silently
  dropped by the semi-join in 04. With the fetch, they survive; proteins
  still lacking an AlphaFold structure are then excluded by the ledger
  (`no_pdb`) exactly like legacy. NSP3 coverage: proteins without an NSP3
  run get NA features (imputed downstream).
- **netMHCpan peptide lengths**: a0101 runners use `-l 9` only (legacy used
  8–14). 04 filters to 9-mers anyway.
- **Batch FASTAs for netMHCpan**: the legacy batch splitter
  (`src/util/batch_fasta.py` + fasta prep) was built for the A0201 protein
  set. For full a0101 coverage, rebuild batches over the union protein set
  (legacy + a0101 FASTAs) before step 0.
- **Anchor check** in a0101 08 uses A\*01:01 anchors (P2 V/I/T/M, P3 D/E,
  P9 Y) — allele-specific, deliberately NOT the legacy A\*02:01 anchors.
- **Server-side Slurm netMHCpan wrapper** is not in this repo — if it exists
  on the server, apply the same two edits there: `-a HLA-A01:01`, `-l 9`,
  output to `data/processed/hla_a0101/netmhcpan`.
- Both raw IEDB exports are comma-delimited and parse with `sep=","`
  (verified 2026-11-11: legacy 200,381 rows / a0101 55,183 rows, each 100%
  single-allele / ligand presentation / linear peptide). Semicolons occur
  only *inside* legacy data fields (author lists) — there is no delimiter
  bug. Note: the legacy file contains latin-1 bytes (parse with
  `encoding='latin-1'` in Python). The legacy 02 script is untouched by design.
