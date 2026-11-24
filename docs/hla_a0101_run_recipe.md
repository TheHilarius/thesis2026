# HLA-A\*01:01 (a0101) pipeline run recipe

Reruns the **entire pipeline** for the second allele HLA-A\*01:01 with the
**integrated, allele-aware `src/pipeline/`** scripts — same exact code per
allele, only paths and labels switch. Everything runs on the **server**
(DTU hub1, `/home/projects1/thesis_s204692_s204581/thesis2026`); the local
repo holds legacy A\*02:01 outputs for parity checks only.

The old `src/pipeline_a0101/` fork and its outputs are **superseded and
deleted** — do not reuse fork-era results (netMHCpan union batches etc.).

## Verified input

`data/raw/mhc_ligand_table_export_HLA-A0101.csv` — checked 2026-11-11:
55,183 rows, 112 columns (same IEDB schema as the legacy export),
comma-delimited, **100% HLA-A\*01:01**, **100% ligand presentation**,
**100% linear peptide**. 9-mers: 22,394; 10-mers: 11,109; 8-mers: 1,850.
(The 97 MB xlsx in `~/Downloads` is A\*02:01-filtered — do NOT use it.)
Legacy raw export parses with `encoding='latin-1'` in Python; semicolons
occur only inside data fields (no delimiter bug).

## Shared vs per-allele resources

| Resource | Location | Why |
|---|---|---|
| AlphaFold PDBs + logs | `data/processed/structures/` (shared) | per protein, allele-independent |
| NetSurfP-3 output | `data/processed/nsp3/` (shared) | per protein; batch dirs named after batch files, per-allele prefixes keep runs separate |
| Trained models + artifacts | `models/` (shared, read-only) | loaded by 14_predict_external |
| All a0101 data | `data/processed/hla_a0101/` | per-allele |
| a0101 figures | `results/figures/netmhcpan_hla_a0101/`, `netsurfp3_hla_a0101/`, `models_hla_a0101/` | per-allele |
| FASTAs | `data/raw/fasta/hla_a0101_positives_only.fasta` (+ `_full_length`) | per-allele, never merged on disk |

## Run order (all on hub1, repo root)

### Stage A — IEDB → FASTA → netMHCpan

```bash
# 1. Clean IEDB export → positives
Rscript src/pipeline/r/02_load_iedb_data.R hla_a0101
#    OUT data/processed/hla_a0101/pos_EL_all_epitopes_hla_a0101.csv
#        data/processed/hla_a0101/iedb_stage_counts.csv

# 2. Fetch per-allele UniProt FASTAs
python3 src/fetch/fetch_uniprot_fastas.py \
  --input-csv data/processed/hla_a0101/pos_EL_all_epitopes_hla_a0101.csv \
  --output-fasta data/raw/fasta/hla_a0101_full_length.fasta \
  --positives-fasta data/raw/fasta/hla_a0101_positives_only.fasta

# 3. Batch the positives-only FASTA (OWN allele set, no union)
python3 src/util/batch_fasta.py data/raw/fasta/hla_a0101_positives_only.fasta \
  data/raw/fasta/a0101_batches 50

# 4. netMHCpan (9-mers only, HLA-A01:01) — compute node, NOT login shell.
#    Slurm wrapper with receipts (recommended):
NMPAN_DIR=$(dirname "$(which netMHCpan)") \
  sbatch --export=ALL src/run/run_netmhcpan_a0101_slurm.sh
#    (NMPAN_DIR only needed if netMHCpan is missing from the batch shell PATH)
#    Direct runner call, e.g. inside srun --pty:
bash src/tools/netmhcpan/run_netmhcpan_parallel.sh \
  data/raw/fasta/a0101_batches data/processed/hla_a0101/netmhcpan \
  hla_a0101 10 HLA-A01:01 9
#    Receipts: logs/netmhcpan/a0101_nmpan_<jobid>.out/.err +
#    per-batch .txt (each carries the binary version banner).
#    Resume-safe: completed batches (csv+txt non-empty) are skipped on
#    resubmit. Runner itself exits 0 even on batch failures — the sbatch
#    wrapper cross-checks output counts and exits 1 if incomplete.
#    hub1 ctools binary = NetMHCpan 4.2f (verified banner: "NetMHCpan
#    version 4.2f"; writes a 3-line csv preamble, EL_score/EL_rank header)
#    vs the legacy local build's 2-line preamble (Score/Rank) — note the
#    build drift in thesis methods. functions.R loaders force column
#    types, so both formats parse identically (legacy unaffected).

# 5. netMHCpan sensitivity evaluation + pos/neg construction
Rscript src/pipeline/r/04_evaluate_netmhcpan_sensitivity.R hla_a0101
#    OUT data/processed/hla_a0101/df_combined_pos_and_neg.csv (+ ledgers)
#    figs results/figures/netmhcpan_hla_a0101/
```

### Stage B — features (R chain + NSP3 + AlphaFold)

```bash
# 6. Flanking-region extraction (FLANK_SIZE=5 default → 19-mer windows)
Rscript src/pipeline/r/05_protein_features.R hla_a0101
#    OUT data/processed/hla_a0101/epitopes_pos_and_neg_features.csv

# 7. NetSurfP-3 (GPU, conda nsp3_gpu)
bash src/tools/netsurfp/nsp3_clean_fasta.sh hla_a0101
#    header → >ACC + length filter 130–5000 aa (legacy recipe)
bash src/tools/netsurfp/prepare_nsp3_batches.sh hla_a0101
#    OUT data/raw/fasta/a0101_nsp3_batches/ (a0101_batch_*.fasta;
#    first file is a0101_batch_.fasta — legacy awk quirk, kept)
#    GPU run — sbatch receipt wrapper (recommended):
sbatch src/run/run_nsp3_a0101_slurm.sh
#    Direct runner call, e.g. inside srun --pty:
bash src/tools/netsurfp/run_netsurfp3.sh data/raw/fasta/a0101_nsp3_batches
#    OUT data/processed/nsp3/a0101_batch_*/ (shared root, prefixed batch dirs)
#    hub1 facts (verified 2026-11-24): NetSurfP checkout + 3.3 GB model in
#    /home/projects1/thesis_s204692_s204581/NetSurfP-3.0_standalone (ctools
#    copy also exists); conda env nsp3_gpu = torch 2.5.1+cu121 in the
#    project partner's /home/people/s204581/ENTER — usable by s204692.
#    ~0.2 s/seq on L40S (Sep 2026 legacy jobs) → ~30 batches ≈ 40 min.
#    GPU type PINNED to L40S in the wrapper: cu121 has NO kernels for the
#    Blackwell nodes (compute07-08). Receipts: logs/netsurfp/*.out/.err.

# 8. NSP3 feature join
Rscript src/pipeline/r/06_netsurfp_protein_features.R hla_a0101
#    OUT data/processed/hla_a0101/epitopes_pos_and_neg_features_with_nsp3.csv

# 9. AlphaFold structures — per-allele fetch workspace, SHARED PDB pool.
#    The shared tree (data/processed/structures/) is owned by the legacy
#    run and read-only for us — and it already covers a0101 (04's ledger:
#    2830/2830 needed PDBs present). So: build the per-allele fetch lists
#    (documentation + invalid-ID record) in our OWN tree; the fetch itself
#    is SKIPPED. 07 reads PDBs from the shared pool (allele-independent,
#    per-UniProt AlphaFold models).
bash src/fetch/prepare_fetch_list.sh \
  data/processed/hla_a0101/pos_EL_all_epitopes_hla_a0101.csv \
  data/processed/hla_a0101/structures a0101
#    OUT data/processed/hla_a0101/structures/logs/
#        fetch_list_canonical_a0101.tsv + fetch_list_isoforms_a0101.tsv +
#        truly_invalid_ids_a0101.tsv
# SKIP: bash src/fetch/fetch_structures.sh ...   (nothing missing; shared
# tree not writable by us anyway — only run it, against a writable dir,
# if a future audit finds genuinely missing PDBs)

# 10. AlphaFold pLDDT features + window dedup
Rscript src/pipeline/r/07_alpha_fold_features.R hla_a0101
#     OUT data/processed/hla_a0101/df_all.csv

# 11. AA encodings (A*01:01 anchors: P2 V/I/T/M, P3 D/E, P9 Y)
Rscript src/pipeline/r/08_aa_encoding.R hla_a0101
#     OUT data/processed/hla_a0101/df_<enc>.csv + df_all_<enc>.csv +
#         compositional
```

Steps 02–08 can also run in one shot once NSP3 csvs exist:
`Rscript src/pipeline/r/run_pipeline.R hla_a0101` (FLANK_SIZE=5 default).

### Stage C — Python external validation (0201 models → 0101 data, no retraining)

```bash
# 12. (only for embedding feature sets) raw H5s on the GPU via the legacy
#     embed tools with redirected IO, then prepare:
THESIS_ALLELE=hla_a0101 python 03_prepare_embeddings.py --embedding <key>

# 13. Score with trained models:
THESIS_ALLELE=hla_a0101 python 14_predict_external.py \
  --run-tag <tag> --model xgb --features blosum50
#     OUT data/processed/hla_a0101/predictions/
```

`14_predict_external.py` refuses to run without `THESIS_ALLELE=hla_a0101`
(the external set must never be scored with models trained on it).

## Artifacts requirement

`04_modelling.py` saves `{run_tag}_outer{i}_inner{j}_artifacts.pkl` next to
each model .pkl (scaler, fold PCA objects, column medians, dimselect dims).
Models trained **before** that additive change have no artifacts file and
must be re-run once with identical settings (deterministic: same data,
folds, seed) before 14_predict_external can replay them.

## Gotchas

- **Wipe fork leftovers before the rerun**: `data/processed/hla_a0101/`,
  `data/raw/fasta/a0101_batches/` (union batches), fork-era
  `hla_a0101_positives_only.fasta`. `load_netmhcpan_batches()` reads ALL
  csvs in its dir — stale files pollute.
- **netMHCpan lengths**: `-l 9` only (legacy used 8–14 for its own run);
  04 filters to 9-mers anyway.
- **AlphaFold coverage caveat**: 07 reads PDBs from the SHARED legacy pool
  (`data/processed/structures/alphafold/`, owned by the legacy run,
  read-only). Coverage was recorded against the LEGACY peptide regions —
  if an a0101 peptide extends beyond a recorded region, 07's pLDDT
  extraction imputes the tail. Check the a0101 missingness report if
  pLDDT gaps appear. Per-allele fetch LISTS live in our own tree
  (`data/processed/hla_a0101/structures/logs/`); a fetch would only be
  needed if an audit finds PDBs genuinely absent from the shared pool.
- **NSP3 coverage**: proteins without an NSP3 run get NA features
  (imputed downstream).
- **Cross-modelling (later)**: train a0101 models via
  `THESIS_ALLELE=hla_a0101` on the a0101 R chain once it exists
  (01_datasplit not needed — external set has no folds), then score the
  0201 data as the external set. Model run-tags must stay distinct in the
  shared `models/` dir.
- **Anchor check** in a0101 08 uses A\*01:01 anchors (P2 V/I/T/M, P3 D/E,
  P9 Y) — deliberately NOT the legacy A\*02:01 anchors.
