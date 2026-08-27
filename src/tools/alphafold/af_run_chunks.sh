#!/usr/bin/env bash
set -euo pipefail

export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_FORCE_GPU_ALLOW_GROWTH=true

/home/hilarius/tools/alphafold_localcolabfold/.pixi/envs/default/bin/colabfold_batch \
    --num-models 1 \
    --num-recycle 3 \
    --max-extra-seq 512 \
    --max-seq 128 \
    --use-pallas \
    --stop-at-score 85 \
    --random-seed 0 \
    data/processed/alphafold_output/chunks/all_chunks.fasta \
    data/processed/alphafold_output/chunks/ \
    2>&1 | tee data/processed/af_chunks_output.log
