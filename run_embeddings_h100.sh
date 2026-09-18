#!/bin/bash
#SBATCH --job-name=esm_embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/esm_embed_%j.out
#SBATCH --error=logs/esm_embed_%j.err

# Navigate to the project root on the compute node
cd /home/projects/thesis_s204692_s204581/thesis2026

# Force Python to print instantly and restrict to 1 GPU
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Ensure output directories exist
mkdir -p logs
mkdir -p data/processed/embeddings

echo "Starting ESM-C Embedding..."
# Activate the native ESM-C environment
source env_esmc_server/bin/activate

# FIXED: use --csv and --out flags (script uses argparse, not positional args)
python src/tools/esm/embed_peptides_w_esm.py \
    --csv data/processed/df_all.csv \
    --out data/processed/embeddings/esmc_context_embeddings.h5 \
    --model esmc_600m

deactivate

echo "Starting ESM-IF Embedding..."
# Switch to the native ESM-IF environment
source env_esmif_server/bin/activate

# FIXED: use --csv, --pdb, --out, --af2 flags
# FIXED: --af2-fallback-dir → --af2 (script flag name)
python src/tools/esm/embed_structures_esmif_gpu.py \
    --csv data/processed/df_all.csv \
    --pdb data/processed/structures/alphafold/ \
    --out data/processed/embeddings/esmif_context_embeddings.h5 \
    --af2 data/processed/structures/alphafold/

deactivate
echo "All embeddings completed successfully."
