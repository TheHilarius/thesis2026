#!/bin/bash
#SBATCH --job-name=esmc6b_embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/esmc6b_embed_%j.out
#SBATCH --error=logs/esmc6b_embed_%j.err

# ESM-C 6B embeddings (separate from the frozen 300m/600m pipeline).
# Uses the new esm 3.4.1 env; the old env_esmc (esm 3.2.1) only knows 300m/600m.

# GPU isolation (SOP: without this PyTorch can span all GPUs and deadlock).
export CUDA_VISIBLE_DEVICES=0

# Navigate to the project root on the compute node
cd /home/projects/thesis_s204692_s204581/thesis2026

# Activate the 6B venv
if [ -f "env_esmc6b/bin/activate" ]; then
    source env_esmc6b/bin/activate
    echo "Activated env_esmc6b"
else
    echo "ERROR: env_esmc6b not found. Create it first (see project docs)."
    exit 1
fi

# Ensure output directories exist
mkdir -p logs
mkdir -p data/processed/embeddings

export PYTHONUNBUFFERED=1

MODES="${MODES:-zero,impute_boundary,impute_bos_eos,pad_token}"

echo "Starting ESM-C 6B embedding ($(date))..."
echo "  Model: esmc_6b (biohub/ESMC-6B)"
echo "  Output: data/processed/embeddings/"
echo "  Modes: $MODES"

python src/tools/esm/embed_peptides_esmc6b.py \
    --out-dir data/processed/embeddings/ \
    --modes "$MODES"

echo "ESM-C 6B embedding completed ($(date))."
