#!/bin/bash
#SBATCH --job-name=esmc_embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/esmc_embed_%j.out
#SBATCH --error=logs/esmc_embed_%j.err

# Navigate to the project root on the compute node
cd /home/projects/thesis_s204692_s204581/thesis2026

# Activate ESM-C venv (server uses venvs, not conda)
if [ -f "env_esmc_server/bin/activate" ]; then
    source env_esmc_server/bin/activate
    echo "Activated env_esmc_server"
elif [ -f "env_esmc/bin/activate" ]; then
    source env_esmc/bin/activate
    echo "Activated env_esmc (local fallback)"
else
    echo "ERROR: No ESM-C venv found"
    exit 1
fi

# Ensure output directories exist
mkdir -p logs
mkdir -p data/processed/embeddings

export PYTHONUNBUFFERED=1

echo "Starting ESM-C 4-mode embedding ($(date))..."
echo "  Output: data/processed/embeddings/"
echo "  Modes: zero, impute_boundary, impute_bos_eos, pad_token"

python src/tools/esm/embed_peptides_w_esm.py \
    --out-dir data/processed/embeddings/

echo "ESM-C embedding completed ($(date))."
