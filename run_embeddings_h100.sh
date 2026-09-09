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

# Activate the correct environment
eval "$(conda shell.bash hook)"
conda activate esm_gpu

# Ensure output directories exist
mkdir -p logs
mkdir -p data/processed/embeddings

echo "Starting ESM-C Embedding..."
# Because of OpenCode's changes, we don't even need to pass arguments!
python src/tools/esm/embed_peptides_w_esm.py

echo "Starting ESM-IF Embedding..."
python src/tools/esm/embed_structures_esmif_gpu.py

echo "All embeddings completed successfully."
