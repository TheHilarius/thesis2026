#!/bin/bash
#SBATCH --job-name=nsp3_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=nsp3_h100_%j.out
#SBATCH --error=nsp3_h100_%j.err

eval "$(conda shell.bash hook)"
conda activate /home/people/s204581/ENTER/envs/nsp3_gpu

# Updated paths
PROJ_DIR="/home/projects/thesis_s204692_s204581/thesis2026"
NSP3_DIR="/home/projects/thesis_s204692_s204581/NetSurfP-3.0_standalone"
BATCH_DIR="$PROJ_DIR/data/raw/fasta/batches"
OUTPUT_BASE="$PROJ_DIR/data/processed/nsp3"

mkdir -p "$OUTPUT_BASE"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running NetSurfP-3.0 on H100..."

for batch_file in "$BATCH_DIR"/batch_*.fasta; do
    batch=$(basename "$batch_file" .fasta)
    
    if [ -d "${OUTPUT_BASE}/${batch}" ] && [ "$(ls "${OUTPUT_BASE}/${batch}"/*.csv 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[SKIP] $batch — already complete"
        continue
    fi

    echo "[RUN] $batch — starting at $(date +%H:%M:%S)"
    python "$NSP3_DIR/nsp3.py" \
        -m "$NSP3_DIR/models/nsp3.pth" \
        -i "$batch_file" \
        -o "${OUTPUT_BASE}/${batch}" \
        -gpu True
done
