#!/bin/bash
# run_dimselect_sweep.sh — per-dim supervised screening arms (dimselect).
#
# Supervisor's "50 rows" idea: fold-nested per-dim screening (12_dim_screening),
# keep top-N dims RAW (29 x N, no PCA), final XGB. Arms:
#   emb_esmc300m_win_padtoken (D=960)  x N{25,50,100,200} x screen{xgb,lr_l2}
#   emb_esmif_eos_bos_repeat  (D=512)  x N{25,50,100,200} x screen{xgb,lr_l2}
# = 16 jobs into runs/dimselect_sweep.
#
# Screenings are cached in $OUT_ROOT/screening/ keyed
# (feature, excl_bucket, test_fold, screen) — deterministic, so N-sweep arms
# reuse rankings computed by the first arm of each (embedding, screen) pair.
#
# SUBMISSION ORDER MATTERS (no lock around the cache): submit ONE pilot per
# (embedding, screen) first, let it finish, then the remaining N arms:
#   bash src/run/run_dimselect_sweep.sh --submit --out runs/dimselect_sweep --arm \
#     emb_esmc300m_win_padtoken:50:xgb:12:48:14 \
#     emb_esmc300m_win_padtoken:50:lr_l2:12:32:10 \
#     emb_esmif_eos_bos_repeat:50:xgb:12:32:12 \
#     emb_esmif_eos_bos_repeat:50:lr_l2:12:32:8
# then the rest:
#   bash src/run/run_dimselect_sweep.sh --submit --out runs/dimselect_sweep --arm \
#     emb_esmc300m_win_padtoken:25:xgb:12:48:14 \
#     emb_esmc300m_win_padtoken:100:xgb:12:48:14 \
#     emb_esmc300m_win_padtoken:200:xgb:12:48:14 ... (x4 per emb per screen)
#
# Baselines reused from existing OOF (no rerun): blosum50, ESM-C flat-PCA
# k=2278 (encodings_sweep / pca_sweep runs).
#SBATCH --job-name=dimselect
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

REPO=/home/projects/thesis_s204692_s204581/thesis2026

# ── submitter mode ──
if [[ "${1:-}" == "--submit" ]]; then
    shift
    OUT_ROOT=""
    ARM_SPECS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --out)  OUT_ROOT="$2"; shift 2 ;;
            --arm)  shift; while [[ $# -gt 0 && "$1" != --* ]]; do
                        ARM_SPECS+=("$1"); shift; done ;;
            *) echo "unknown arg: $1" >&2; exit 1 ;;
        esac
    done
    [[ -n "$OUT_ROOT" && ${#ARM_SPECS[@]} -gt 0 ]] || {
        echo "usage: --submit --out DIR --arm feature:topn:screen:cpus:memGB:hours ..." >&2
        exit 1; }
    mkdir -p "$OUT_ROOT"/{models/dimselect,logs/slurm,results,plots,screening}
    for spec in "${ARM_SPECS[@]}"; do
        IFS=: read -r feature topn screen cpus mem hrs <<< "$spec"
        echo "submit feature=$feature topn=$topn screen=$screen  cpus=$cpus mem=${mem}G time=${hrs}h -> $OUT_ROOT"
        sbatch --job-name="dim_${feature}_n${topn}_${screen}" \
            --cpus-per-task="$cpus" --mem="${mem}G" --time="${hrs}:00:00" \
            --output="$OUT_ROOT/logs/slurm/${feature}_n${topn}_${screen}_%j.out" \
            --error="$OUT_ROOT/logs/slurm/${feature}_n${topn}_${screen}_%j.err" \
            --export=ALL,FEATURE="$feature",TOPN="$topn",SCREEN="$screen",OUT_ROOT="$OUT_ROOT" \
            "$0"
    done
    echo "submitted ${#ARM_SPECS[@]} jobs; squeue -u \$USER -n dim_ to monitor"
    exit 0
fi

# ── worker mode (under SLURM) ──
: "${FEATURE:?FEATURE env missing}"; : "${TOPN:?TOPN env missing}"
: "${SCREEN:?SCREEN env missing (xgb|lr_l2)}"; : "${OUT_ROOT:?}"
cd "$REPO"
. env_esmc_server/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p "$OUT_ROOT"/{models/dimselect,logs/slurm,results,screening}
TIME_FILE="$OUT_ROOT/logs/mem_${FEATURE}_n${TOPN}_${SCREEN}.time"
echo "feature=$FEATURE topn=$TOPN screen=$SCREEN out=$OUT_ROOT  ($(date))"

# Peak RSS per run + cgroup peak (JobAcctGather=none on this cluster).
# NOTE: sacct exit codes lie here — trust the 'Success:' line python prints.
/usr/bin/time -v -o "$TIME_FILE" \
    python src/pipeline/python/04_modelling.py \
        --model xgb --features "$FEATURE" \
        --pca-mode dimselect --top-n "$TOPN" --screen-model "$SCREEN" \
        --screen-cache-dir "$OUT_ROOT/screening" \
        --screen-jobs "${SLURM_CPUS_PER_TASK:-12}" \
        --out-root "$OUT_ROOT"

cg=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)
cgroup_peak=$(cat "/sys/fs/cgroup${cg}/memory.peak" 2>/dev/null || echo NA)
peak_kb=$(awk -F': ' '/Maximum resident set size/{print $2}' "$TIME_FILE")
wall=$(awk -F': ' '/Elapsed \(wall clock\) time/{print $2}' "$TIME_FILE")
wall_s=$(echo "$wall" | awk -F: '
    NF==3 {print $1*3600+$2*60+$3}
    NF==2 {print $1*60+$2}
    NF==1 {print $1}')
{
    printf 'feature\ttopn\tscreen\tpeak_rss_kb\tcgroup_peak_bytes\twall_s\n'
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$FEATURE" "$TOPN" "$SCREEN" "$peak_kb" "$cgroup_peak" "$wall_s"
} > "$OUT_ROOT/results/mem_${FEATURE}_n${TOPN}_${SCREEN}.tsv"
echo "mem: peak_rss_kb=$peak_kb cgroup_peak=$cgroup_peak wall_s=$wall_s"
