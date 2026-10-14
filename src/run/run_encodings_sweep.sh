#!/bin/bash
# run_encodings_sweep.sh — AA-encoding × embedding comparison (encodings_sweep).
#
# One sbatch per arm; per-arm resources via submitter mode. CPU-only (no GPU).
# Arms: 6 encoding-alone + 6 encoding+ESM-IF-eos/bos-repeat@k632 +
#       6 encoding+ESM-C-300M-padtoken@k2278 = 18 jobs into runs/encodings_sweep.
# Baselines (emb-only @ chosen k) are symlinked separately into
# runs/encodings_sweep/models/flat/ from the pca-sweep run dirs.
#
# Memory: JobAcctGather=none on this cluster, so measure inside the job —
# GNU /usr/bin/time -v (per-run peak RSS, exact) + cgroup memory.peak
# (whole-job high-water mark). Sidecar per arm:
#   <out>/results/mem_<feature>.tsv  (feature, peak_rss_kb, cgroup_peak_bytes, wall_s, cpu_pct)
#
# Submit (interactive shell, repo root) — e.g. solo arms first:
#   bash src/run/run_encodings_sweep.sh --submit --out runs/encodings_sweep --arm \
#     sparse:None:4:8:1  blosum50:None:4:8:1  blosum62:None:4:8:1 \
#     blosum80:None:4:8:1  kidera:None:4:8:1  atchley:None:4:8:1
# then embedding arms (--arm spec = feature:pca:cpus:memGB:hours):
#   bash src/run/run_encodings_sweep.sh --submit --out runs/encodings_sweep --arm \
#     sparse:esmif_eos_bos_repeat=632:12:32:4 \
#     blosum50:esmif_eos_bos_repeat=632:12:32:4  ... (6 total)
#   bash src/run/run_encodings_sweep.sh --submit --out runs/encodings_sweep --arm \
#     sparse:esmc300m_win_padtoken=2278:12:48:6 \
#     blosum50:esmc300m_win_padtoken=2278:12:48:6  ... (6 total)
#SBATCH --job-name=enc_sweep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=06:00:00
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
        echo "usage: --submit --out DIR --arm feature:pca:cpus:memGB:hours ..." >&2
        exit 1; }
    mkdir -p "$OUT_ROOT"/{models/flat,logs/slurm,results,plots,data}
    ln -sfn ../../../data/processed/df_all_with_folds.csv \
        "$OUT_ROOT/data/df_all_with_folds.csv" 2>/dev/null || true
    for spec in "${ARM_SPECS[@]}"; do
        IFS=: read -r feature pca cpus mem hrs <<< "$spec"
        echo "submit feature=$feature pca=$pca  cpus=$cpus mem=${mem}G time=${hrs}h -> $OUT_ROOT"
        sbatch --job-name="enc_${feature}" \
            --cpus-per-task="$cpus" --mem="${mem}G" --time="${hrs}:00:00" \
            --output="$OUT_ROOT/logs/slurm/${feature}_%j.out" \
            --error="$OUT_ROOT/logs/slurm/${feature}_%j.err" \
            --export=ALL,FEATURE="$feature",PCA="$pca",OUT_ROOT="$OUT_ROOT" \
            "$0"
    done
    echo "submitted ${#ARM_SPECS[@]} jobs; squeue -u \$USER -n enc_ to monitor"
    exit 0
fi

# ── worker mode (under SLURM) ──
: "${FEATURE:?FEATURE env missing}"; : "${OUT_ROOT:?}"; : "${PCA:?PCA env missing (use None)}"
cd "$REPO"
. env_esmc_server/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p "$OUT_ROOT"/{models/flat,logs/slurm,results}
TIME_FILE="$OUT_ROOT/logs/mem_${FEATURE}.time"
echo "feature=$FEATURE pca=$PCA out=$OUT_ROOT  ($(date))"

# Peak RSS per run (exact, kernel high-water mark) + whole-job cgroup peak.
/usr/bin/time -v -o "$TIME_FILE" \
    python src/pipeline/python/04b_run_matrix.py --pca-mode flat --no-aggregate \
        --out-root "$OUT_ROOT" --combos "xgb,${FEATURE},${PCA}"

cg=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)
cgroup_peak=$(cat "/sys/fs/cgroup${cg}/memory.peak" 2>/dev/null || echo NA)
peak_kb=$(awk -F': ' '/Maximum resident set size/{print $2}' "$TIME_FILE")
cpu_pct=$(awk -F': ' '/Percent of CPU/{print $2}' "$TIME_FILE" | tr -d '%')
wall=$(awk -F': ' '/Elapsed \(wall clock\) time/{print $2}')
wall_s=$(echo "$wall" | awk -F: '
    NF==3 {print $1*3600+$2*60+$3}
    NF==2 {print $1*60+$2}
    NF==1 {print $1}')
{
    printf 'feature\tpeak_rss_kb\tcgroup_peak_bytes\twall_s\tcpu_pct\n'
    printf '%s\t%s\t%s\t%s\t%s\n' "$FEATURE" "$peak_kb" "$cgroup_peak" "$wall_s" "$cpu_pct"
} > "$OUT_ROOT/results/mem_${FEATURE}.tsv"
echo "mem: peak_rss_kb=$peak_kb cgroup_peak=$cgroup_peak wall_s=$wall_s cpu_pct=$cpu_pct%"
