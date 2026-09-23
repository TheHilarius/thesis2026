#!/usr/bin/env python3
"""
03b_run_matrix.py
Run 03_prepare_embeddings.py across a list of embedding keys and aggregate
the prepared-artifact summary into one table.

Usage:
    python 03b_run_matrix.py                          # default window keys
    python 03b_run_matrix.py --dry-run                # print only
    python 03b_run_matrix.py --keys esmif_zero esmif_padtoken esmif_eosrepeat
"""

import subprocess
import sys
import time
import argparse
import h5py
from pathlib import Path
from datetime import datetime

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent.parent
PREPARE_SCRIPT = SRC_DIR / "03_prepare_embeddings.py"

sys.path.insert(0, str(SRC_DIR))
from config import get_embedding_source

# Default window keys (all esmif + esmc fixed-29 window modes).
DEFAULT_KEYS = [
    "esmif_zero", "esmif_padtoken", "esmif_eosrepeat",
    "esmc_win_zeropad", "esmc_win_padtoken",
    "esmc_win_impute_bos_eos", "esmc_win_impute_boundary",
]


def run_key(key, dry_run=False):
    """Run 03_prepare_embeddings.py for one embedding key."""
    cmd = [sys.executable, str(PREPARE_SCRIPT), "--embedding", key]
    print(f"\n{'_' * 60}")
    print(f"  PREPARING: {key}")
    print(f"{'_' * 60}")

    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        return True

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"  DONE in {elapsed:.1f}s")
    else:
        print(f"  FAILED (exit code {result.returncode})")
    return result.returncode == 0


def summarize_prepared(key):
    """Read a prepared window HDF5 and return a summary dict (or None)."""
    try:
        src = get_embedding_source(key)
        path = src["prepared_path"]
    except Exception:
        return None

    if not path.exists():
        return None

    try:
        with h5py.File(path, "r") as f:
            return {
                "key": key,
                "n_samples": int(f.attrs.get("n_samples", -1)),
                "emb_dim": int(f.attrs.get("emb_dim", -1)),
                "pad_mode": f.attrs.get("pad_mode", "?"),
                "window": int(f.attrs.get("window", -1)),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
                "has_windows": "windows" in f,
                "has_pad_counts": "pad_counts" in f,
            }
    except Exception as e:
        return {"key": key, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run 03_prepare_embeddings across embedding keys.",
    )
    parser.add_argument(
        "--keys", nargs="*", default=None,
        help="Embedding keys to prepare (default: esmif_zero/padtoken/eosrepeat).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without running them.",
    )
    args = parser.parse_args()

    keys = args.keys if args.keys is not None else DEFAULT_KEYS

    print("=" * 60)
    print("  PREPARE-EMBEDDINGS MATRIX RUNNER")
    print("=" * 60)
    print(f"  Keys:  {len(keys)}")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.dry_run:
        print(f"  MODE:  DRY RUN")
    print("=" * 60)

    t_start = time.time()
    successes = 0
    failures = 0
    summaries = []

    for key in keys:
        ok = run_key(key, dry_run=args.dry_run)
        if ok:
            successes += 1
        else:
            failures += 1
        if not args.dry_run:
            summaries.append(summarize_prepared(key))

    t_end = time.time()

    print(f"\n{'=' * 60}")
    print(f"  MATRIX RUN COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total:   {successes + failures} keys")
    print(f"  Success: {successes}")
    print(f"  Failed:  {failures}")
    print(f"  Runtime: {t_end - t_start:.1f}s")

    if summaries and any(s is not None for s in summaries):
        print(f"\n  {'Key':<20} {'N':>7} {'Dim':>5} {'Window':>7} "
              f"{'Pad mode':<12} {'Size MB':>9}")
        print(f"  {'-' * 66}")
        for s in summaries:
            if s is None:
                continue
            if "error" in s:
                print(f"  {s['key']:<20} ERROR: {s['error']}")
                continue
            print(f"  {s['key']:<20} {s['n_samples']:>7} {s['emb_dim']:>5} "
                  f"{s['window']:>7} {s['pad_mode']:<12} {s['size_mb']:>9.1f}")
    print()


if __name__ == "__main__":
    main()
