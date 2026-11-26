#!/usr/bin/env python3
"""
06b_run_ffnn_matrix.py
Mirror of 05b_run_cnn_matrix.py for FFNN sweeps: runs 06a_ffnn_modelling.py
across (model, feature_set, epochs[, width]) combos and ranks cv_results JSONs.

Sweep spec has an optional 4th field WIDTH (FFNN's sweep dimension, the
analog of the CNN channels sweeps): passing it forwards --width to 06a,
which tags the run as _wW and records it in the results JSON.

Usage:
    python 06b_run_ffnn_matrix.py --dry-run
    python 06b_run_ffnn_matrix.py --out-root runs/ffnn_matrix_20261126 \\
        --ffnn-sweep ffnn_simple:emb_esmc300m_win_padtoken:30 ffnn_2layer:emb_esmc300m_win_padtoken:30
    python 06b_run_ffnn_matrix.py --out-root runs/ffnn_width_dense \\
        --ffnn-sweep ffnn_simple:emb_esmc300m_win_padtoken:30:64
"""
import subprocess
import sys
import time
import argparse
import json
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent.parent
FFNN_SCRIPT = SRC_DIR / "06a_ffnn_modelling.py"

# Symmetric with 05b DEFAULT_COMBOS; epochs=30 = FFNN protocol
# (run_ffnn_width_slurm.sh / 06a default), not the CNN's 20.
DEFAULT_COMBOS = [
    ("ffnn_simple", "emb_esmc300m_win_padtoken", 30, None),
    ("ffnn_2layer", "emb_esmc300m_win_padtoken", 30, None),
    ("ffnn_simple", "emb_esmif_eos_bos_repeat", 30, None),
]


def run_combo(model, features, epochs, width=None, dry_run=False, out_root=None):
    cmd = [sys.executable, str(FFNN_SCRIPT), "--model", model, "--features", features,
           "--epochs", str(epochs)]
    if width is not None:
        cmd += ["--width", str(width)]
    if out_root and out_root != ".":
        cmd += ["--out-root", out_root]
    wdesc = f", width={width}" if width is not None else ""
    print(f"\n  RUNNING: {model} x {features} (epochs={epochs}{wdesc})")
    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return True
    t0 = time.time()
    ok = subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode == 0
    print(f"  {'DONE' if ok else 'FAILED'} in {time.time() - t0:.1f}s")
    return ok


def expand_ffnn_sweep(spec):
    """model:features:epochs[:width] -> (model, features, epochs, width|None)."""
    parts = spec.split(":")
    if len(parts) == 3:
        model, features, epochs = parts
        width = None
    elif len(parts) == 4:
        model, features, epochs, width = parts
        width = int(width.strip())
    else:
        raise ValueError(f"Bad --ffnn-sweep spec {spec!r}: use model:features:epochs[:width]")
    return model.strip(), features.strip(), int(epochs.strip()), width


def aggregate(out_root):
    rows = []
    for fp in sorted(Path(out_root).glob("models/ffnn/cv_results_*.json")):
        d = json.loads(fp.read_text())
        s = d.get("cv_summary", {})
        w = (d.get("config") or {}).get("width_override")
        rows.append((s.get("auc_roc", {}).get("mean", 0.0),
                     d.get("model_key"), w, d.get("features_key"), fp.name))
    rows.sort(reverse=True)
    print(f"\n{'FFNN RANKING':^60}\n" + "-" * 60)
    for auc, m, w, f, n in rows:
        wtag = f" w={w}" if w is not None else ""
        print(f"  {auc:.4f}  {m:<14}{wtag:<8} {f:<35} {n}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default=None)
    p.add_argument("--ffnn-sweep", nargs="*", default=None,
                   help="specs: model:features:epochs[:width]")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-aggregate", action="store_true")
    a = p.parse_args()
    combos = ([expand_ffnn_sweep(s) for s in a.ffnn_sweep]
              if a.ffnn_sweep else DEFAULT_COMBOS)
    out_root = a.out_root or "."
    ok = all(run_combo(m, f, e, w, a.dry_run, out_root) for m, f, e, w in combos)
    if not a.dry_run and not a.no_aggregate and out_root != ".":
        aggregate(out_root)
    sys.exit(0 if ok else 1)
