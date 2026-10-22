#!/usr/bin/env python3
"""
05b_run_cnn_matrix.py
Mirror of 04b_run_matrix.py for CNN sweeps: runs 05_cnn_modelling.py across
(model, feature_set, epochs) combos and ranks cv_results JSONs.

Usage:
    python 05b_run_cnn_matrix.py --dry-run
    python 05b_run_cnn_matrix.py --out-root runs/cnn_baseline_20261020 \\
        --cnn-sweep cnn_simple:emb_esmc300m_win_padtoken:20 cnn_2layer:emb_esmc300m_win_padtoken:20
"""
import subprocess
import sys
import time
import argparse
import json
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent.parent
CNN_SCRIPT = SRC_DIR / "05_cnn_modelling.py"

DEFAULT_COMBOS = [
    ("cnn_simple", "emb_esmc300m_win_padtoken", 20),
    ("cnn_2layer", "emb_esmc300m_win_padtoken", 20),
    ("cnn_simple", "emb_esmif_eos_bos_repeat", 20),
]


def run_combo(model, features, epochs, dry_run=False, out_root=None):
    cmd = [sys.executable, str(CNN_SCRIPT), "--model", model, "--features", features,
           "--epochs", str(epochs)]
    if out_root and out_root != ".":
        cmd += ["--out-root", out_root]
    print(f"\n  RUNNING: {model} x {features} (epochs={epochs})")
    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return True
    t0 = time.time()
    ok = subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode == 0
    print(f"  {'DONE' if ok else 'FAILED'} in {time.time() - t0:.1f}s")
    return ok


def expand_cnn_sweep(spec):
    model, features, epochs = spec.split(":")
    return model.strip(), features.strip(), int(epochs.strip())


def aggregate(out_root):
    rows = []
    for fp in sorted(Path(out_root).glob("models/cnn/cv_results_*.json")):
        d = json.loads(fp.read_text())
        s = d.get("cv_summary", {})
        rows.append((s.get("auc_roc", {}).get("mean", 0.0),
                     d.get("model_key"), d.get("features_key"), fp.name))
    rows.sort(reverse=True)
    print(f"\n{'CNN RANKING':^60}\n" + "-" * 60)
    for auc, m, f, n in rows:
        print(f"  {auc:.4f}  {m:<10} {f:<35} {n}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default=None)
    p.add_argument("--cnn-sweep", nargs="*", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-aggregate", action="store_true")
    a = p.parse_args()
    combos = [expand_cnn_sweep(s) for s in a.cnn_sweep] if a.cnn_sweep else DEFAULT_COMBOS
    out_root = a.out_root or "."
    ok = all(run_combo(m, f, e, a.dry_run, out_root) for m, f, e in combos)
    if not a.dry_run and not a.no_aggregate and out_root != ".":
        aggregate(out_root)
    sys.exit(0 if ok else 1)
