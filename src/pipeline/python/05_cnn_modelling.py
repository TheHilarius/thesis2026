"""
05_cnn_modelling.py
CNN counterpart of 04_modelling.py: same folds/labels/splits, same
logs/<mode> + models/<mode>/cv_results JSON layout, but raw (N,L,D)
windows + torch Conv1d instead of flat PCA + sklearn.

CANONICAL PROTOCOL (from Nov 2026): flank5-19mer embeddings via
THESIS_VARIANT=flank5 (L=19), 30 epochs / patience 5 / per-fold seeds.
Proven better than full-29mer (k5 flank5 0.6674 > full 0.6536) and than
short 20/3 budget (stops in flat noise). Full-29 (L=29) + 20/3 fixed-seed
is legacy record only, not for new runs.

SIMPLE 6-FOLD CV (experimentation): 6 buckets, 6 models.
Each bucket serves once as validation, the other 5 as train.
Seeds differ per fold by default (base seed + bucket id); --seed-mode fixed
restores the legacy single-seed protocol.
Per-epoch train-loss / val-AUC histories logged per bucket + one PNG.

Usage:
    python 05_cnn_modelling.py --model cnn_simple --features emb_esmc300m_win_padtoken
    python 05_cnn_modelling.py --model cnn_2layer --features emb_esmif_eos_bos_repeat
    python 05_cnn_modelling.py --model cnn_simple_k7 --channels 64  (channel sweep override)

Pooling is naive GAP over all L slots (L=29 full, L=19 flank5; no pad mask).

Outputs (mirrors 04):
    <out-root>/logs/cnn/05_cnn_modelling_<run_tag>_log_<ts>.txt
    <out-root>/models/cnn/cv_results_<run_tag>_<ts>.json
    <out-root>/models/cnn/<run_tag>_val<vb>.pt  (6 models)
    <out-root>/plots/learning_curves_<run_tag>_<ts>.png (+ .csv + _meta.json)

Plotting lives in 05_plotting.py (called via subprocess each run; also
runnable standalone to re-plot any old curves CSV).
"""
import sys
import os
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             f1_score, matthews_corrcoef, confusion_matrix)

from config import (LOG_DIR, N_CV_FOLDS,
                    RANDOM_STATE, get_feature_set_config, get_feature_component,
                    get_embedding_source, validate_feature_set, validate_config)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("cnn_zoo", str(Path(SRC_DIR) / "05_cnn_models.py"))
_cnn_zoo = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cnn_zoo)
get_cnn_config, build_cnn = _cnn_zoo.get_cnn_config, _cnn_zoo.build_cnn

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("FATAL: torch not installed. Activate env_esmc_server / GPU env.")
    sys.exit(1)


# ── Logger (copy of 04_modelling.Logger) ──
class Logger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    m = {"auc_roc": roc_auc_score(y_true, y_prob),
         "auc_pr": average_precision_score(y_true, y_prob),
         "accuracy": accuracy_score(y_true, y_pred),
         "f1": f1_score(y_true, y_pred),
         "mcc": matthews_corrcoef(y_true, y_pred)}
    tn, fp, fn, tp = confusion_matrix(y_true, y_prob >= 0.5).ravel()
    m["sensitivity"] = tp / (tp + fn) if (tp + fn) else 0.0
    m["specificity"] = tn / (tn + fp) if (tn + fp) else 0.0
    m["ppv"] = tp / (tp + fp) if (tp + fp) else 0.0
    m["npv"] = tn / (tn + fn) if (tn + fn) else 0.0
    return m


def print_metrics(metrics, prefix=""):
    print(f"{prefix}  AUC-ROC: {metrics['auc_roc']:.4f}  |  AUC-PR: {metrics['auc_pr']:.4f}  |  "
          f"Acc: {metrics['accuracy']:.4f}  |  F1: {metrics['f1']:.4f}  |  "
          f"MCC: {metrics['mcc']:.4f}")
    print(f"{prefix}  Sens: {metrics['sensitivity']:.4f}  |  Spec: {metrics['specificity']:.4f}  |  "
          f"PPV: {metrics['ppv']:.4f}  |  NPV: {metrics['npv']:.4f}")


def resolve_window_ds(h5, preferred):
    """Prepared H5s disagree on ds name; try preferred then known fallbacks."""
    candidates = [preferred, "windows", "window_embeddings", "window_if_struct"]
    for name in candidates:
        if name and name in h5:
            return name
    raise KeyError(f"No window dataset found. Tried {candidates}. Keys: {list(h5.keys())}")


class WindowDataset(Dataset):
    """On-demand H5 reads: rows streamed per batch, never full (N,29,D) in RAM."""

    def __init__(self, h5_path, ds_name, idx, labels):
        self.h5_path, self.ds_name = h5_path, ds_name
        self.idx, self.labels = np.asarray(idx), np.asarray(labels, dtype=np.float32)
        self._h5 = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        r = int(self.idx[i])
        x = self._h5[self.ds_name][r].astype(np.float32).T  # (29,D)->(D,29)
        return torch.from_numpy(x), torch.tensor(self.labels[i])


def train_one_fold(model, train_loader, val_loader, device, lr, epochs, patience, seed):
    """Train with early-stop on val AUC. Seed passed in per fold. Returns model + history."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc, best_state, bad = -1.0, None, 0
    hist = {"train_loss": [], "val_auc": []}
    model.to(device)
    for _ in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        hist["train_loss"].append(tot / max(n, 1))
        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                ps += torch.sigmoid(model(xb.to(device))).cpu().tolist()
                ys += yb.tolist()
        auc = roc_auc_score(ys, ps)
        hist["val_auc"].append(auc)
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist, best_auc


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    ps, ys = [], []
    for xb, yb in loader:
        ps += torch.sigmoid(model(xb.to(device))).cpu().tolist()
        ys += yb.tolist()
    return np.asarray(ys), np.asarray(ps)


def write_curves_csv(path, curves):
    """Flat 6-fold format: {bucket: hist}. One row per bucket/epoch."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "epoch", "train_loss", "val_auc"])
        for b in sorted(curves, key=int):
            h = curves[b]
            for e, (tl, va) in enumerate(zip(h["train_loss"], h["val_auc"]), 1):
                w.writerow([b, e, f"{tl:.6f}", f"{va:.6f}"])


def parse_args():
    p = argparse.ArgumentParser(description="05_cnn_modelling: simple 6-fold CNN CV")
    p.add_argument("--model", default="cnn_simple", help="CNN key from 05_cnn_models.CNN_REGISTRY")
    p.add_argument("--features", default="emb_esmc300m_win_padtoken",
                   help="Feature set key from FEATURE_SETS (one embedding_windows component)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=3,
                   help="Early-stop patience on val AUC (epochs without improvement)")
    p.add_argument("--seed", type=int, default=RANDOM_STATE,
                     help="Base seed; per-fold seed = base + bucket unless --seed-mode fixed")
    p.add_argument("--seed-mode", choices=["perfold", "fixed"], default="perfold",
                     help="perfold: seed 42+bucket per fold (default); fixed: same seed all folds (legacy)")
    p.add_argument("--channels", type=int, default=None,
                   help="Override registry channels for channel sweeps; "
                        "recorded in run_tag as _cC and in the results JSON")
    p.add_argument("--out-root", default=None, help="Campaign folder (mirrors 04 --out-root)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_config()
    feat_cfg = get_feature_set_config(args.features)
    validate_feature_set(feat_cfg)
    cnn_cfg = get_cnn_config(args.model)
    eff_params = dict(cnn_cfg["params"])  # copy: registry dict stays untouched
    if args.channels is not None:
        eff_params["channels"] = args.channels

    # Emb-only v1: exactly one embedding_windows component
    emb_keys = [c for c in feat_cfg["components"]
                if get_feature_component(c)["type"] == "embedding_windows"]
    assert len(emb_keys) == 1, f"CNN v1 needs 1 embedding_windows component, got {feat_cfg['components']}"
    comp = get_feature_component(emb_keys[0])
    src = get_embedding_source(comp["embedding_key"])
    h5_path = str(src["prepared_path"])
    prefer_ds = src["window_ds"]

    if not Path(h5_path).exists():
        print(f"FATAL: prepared H5 missing: {h5_path}")
        print(f"  Build it: python src/pipeline/python/03_prepare_embeddings.py "
              f"--embedding {comp['embedding_key']}")
        sys.exit(1)

    with h5py.File(h5_path, "r") as f:
        ds_name = resolve_window_ds(f, prefer_ds)
        labels, folds = f["labels"][:], f["folds"][:]
        emb_dim = int(f.attrs.get("emb_dim", 0)) or int(f[ds_name].shape[2])
        window = int(f.attrs.get("window", 29))

    run_tag = f"{args.model}_{args.features}_cnn"
    if args.channels is not None:
        run_tag = f"{args.model}_c{args.channels}_{args.features}_cnn"  # file separation per C
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_root and args.out_root != ".":
        out_base = Path(args.out_root)
    else:
        out_base = LOG_DIR.parent / "runs" / f"{run_tag}_{ts}"
    log_dir, model_dir = out_base / "logs" / "cnn", out_base / "models" / "cnn"
    plot_dir = out_base / "plots"
    for d in (log_dir, model_dir, plot_dir):
        os.makedirs(d, exist_ok=True)
    logger = Logger(str(log_dir / f"05_cnn_modelling_{run_tag}_log_{ts}.txt"))
    sys.stdout = logger

    t_start = time.time()
    print("=" * 80)
    print(f"  05_CNN_MODELLING -- {cnn_cfg['display_name'].upper()} + {args.features.upper()}")
    print("=" * 80)
    print(f"  Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  H5: {h5_path}\n  ds={ds_name} (naive GAP over all 29, no mask)")
    print(f"  N={len(labels)} D={emb_dim} window={window} folds={sorted(map(int, np.unique(folds)))}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"  device={device} epochs={args.epochs} bs={args.batch_size} lr={args.lr} patience={args.patience}")
    print(f"  params: {eff_params}" + ("  (channels override)" if args.channels is not None else ""))
    if device == "cuda":
        print(f"  gpu_mem_total={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    seed_desc = (f"fixed seed {args.seed}" if args.seed_mode == "fixed"
                 else f"per-fold seeds (base {args.seed} + bucket id)")
    print(f"\n{'=' * 80}\nSIMPLE 6-FOLD CV  6 buckets = 6 models ({seed_desc})\n{'=' * 80}")
    buckets = list(range(N_CV_FOLDS + 1))
    outer_val_metrics, outer_val_predictions = [], {}
    curves, bucket_stats = {}, {}
    fold_times, n_params = [], None
    t_cv_start = time.time()

    for vb in buckets:
        t1 = time.time()
        seed_fold = args.seed if args.seed_mode == "fixed" else args.seed + int(vb)
        print(f"\n  BUCKET {vb}/{N_CV_FOLDS} (validation bucket = {vb}, seed = {seed_fold})")
        tr_idx = np.where(folds != vb)[0]
        va_idx = np.where(folds == vb)[0]
        tl = DataLoader(WindowDataset(h5_path, ds_name, tr_idx, labels[tr_idx]),
                        batch_size=args.batch_size, shuffle=True)
        vl = DataLoader(WindowDataset(h5_path, ds_name, va_idx, labels[va_idx]),
                        batch_size=args.batch_size)
        net = build_cnn(args.model, emb_dim, window, channels_override=args.channels)
        if n_params is None:
            n_params = sum(p.numel() for p in net.parameters())
        net, hist, bauc = train_one_fold(net, tl, vl, device, args.lr, args.epochs,
                                         args.patience, seed_fold)
        curves[str(vb)] = {"train_loss": hist["train_loss"],
                           "val_auc": hist["val_auc"],
                           "best_val_auc": bauc,
                           "epochs_run": len(hist["val_auc"]),
                           "seed": seed_fold}
        torch.save(net.state_dict(), model_dir / f"{run_tag}_val{vb}.pt")
        y_va = labels[va_idx].astype(int)
        bucket_stats[str(vb)] = {"n": int(len(y_va)),
                                 "pos_pct": round(float(y_va.mean() * 100), 1)}
        _, probs = predict_probs(net, vl, device)
        del net
        m = compute_metrics(y_va, probs)
        outer_val_metrics.append(m)
        outer_val_predictions[str(vb)] = {"y_true": y_va.tolist(), "y_prob": probs.tolist()}
        fold_times.append(round(time.time() - t1, 1))
        n_pos = int(y_va.sum())
        print(f"  Bucket {vb} validation (n={len(y_va)}, pos={n_pos}, neg={len(y_va) - n_pos}) "
              f"bestAUC={bauc:.4f} ep={len(hist['val_auc'])} ({fold_times[-1]}s):")
        print_metrics(m, prefix="  ")

    t_cv_end = time.time()
    summary = {k: {"mean": float(np.mean([f[k] for f in outer_val_metrics])),
                   "std": float(np.std([f[k] for f in outer_val_metrics])),
                   "values": [float(f[k]) for f in outer_val_metrics]}
               for k in outer_val_metrics[0]}
    print(f"\n{'=' * 80}\nCV RESULTS (mean +/- std across 6 buckets)\n{'=' * 80}")
    for metric, s in summary.items():
        vals = "  ".join(f"{v:.4f}" for v in s["values"])
        print(f"  {metric:<15} {s['mean']:.4f} +/- {s['std']:.4f}    [{vals}]")

    curves_csv = plot_dir / f"learning_curves_{run_tag}_{ts}.csv"
    write_curves_csv(curves_csv, curves)
    model_desc = args.model if args.channels is None else f"{args.model} C={args.channels}"
    meta = {"title": f"{cnn_cfg['display_name']} + {feat_cfg['display_name']} "
                     f"({model_desc}, {args.features})",
            "buckets": bucket_stats}
    meta_path = plot_dir / f"learning_curves_{run_tag}_{ts}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    curves_png = plot_dir / f"learning_curves_{run_tag}_{ts}.png"
    try:
        subprocess.run([sys.executable, str(Path(SRC_DIR) / "05_plotting.py"),
                        "--csv", str(curves_csv), "--png", str(curves_png),
                        "--meta", str(meta_path), "--title", meta["title"]],
                       check=True, capture_output=True, text=True)
        print(f"  curves: {curves_csv} + {curves_png}")
    except Exception as e:
        print(f"  WARNING: plotting failed ({e}); CSV + meta kept, re-plot standalone:")
        print(f"    python 05_plotting.py --csv {curves_csv} --png {curves_png} --meta {meta_path}")
        curves_png = None

    t_results = time.time()
    peak_gpu_gb = (torch.cuda.max_memory_allocated() / 1e9
                   if device == "cuda" else 0.0)
    out = {"timestamp": ts, "runtime_seconds": round(t_results - t_start, 1),
           "cv_seconds": round(t_cv_end - t_cv_start, 1),
           "fold_seconds": fold_times,
           "peak_gpu_mem_gb": round(peak_gpu_gb, 3),
           "mode": "cnn_cv", "model_key": args.model,
           "model_type": cnn_cfg["display_name"], "features_key": args.features,
           "features_display": feat_cfg["display_name"],
           "components": feat_cfg["components"],
           "config": {"n_folds": N_CV_FOLDS + 1,
                      "total_models": N_CV_FOLDS + 1,
                      "mask": "naive GAP (no mask)", "epochs": args.epochs,
                      "batch_size": args.batch_size, "lr": args.lr, "patience": args.patience,
                      "hyperparameters": eff_params, "n_params": n_params,
                      "channels_override": args.channels,
                      "emb_dim": emb_dim, "window": window, "device": device,
                      "random_state": args.seed, "seed_mode": args.seed_mode},
           "cv_summary": summary, "outer_val_metrics": outer_val_metrics,
           "outer_val_predictions": outer_val_predictions,
           "learning_curves": curves,
           "curves_csv": str(curves_csv),
           "curves_png": str(curves_png) if curves_png else None,
           "curves_meta": str(meta_path)}
    results_path = model_dir / f"cv_results_{run_tag}_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved: {results_path}")

    t_end = time.time()
    peak_gpu_gb = (torch.cuda.max_memory_allocated() / 1e9
                   if device == "cuda" else 0.0)
    print(f"\n{'=' * 80}\nRUN SUMMARY (6-FOLD CV)\n{'=' * 80}")
    print(f"  Total runtime:      {t_end - t_start:.1f}s ({(t_end - t_start) / 60:.1f} min)")
    print(f"  CV time:            {t_cv_end - t_cv_start:.1f}s "
          f"({(t_cv_end - t_cv_start) / 60:.1f} min)")
    print(f"  Peak GPU mem:       {peak_gpu_gb:.2f}GB")
    print(f"  Model:              {cnn_cfg['display_name']} ({args.model}, {n_params} params)")
    print(f"  Feature set:        {feat_cfg['display_name']} ({args.features})")
    print(f"  Folds:              {N_CV_FOLDS + 1}  |  Total models: {N_CV_FOLDS + 1}")
    print(f"  Val AUC-ROC:        {summary['auc_roc']['mean']:.4f} +/- {summary['auc_roc']['std']:.4f}")
    print(f"  Val MCC:            {summary['mcc']['mean']:.4f} +/- {summary['mcc']['std']:.4f}")
    print(f"  Models dir:         {model_dir}/")
    print(f"  Results file:       {results_path}")
    print(f"  Curves:             {curves_csv} + {curves_png if curves_png else '(no PNG)'}")
    print(f"  Completed:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    logger.close()
