"""
05_cnn_modelling.py
CNN counterpart of 04_modelling.py: same folds/labels/splits, same
logs/<mode> + models/<mode>/cv_results JSON layout, but raw (N,29,D)
windows + torch Conv1d instead of flat PCA + sklearn.

NESTED CV (mirrors 04): 6 outer iterations x 5 inner folds = 30 models.
Each inner model early-stops on its inner-val AUC; the 5 inner models'
probs are averaged on the held-out outer bucket. Per-epoch train-loss /
val-AUC histories are logged for every inner model + one combined PNG.

Usage:
    python 05_cnn_modelling.py --model cnn_simple --features emb_esmc300m_win_padtoken
    python 05_cnn_modelling.py --model cnn_simple --features emb_esmif_eos_bos_repeat --mask-mode nomask

Outputs (mirrors 04):
    <out-root>/logs/cnn/05_cnn_modelling_<run_tag>_log_<ts>.txt
    <out-root>/models/cnn/cv_results_<run_tag>_<ts>.json
    <out-root>/models/cnn/<run_tag>_outer<o>_inner<i>.pt  (all 30, best flagged in JSON)
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

from config import (SPLIT_DATA_PATH, LOG_DIR, N_CV_FOLDS, HELD_OUT_INDEX,
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

    def __init__(self, h5_path, ds_name, mask_name, idx, labels):
        self.h5_path, self.ds_name, self.mask_name = h5_path, ds_name, mask_name
        self.idx, self.labels = np.asarray(idx), np.asarray(labels, dtype=np.float32)
        self._h5 = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        r = int(self.idx[i])
        x = self._h5[self.ds_name][r].astype(np.float32).T  # (29,D)->(D,29)
        if self.mask_name in self._h5:
            m = self._h5[self.mask_name][r].astype(bool)
        else:
            m = np.zeros(x.shape[1], dtype=bool)
        return torch.from_numpy(x), torch.from_numpy(m), torch.tensor(self.labels[i])


def train_one_inner(model, train_loader, val_loader, device, lr, epochs, patience, seed):
    """Train with early-stop on inner-val AUC. Returns model + per-epoch history."""
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
        for xb, mb, yb in train_loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, mb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        hist["train_loss"].append(tot / max(n, 1))
        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for xb, mb, yb in val_loader:
                ps += torch.sigmoid(model(xb.to(device), mb.to(device))).cpu().tolist()
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
    for xb, mb, yb in loader:
        ps += torch.sigmoid(model(xb.to(device), mb.to(device))).cpu().tolist()
        ys += yb.tolist()
    return np.asarray(ys), np.asarray(ps)


def write_curves_csv(path, curves):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["outer", "inner", "epoch", "train_loss", "val_auc"])
        for o in sorted(curves, key=int):
            for i in sorted(curves[o], key=int):
                h = curves[o][i]
                for e, (tl, va) in enumerate(zip(h["train_loss"], h["val_auc"]), 1):
                    w.writerow([o, i, e, f"{tl:.6f}", f"{va:.6f}"])


def parse_args():
    p = argparse.ArgumentParser(description="05_cnn_modelling: nested CNN CV + held-out")
    p.add_argument("--model", default="cnn_simple", help="CNN key from 05_cnn_models.CNN_REGISTRY")
    p.add_argument("--features", default="emb_esmc300m_win_padtoken",
                   help="Feature set key from FEATURE_SETS (one embedding_windows component)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=3,
                   help="Early-stop patience on inner-val AUC (epochs without improvement)")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--out-root", default=None, help="Campaign folder (mirrors 04 --out-root)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_config()
    feat_cfg = get_feature_set_config(args.features)
    validate_feature_set(feat_cfg)
    cnn_cfg = get_cnn_config(args.model)

    # Emb-only v1: exactly one embedding_windows component
    emb_keys = [c for c in feat_cfg["components"]
                if get_feature_component(c)["type"] == "embedding_windows"]
    assert len(emb_keys) == 1, f"CNN v1 needs 1 embedding_windows component, got {feat_cfg['components']}"
    comp = get_feature_component(emb_keys[0])
    src = get_embedding_source(comp["embedding_key"])
    h5_path = str(src["prepared_path"])
    prefer_ds, mask_name = src["window_ds"], src["pad_mask_ds"]

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
        pad_mask_all = f[mask_name][:] if mask_name in f else None
    n_pad_rows = int((pad_mask_all.any(axis=1)).sum()) if pad_mask_all is not None else 0
    mean_pads = float(pad_mask_all.sum(axis=1).mean()) if pad_mask_all is not None else 0.0
    del pad_mask_all

    run_tag = f"{args.model}_{args.features}_cnn"
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
    print(f"  H5: {h5_path}\n  ds={ds_name} mask={mask_name} (post-conv GAP, always on)")
    print(f"  N={len(labels)} D={emb_dim} window={window} folds={sorted(map(int, np.unique(folds)))}")
    print(f"  padded rows: {n_pad_rows}/{len(labels)} ({n_pad_rows / len(labels) * 100:.1f}%), "
          f"mean pads/row: {mean_pads:.2f}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"  device={device} epochs={args.epochs} bs={args.batch_size} lr={args.lr} patience={args.patience}")
    print(f"  params: {cnn_cfg['params']}")
    if device == "cuda":
        print(f"  gpu_mem_total={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    print(f"\n{'=' * 80}\nNESTED CROSS-VALIDATION  6 outer x 5 inner = 30 models\n{'=' * 80}")
    buckets = list(range(N_CV_FOLDS + 1))
    outer_val_metrics, outer_val_predictions = [], {}
    curves, best_inner, bucket_stats = {}, {}, {}
    fold_times, n_params = [], None
    t_nested_start = time.time()

    for outer in buckets:
        print(f"\n  OUTER {outer}/{N_CV_FOLDS} (validation bucket = {outer})")
        curves[str(outer)] = {}
        inner_folds = [b for b in buckets if b != outer]
        inner_probs, inner_histories = [], []
        best_auc_o, best_i_o = -1.0, -1
        for pos, inner in enumerate(inner_folds):
            t1 = time.time()
            tr_idx = np.where((folds != outer) & (folds != inner))[0]
            va_idx = np.where(folds == inner)[0]
            tl = DataLoader(WindowDataset(h5_path, ds_name, mask_name, tr_idx, labels[tr_idx]),
                            batch_size=args.batch_size, shuffle=True)
            vl = DataLoader(WindowDataset(h5_path, ds_name, mask_name, va_idx, labels[va_idx]),
                            batch_size=args.batch_size)
            net = build_cnn(args.model, emb_dim, window)
            if n_params is None:
                n_params = sum(p.numel() for p in net.parameters())
            net, hist, bauc = train_one_inner(net, tl, vl, device, args.lr, args.epochs,
                                              args.patience, args.seed + outer * 10 + pos)
            curves[str(outer)][str(inner)] = {"train_loss": hist["train_loss"],
                                              "val_auc": hist["val_auc"],
                                              "best_val_auc": bauc,
                                              "epochs_run": len(hist["val_auc"])}
            if bauc > best_auc_o:
                best_auc_o, best_i_o = bauc, inner
            # predict outer bucket now (model in memory, then save + free)
            ou_idx = np.where(folds == outer)[0]
            ol = DataLoader(WindowDataset(h5_path, ds_name, mask_name, ou_idx, labels[ou_idx]),
                            batch_size=args.batch_size)
            _, op = predict_probs(net, ol, device)
            inner_probs.append(op)
            torch.save(net.state_dict(), model_dir / f"{run_tag}_outer{outer}_inner{pos}.pt")
            del net
            fold_times.append(round(time.time() - t1, 1))
            print(f"    inner {pos + 1}/5 (val bucket {inner}) bestAUC={bauc:.4f} "
                  f"ep={len(hist['val_auc'])} ({fold_times[-1]}s)")
        best_inner[str(outer)] = {"bucket": best_i_o, "best_val_auc": round(best_auc_o, 4)}
        y_ou = labels[np.where(folds == outer)[0]].astype(int)
        bucket_stats[str(outer)] = {"n": int(len(y_ou)),
                                    "pos_pct": round(float(y_ou.mean() * 100), 1)}
        avg = np.mean(inner_probs, axis=0)
        m = compute_metrics(y_ou, avg)
        outer_val_metrics.append(m)
        outer_val_predictions[str(outer)] = {"y_true": y_ou.tolist(), "y_prob": avg.tolist()}
        n_pos = int(y_ou.sum())
        print(f"  Outer {outer} validation (n={len(y_ou)}, pos={n_pos}, neg={len(y_ou) - n_pos}):")
        print_metrics(m, prefix="  ")

    t_nested_end = time.time()
    summary = {k: {"mean": float(np.mean([f[k] for f in outer_val_metrics])),
                   "std": float(np.std([f[k] for f in outer_val_metrics])),
                   "values": [float(f[k]) for f in outer_val_metrics]}
               for k in outer_val_metrics[0]}
    print(f"\n{'=' * 80}\nNESTED CV RESULTS (mean +/- std across 6 outer)\n{'=' * 80}")
    for metric, s in summary.items():
        vals = "  ".join(f"{v:.4f}" for v in s["values"])
        print(f"  {metric:<15} {s['mean']:.4f} +/- {s['std']:.4f}    [{vals}]")

    curves_csv = plot_dir / f"learning_curves_{run_tag}_{ts}.csv"
    write_curves_csv(curves_csv, curves)
    meta = {"title": f"{cnn_cfg['display_name']} + {feat_cfg['display_name']} "
                     f"({args.model}, {args.features})",
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
           "nested_cv_seconds": round(t_nested_end - t_nested_start, 1),
           "fold_seconds": fold_times,
           "peak_gpu_mem_gb": round(peak_gpu_gb, 3),
           "mode": "cnn_nested_cv", "model_key": args.model,
           "model_type": cnn_cfg["display_name"], "features_key": args.features,
           "features_display": feat_cfg["display_name"],
           "components": feat_cfg["components"],
           "config": {"n_outer_folds": N_CV_FOLDS + 1, "n_inner_folds": N_CV_FOLDS,
                      "total_models": (N_CV_FOLDS + 1) * N_CV_FOLDS,
                      "mask": "post-conv GAP (always on)", "epochs": args.epochs,
                      "batch_size": args.batch_size, "lr": args.lr, "patience": args.patience,
                      "hyperparameters": cnn_cfg["params"], "n_params": n_params,
                      "emb_dim": emb_dim, "window": window, "device": device,
                      "n_padded_rows": n_pad_rows,
                      "mean_pads_per_row": round(mean_pads, 3),
                      "random_state": args.seed},
           "cv_summary": summary, "outer_val_metrics": outer_val_metrics,
           "outer_val_predictions": outer_val_predictions,
           "best_inner_per_outer": best_inner, "learning_curves": curves,
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
    print(f"\n{'=' * 80}\nRUN SUMMARY (NESTED CV)\n{'=' * 80}")
    print(f"  Total runtime:      {t_end - t_start:.1f}s ({(t_end - t_start) / 60:.1f} min)")
    print(f"  Nested CV time:     {t_nested_end - t_nested_start:.1f}s "
          f"({(t_nested_end - t_nested_start) / 60:.1f} min)")
    print(f"  Peak GPU mem:       {peak_gpu_gb:.2f}GB")
    print(f"  Model:              {cnn_cfg['display_name']} ({args.model}, {n_params} params)")
    print(f"  Feature set:        {feat_cfg['display_name']} ({args.features})")
    print(f"  Outer iterations:   {N_CV_FOLDS + 1}  |  Inner folds: {N_CV_FOLDS}  |  Total: 30")
    print(f"  Val AUC-ROC:        {summary['auc_roc']['mean']:.4f} +/- {summary['auc_roc']['std']:.4f}")
    print(f"  Val MCC:            {summary['mcc']['mean']:.4f} +/- {summary['mcc']['std']:.4f}")
    print(f"  Models dir:         {model_dir}/")
    print(f"  Results file:       {results_path}")
    print(f"  Curves:             {curves_csv} + {curves_png if curves_png else '(no PNG)'}")
    print(f"  Completed:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    logger.close()
