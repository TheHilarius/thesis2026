"""
05_cnn_modelling.py
CNN counterpart of 04_modelling.py: same folds/labels/splits, same
logs/<mode> + models/<mode>/cv_results JSON layout, but raw (N,29,D)
windows + torch Conv1d instead of flat PCA + sklearn.

Simplest v1: one embedding_windows component, one SimpleCNN, 6-bucket CV
(train on 5 buckets, validate on 1, rotate incl. held-out) -- same
validation buckets as 04 nested CV but 1 model per bucket, not 5x.

Usage:
    python 05_cnn_modelling.py --model cnn_simple --features emb_esmc300m_win_padtoken
    python 05_cnn_modelling.py --model cnn_simple --features emb_esmif_eos_bos_repeat --mask-mode nomask

Outputs (mirrors 04):
    <out-root>/logs/cnn/05_cnn_modelling_<run_tag>_log_<ts>.txt
    <out-root>/models/cnn/cv_results_<run_tag>_<ts>.json
    <out-root>/models/cnn/<run_tag>_val<b>.pt  (one state_dict per bucket)
"""
import sys
import os
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse
import json
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
    return m


def resolve_window_ds(h5, preferred):
    """Prepared H5s disagree on ds name; try preferred then known fallbacks."""
    candidates = [preferred, "windows", "window_embeddings", "window_if_struct"]
    for name in candidates:
        if name and name in h5:
            return name
    raise KeyError(f"No window dataset found. Tried {candidates}. Keys: {list(h5.keys())}")


class WindowDataset(Dataset):
    """On-demand H5 reads: rows streamed per batch, never full (N,29,D) in RAM."""

    def __init__(self, h5_path, ds_name, mask_name, idx, labels, use_mask=True):
        self.h5_path, self.ds_name, self.mask_name = h5_path, ds_name, mask_name
        self.idx, self.labels = np.asarray(idx), np.asarray(labels, dtype=np.float32)
        self.use_mask = use_mask
        self._h5 = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        r = int(self.idx[i])
        x = self._h5[self.ds_name][r].astype(np.float32).T  # (29,D)->(D,29)
        if self.use_mask and self.mask_name in self._h5:
            m = self._h5[self.mask_name][r].astype(bool)
        else:
            m = np.zeros(x.shape[1], dtype=bool)
        return torch.from_numpy(x), torch.from_numpy(m), torch.tensor(self.labels[i])


def train_one_fold(model, train_loader, val_loader, device, lr, epochs, patience, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc, best_state, bad = -1.0, None, 0
    model.to(device)
    for _ in range(epochs):
        model.train()
        for xb, mb, yb in train_loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb, mb), yb).backward()
            opt.step()
        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for xb, mb, yb in val_loader:
                ps += torch.sigmoid(model(xb.to(device), mb.to(device))).cpu().tolist()
                ys += yb.tolist()
        auc = roc_auc_score(ys, ps)
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def parse_args():
    p = argparse.ArgumentParser(description="05_cnn_modelling: simplest CNN CV + held-out")
    p.add_argument("--model", default="cnn_simple", help="CNN key from 05_cnn_models.CNN_REGISTRY")
    p.add_argument("--features", default="emb_esmc300m_win_padtoken",
                   help="Feature set key from FEATURE_SETS (one embedding_windows component)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=3,
                   help="Early-stop patience on val AUC (epochs without improvement)")
    p.add_argument("--mask-mode", default="mask", choices=["mask", "nomask"],
                   help="mask=zero pads via pad_mask (default); nomask=ignore mask")
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

    with h5py.File(h5_path, "r") as f:
        ds_name = resolve_window_ds(f, prefer_ds)
        labels, folds = f["labels"][:], f["folds"][:]
        emb_dim = int(f.attrs.get("emb_dim", 0)) or int(f[ds_name].shape[2])
        window = int(f.attrs.get("window", 29))

    run_tag = f"{args.model}_{args.features}_cnn"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_root and args.out_root != ".":
        out_base = Path(args.out_root)
    else:
        out_base = LOG_DIR.parent / "runs" / f"{run_tag}_{ts}"
    log_dir, model_dir = out_base / "logs" / "cnn", out_base / "models" / "cnn"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    logger = Logger(str(log_dir / f"05_cnn_modelling_{run_tag}_log_{ts}.txt"))
    sys.stdout = logger

    print("=" * 80)
    print(f"  05_CNN_MODELLING -- {cnn_cfg['display_name'].upper()} + {args.features.upper()}")
    print("=" * 80)
    print(f"  H5: {h5_path}\n  ds={ds_name} mask={mask_name} mode={args.mask_mode}")
    print(f"  N={len(labels)} D={emb_dim} window={window} folds={sorted(map(int, np.unique(folds)))}")
    use_mask = args.mask_mode == "mask"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device={device} epochs={args.epochs} bs={args.batch_size} lr={args.lr} patience={args.patience}")
    print(f"  params: {cnn_cfg['params']}")

    t0 = time.time()
    buckets = list(range(N_CV_FOLDS + 1))  # 0..5 incl. held-out, same buckets as 04
    fold_metrics, fold_preds, fold_times = [], {}, []
    for vb in buckets:
        va = np.where(folds == vb)[0]
        tr = np.where(folds != vb)[0]
        tl = DataLoader(WindowDataset(h5_path, ds_name, mask_name, tr, labels[tr], use_mask),
                        batch_size=args.batch_size, shuffle=True)
        vl = DataLoader(WindowDataset(h5_path, ds_name, mask_name, va, labels[va], use_mask),
                        batch_size=args.batch_size)
        t1 = time.time()
        net = build_cnn(args.model, emb_dim, window)
        n_params = sum(p.numel() for p in net.parameters())
        train_one_fold(net, tl, vl, device, args.lr, args.epochs, args.patience, args.seed + vb)
        net.eval()
        ps, ys = [], []
        with torch.no_grad():
            for xb, mb, yb in vl:
                ps += torch.sigmoid(net(xb.to(device), mb.to(device))).cpu().tolist()
                ys += yb.tolist()
        m = compute_metrics(np.asarray(ys), np.asarray(ps))
        fold_metrics.append(m)
        fold_preds[str(vb)] = {"y_true": ys, "y_prob": ps}
        fold_times.append(round(time.time() - t1, 1))
        print(f"  bucket {vb} AUC={m['auc_roc']:.4f} params={n_params} ({fold_times[-1]}s)")
        torch.save(net.state_dict(), model_dir / f"{run_tag}_val{vb}.pt")

    summary = {k: {"mean": float(np.mean([f[k] for f in fold_metrics])),
                   "std": float(np.std([f[k] for f in fold_metrics]))} for k in fold_metrics[0]}
    print(f"\nCV AUC-ROC: {summary['auc_roc']['mean']:.4f} +/- {summary['auc_roc']['std']:.4f}")

    out = {"timestamp": ts, "runtime_seconds": round(time.time() - t0, 1),
           "fold_seconds": fold_times,
           "mode": "cnn_cv", "model_key": args.model,
           "model_type": cnn_cfg["display_name"], "features_key": args.features,
           "features_display": feat_cfg["display_name"],
           "components": feat_cfg["components"],
           "config": {"n_folds": len(buckets), "mask_mode": args.mask_mode,
                      "epochs": args.epochs, "batch_size": args.batch_size,
                      "lr": args.lr, "patience": args.patience,
                      "hyperparameters": cnn_cfg["params"],
                      "emb_dim": emb_dim, "window": window, "device": device,
                      "random_state": args.seed},
           "cv_summary": summary, "outer_val_metrics": fold_metrics,
           "outer_val_predictions": fold_preds}
    with open(model_dir / f"cv_results_{run_tag}_{ts}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {model_dir}/cv_results_{run_tag}_{ts}.json")
    logger.close()
