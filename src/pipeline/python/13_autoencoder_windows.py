#!/usr/bin/env python3
"""
13_autoencoder_windows.py
Fold-nested 1x1-conv bottleneck autoencoder for (N, 29, D) window
embeddings — the non-linear counterpart of flat-window PCA
(--pca-mode autoencoder in 04_modelling.py).

Architecture: a shared per-position Linear(D -> hidden -> latent). Linear
on the last dim of (B, 29, D) is mathematically identical to a 1x1 Conv1d,
so every one of the 29 slots (peptide core AND flanks) is compressed by
the SAME mapping — the position axis is never mixed and never collapses.
Decoder mirrors the encoder and is discarded after training.

Leak-free by construction: train_encoder receives ONLY train_indices
(prepare_fold's train split — held-out bucket and test fold already
excluded). Early stopping uses a random 10% slice of those train rows;
no labels are involved anywhere (reconstruction only), so homology
separation of the supervised model is untouched.

Pad handling: padded slots (pad_mask True) are excluded from the masked
MSE loss AND from the normalisation denominator — the encoder never
learns to reconstruct filler (padtoken / BOS-EOS vectors).

Usage (normally via 04_modelling.py):
    python 04_modelling.py --pca-mode autoencoder --ae-latent 64 ...

Standalone smoke test:
    python 13_autoencoder_windows.py --key esmc300m_win_padtoken \
        --latent 64 --epochs 3
"""

import argparse
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import h5py
from pathlib import Path

from config import RANDOM_STATE

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - import-time hint only
    torch = None
    nn = None

CHUNK = 1024
DEFAULT_EPOCHS = 15
DEFAULT_LR = 1e-3
DEFAULT_BATCH = 256
DEFAULT_VAL_FRAC = 0.1
DEFAULT_PATIENCE = 3


def _require_torch():
    if torch is None:
        raise SystemExit(
            "torch not installed — autoencoder mode needs it. "
            "Activate env_esmc_server / the GPU env.")


class WindowAE(nn.Module):
    """Shared per-position bottleneck: encoder D -> latent, decoder back.

    Layout (B, 29, D) throughout; Linear applies to the last dim, i.e. a
    1x1 conv with shared weights across all 29 positions.
    """

    def __init__(self, dim, latent, hidden=None):
        super().__init__()
        _require_torch()
        hidden = hidden or max(dim // 2, latent)
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, dim),
        )

    def encode(self, x):
        return self.encoder(x)          # (B, 29, D) -> (B, 29, latent)

    def forward(self, x):
        return self.decoder(self.encode(x))


class TrainedEncoder:
    """Replay wrapper stored in fold_artifacts['pca_dict'].

    Duck-typed by 04_modelling._transform_embeddings (hasattr
    'encode_indices'), so validation / held-out replay uses exactly the
    encoder fitted on that fold's train rows.
    """

    def __init__(self, module, latent, device):
        self.module = module.eval().to(device)
        self.latent = int(latent)
        self.device = device

    def encode_indices(self, emb_data, indices):
        """(n, 29, D) rows for `indices` -> (n, 29, latent) float32."""
        prepared_path = emb_data["prepared_path"]
        n = len(indices)
        out = np.empty((n, emb_data["window"], self.latent), dtype=np.float32)
        cur = 0
        with h5py.File(prepared_path, "r") as f:
            W = f["windows"]
            for i0 in range(0, n, CHUNK):
                idx = indices[i0:i0 + CHUNK]
                block = torch.from_numpy(W[idx].astype(np.float32))
                with torch.no_grad():
                    # module is the encoder Sequential (nn.Module call)
                    lat = self.module(block.to(self.device))
                c = lat.shape[0]
                out[cur:cur + c] = lat.cpu().numpy()
                cur += c
        return out


def load_windows(prepared_path, indices, chunk=CHUNK):
    """Stream (n, 29, D) float32 + (n, 29) bool pad_mask (True = PAD)."""
    xs, ms = [], []
    with h5py.File(prepared_path, "r") as f:
        W = f["windows"]
        M = f["pad_mask"]
        for i0 in range(0, len(indices), chunk):
            idx = indices[i0:i0 + chunk]
            xs.append(W[idx].astype(np.float32))
            ms.append(M[idx].astype(bool))
    return np.concatenate(xs, axis=0), np.concatenate(ms, axis=0)


def masked_mse(out, x, real):
    """MSE over real (non-pad) slots only; real: (B, 29) bool."""
    se = ((out - x) ** 2).sum(-1)               # (B, 29) summed over D
    denom = real.sum().clamp(min=1)
    return (se * real.float()).sum() / denom


def train_encoder(emb_data, train_indices, latent,
                  epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR,
                  batch=DEFAULT_BATCH, val_frac=DEFAULT_VAL_FRAC,
                  patience=DEFAULT_PATIENCE, seed=RANDOM_STATE,
                  device=None):
    """Train the bottleneck AE on train_indices ONLY. Returns
    (TrainedEncoder, log_dict). Decoder weights are discarded."""

    _require_torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    prepared_path = emb_data["prepared_path"]
    D = emb_data["emb_dim"]
    x_all, m_all = load_windows(prepared_path, train_indices)
    n = x_all.shape[0]

    # Early-stop slice: random 10% of train rows (reconstruction only, no
    # labels — kept in-distribution on purpose; test fold never touched)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    x_tr = torch.from_numpy(x_all[tr_idx])
    m_tr = torch.from_numpy(m_all[tr_idx])
    x_val = torch.from_numpy(x_all[val_idx])
    m_val = torch.from_numpy(m_all[val_idx])
    real_val = ~m_val
    del x_all, m_all

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = WindowAE(D, latent).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    n_tr = x_tr.shape[0]
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    epochs_run = 0

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(n_tr)
        for i0 in range(0, n_tr, batch):
            bi = order[i0:i0 + batch]
            xb = x_tr[bi].to(device)
            real = (~m_tr[bi]).to(device)
            opt.zero_grad()
            loss = masked_mse(model(xb), xb, real)
            loss.backward()
            opt.step()
        epochs_run = epoch + 1

        model.eval()
        with torch.no_grad():
            val_loss = masked_mse(
                model(x_val.to(device)), x_val.to(device),
                real_val.to(device)).item()
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch - best_epoch >= patience:
            break

    # reload best weights, discard decoder
    if best_state is not None:
        model.load_state_dict(best_state)
    encoder_only = WindowAE(D, latent).to(device)
    encoder_only.encoder.load_state_dict(model.encoder.state_dict())

    log = {
        "train_rows": int(n_tr),
        "val_rows": int(n_val),
        "best_val_masked_mse": float(best_val),
        "best_epoch": int(best_epoch),
        "epochs_run": int(epochs_run),
        "latent": int(latent),
        "device": device,
    }
    return TrainedEncoder(encoder_only.encoder, latent, device), log


def encode_block(encoder, emb_data, indices):
    """Convenience: (n, 29, latent) float32 for `indices`."""
    return encoder.encode_indices(emb_data, indices)


# ──────────────────────────────────────────────
# Standalone smoke test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from config import get_embedding_source

    parser = argparse.ArgumentParser(description="AE smoke test")
    parser.add_argument("--key", required=True)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    src = get_embedding_source(args.key)
    prepared = src.get("prepared_path")
    if prepared is None or not Path(prepared).exists():
        raise SystemExit(f"prepared H5 missing for {args.key} ({prepared})")

    with h5py.File(prepared, "r") as f:
        n = int(f.attrs.get("n_samples", len(f["labels"])))
        folds = f["folds"][:]

    # fold-0 rows as a fake train split; fold-1 rows test generalisation
    tr = np.where(folds != 1)[0]
    te = np.where(folds == 1)[0]
    emb_data = {"prepared_path": str(prepared), "window": 29,
                "emb_dim": int(src["emb_dim"])}

    enc, log = train_encoder(emb_data, tr, args.latent, epochs=args.epochs)
    print("train log:", log)

    z_tr = encode_block(enc, emb_data, tr[:256])
    z_te = encode_block(enc, emb_data, te[:256])
    print(f"encode shapes: train {z_tr.shape}, test {z_te.shape}")
    assert z_tr.shape == (256, 29, args.latent)
    assert np.isfinite(z_tr).all()

    # zero-baseline check: trained AE must beat predicting all zeros
    x_tr, m_tr = load_windows(str(prepared), tr[:256])
    real = ~m_tr
    zero_mse = float((((x_tr) ** 2).sum(-1) * real).sum() / real.sum())
    print(f"zero-baseline masked MSE: {zero_mse:.4f} "
          f"(AE best val: {log['best_val_masked_mse']:.4f})")
    print("SMOKE TEST OK" if log["best_val_masked_mse"] < zero_mse
          else "SMOKE TEST WARN: AE did not beat zero baseline")
