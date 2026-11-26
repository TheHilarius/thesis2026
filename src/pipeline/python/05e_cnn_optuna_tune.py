"""
05e_cnn_optuna_tune.py
Optuna TPE tuner for the thesis CNN — fully nested: one Optuna study per
OUTER bucket, each trial = the 05a nested inner loop (5 inner folds = the
other buckets; train on 4, val on 1). Mirrors 46_xgb_optuna_tune.py house
conventions; inner-val mean AUC-ROC is the sole selection metric, logloss
rides along as a user_attr. Near-duplicate of 06e_ffnn_optuna_tune.py by
house convention (05a/06a are also near-duplicates); only the search
space, parametric builder and anchors differ.

Conventions inherited from 46 (locked 2026-11-26 plan):
  - Selection metric: mean inner-val AUC-ROC over the 5 inner folds of one
    outer bucket (direction maximize). Same folds as 05a nested CV — the
    H5 fold column is homology-aware by construction, so inner folds need
    no extra carving.
  - logloss_mean + per-fold AUC/logloss as user_attrs (never selects).
  - TPESampler(multivariate=True, seed=--seed); known-good configs seeded
    via enqueue_trial (anchors: k17 1-layer c32 = manual kernel-sweep
    winner; 2-layer k17/k17 = depth-sweep reference).
  - SQLite + race-retry open_study (copy of 46's, proven on hub1).
  - Per-arm isolation: runs/cnn_optuna_<features>/, study names
    cnn_<features>_outer<k>; one study.db holds all 6 studies.
  - --smoke wiring mode (outer 0, 2 inner folds, 2 trials, 3 epochs,
    separate smoke study).
  - NO early stopping inside trials — fixed epoch budget per trial, epochs
    SEARCHED ({20,30,40,60}; house canonical 30 included as anchor).
    Rationale mirrors 21_xgb_es_diagnostics; the tuner itself is the
    anti-overfit device (overfit regimes score badly on inner-val).
  - Pruner: MedianPruner on the inner-fold step, reporting the running
    mean inner AUC (same metric, partial — never a proxy). Lenient:
    n_warmup_steps=1 (fold-0 AUC is noisy), startup trials guarded.
  - Polish phase: DROPPED by design — lr and epochs are IN the space here
    (46 needed low-LR polish only because they were outside its space).

Search space (CNN; kernel/channel ranges narrowed by the manual sweeps —
kernel curve flat with k17 best, channels flat with overfit at 64+):
  lr              float  1e-4..3e-3 (log)
  epochs          cat    {20, 30, 40, 60}
  weight_decay    float  1e-6..1e-2 (log)
  dropout         float  0.1..0.5
  n_conv_layers   int    1..3          (conditional channels_i/kernel_i)
  channels_i      cat    {8, 16, 32}   (per conv layer i)
  kernel_i        cat    {11, 15, 17}  (per conv layer i)
  dense_units     cat    {32, 64, 128}
  batch_size      cat    {32, 64}
  Fixed: AdamW, ReLU, same padding, naive GAP, BCEWithLogitsLoss.

Usage (repo root; one worker tunes ONE outer bucket):
  python src/pipeline/python/05e_cnn_optuna_tune.py --outer 0 --trials 25
  python src/pipeline/python/05e_cnn_optuna_tune.py --outer 1 --seed 143
  python src/pipeline/python/05e_cnn_optuna_tune.py --smoke
  python src/pipeline/python/05e_cnn_optuna_tune.py --refit           # after
      all 6 studies: per-outer winner refit (multi-seed) on full outer-train,
      score outer bucket -> nested estimate + winner_stability.csv
"""

import sys
import os
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

from config import (N_CV_FOLDS, PROJECT_ROOT, RANDOM_STATE,
                    get_feature_set_config, get_feature_component,
                    get_embedding_source, validate_feature_set, validate_config)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("cnn05a", str(Path(SRC_DIR) / "05a_cnn_modelling.py"))
_05a = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_05a)
WindowDataset, predict_probs = _05a.WindowDataset, _05a.predict_probs

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:
    print("FATAL: torch not installed. Activate env_esmc_server / GPU env.")
    sys.exit(1)

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.study import MaxTrialsCallback
    from optuna.trial import TrialState
    from sqlalchemy.exc import OperationalError as SQLOperationalError
except ImportError:
    sys.exit("optuna not installed. On the server (inside env_esmc_server): "
              "pip install optuna")


# ── parametric CNN (tuner twin of the 05c zoo; anchors map to zoo keys) ──
class ParametricCNN(nn.Module):
    """n x [Conv1d(k_i, same pad) -> ReLU -> Dropout] -> naive GAP ->
    Linear(C_n, dense) -> ReLU -> Dropout -> Linear(dense, 1).

    Superset of 05c SimpleCNN / TwoLayerCNN: depth, per-layer channels and
    kernels, dense head width and dropout all caller-set so Optuna can
    search them. Same (N, D, L) input convention and naive mean-over-slots
    GAP as 05a (no pad mask, masked-vs-naive showed identical performance).
    """

    def __init__(self, in_dim, n_layers, channels, kernels, dense_units, dropout):
        super().__init__()
        convs, prev = [], in_dim
        for c, k in zip(channels, kernels):
            convs += [nn.Conv1d(prev, c, k, padding=k // 2),
                      nn.ReLU(), nn.Dropout(dropout)]
            prev = c
        self.convs = nn.Sequential(*convs)
        self.head = nn.Sequential(
            nn.Linear(prev, dense_units), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dense_units, 1))

    def forward(self, x):  # x: (N, D, L)
        h = self.convs(x)
        h = h.mean(-1)          # naive GAP over all slots
        return self.head(h).squeeze(-1)


def build_trial_cnn(in_dim, p):
    channels = [p[f"channels_{i}"] for i in range(1, p["n_conv_layers"] + 1)]
    kernels = [p[f"kernel_{i}"] for i in range(1, p["n_conv_layers"] + 1)]
    return ParametricCNN(in_dim, p["n_conv_layers"], channels, kernels,
                         p["dense_units"], p["dropout"])


def suggest(trial):
    p = {
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "epochs": trial.suggest_categorical("epochs", [20, 30, 40, 60]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "n_conv_layers": trial.suggest_int("n_conv_layers", 1, 3),
        "dense_units": trial.suggest_categorical("dense_units", [32, 64, 128]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
    }
    for i in range(1, p["n_conv_layers"] + 1):   # conditional per-layer shape
        p[f"channels_{i}"] = trial.suggest_categorical(f"channels_{i}", [8, 16, 32])
        p[f"kernel_{i}"] = trial.suggest_categorical(f"kernel_{i}", [11, 15, 17])
    return p


# known-good configs seeded into every fresh study (trial 1-2): manual
# kernel-sweep winner (k17, 1 layer, c32 = cnn_simple_k17 twin) and the
# 2-layer depth reference (k17/k17, TwoLayerCNN twin).
ANCHORS = [
    {"lr": 1e-3, "epochs": 30, "weight_decay": 1e-6, "dropout": 0.1,
     "n_conv_layers": 1, "dense_units": 32, "batch_size": 64,
     "channels_1": 32, "kernel_1": 17},               # cnn_simple_k17 twin
    {"lr": 1e-3, "epochs": 30, "weight_decay": 1e-6, "dropout": 0.1,
     "n_conv_layers": 2, "dense_units": 32, "batch_size": 64,
     "channels_1": 32, "kernel_1": 17,
     "channels_2": 32, "kernel_2": 17},                # cnn_2layer_k17_k17 twin
]


def train_fixed_budget(model, train_loader, val_loader, device, p, seed):
    """05a train_one_fold WITHOUT early stopping: exactly p['epochs'] epochs
    (budget-atomic, mirrors 46's no-ES convention). Best-val state kept,
    house style. Returns model, history, best val AUC."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"],
                            weight_decay=p["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc, best_state = -1.0, None
    hist = {"train_loss": [], "val_auc": []}
    model.to(device)
    for _ in range(p["epochs"]):
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
        ys, ps = predict_probs(model, val_loader, device)
        auc = roc_auc_score(ys, ps)
        hist["val_auc"].append(auc)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, hist, best_auc


def load_data(features):
    """Same prepared-H5 loading as 05a main (THESIS_VARIANT respected)."""
    feat_cfg = get_feature_set_config(features)
    validate_feature_set(feat_cfg)
    emb_keys = [c for c in feat_cfg["components"]
                if get_feature_component(c)["type"] == "embedding_windows"]
    assert len(emb_keys) == 1, \
        f"CNN tuner needs 1 embedding_windows component, got {feat_cfg['components']}"
    comp = get_feature_component(emb_keys[0])
    src = get_embedding_source(comp["embedding_key"])
    h5_path = str(src["prepared_path"])
    if not Path(h5_path).exists():
        sys.exit(f"FATAL: prepared H5 missing: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        ds_name = _05a.resolve_window_ds(f, src["window_ds"])
        labels, folds = f["labels"][:], f["folds"][:]
        emb_dim = int(f.attrs.get("emb_dim", 0)) or int(f[ds_name].shape[2])
        window = int(f.attrs.get("window", 29))
    print(f"H5: {h5_path}\n  ds={ds_name}  N={len(labels)} D={emb_dim} "
          f"window={window}  folds={sorted(map(int, np.unique(folds)))}")
    return h5_path, ds_name, labels, folds, emb_dim, window


def eval_params(p, h5_path, ds_name, labels, folds, outer, inner_folds,
                emb_dim, device, trial=None, smoke=False):
    """Nested inner loop for ONE trial: 5 (or 2 in smoke) inner folds.
    Returns result dict; reports running-mean AUC to the pruner per fold."""
    aucs, lls = [], []
    for pos, inner in enumerate(inner_folds):
        seed = p["base_seed"] + outer * 10 + pos        # mirrors 05a perfold
        tr_idx = np.where((folds != outer) & (folds != inner))[0]
        va_idx = np.where(folds == inner)[0]
        tl = DataLoader(WindowDataset(h5_path, ds_name, tr_idx, labels[tr_idx]),
                        batch_size=p["batch_size"], shuffle=True)
        vl = DataLoader(WindowDataset(h5_path, ds_name, va_idx, labels[va_idx]),
                        batch_size=p["batch_size"])
        net = build_trial_cnn(emb_dim, p)
        net, _, auc = train_fixed_budget(net, tl, vl, device, p, seed)
        y_va = labels[va_idx].astype(int)
        _, probs = predict_probs(net, vl, device)
        del net
        aucs.append(auc)
        lls.append(log_loss(y_va, probs, labels=[0, 1]))
        if trial is not None:
            trial.report(float(np.mean(aucs)), step=pos)  # running mean, same metric
            if not smoke and trial.should_prune():
                raise optuna.TrialPruned()
    return {"auc_mean": float(np.mean(aucs)),
            "logloss_mean": float(np.mean(lls)),
            "auc_folds": [float(a) for a in aucs],
            "logloss_folds": [float(l) for l in lls]}


def arm_paths(features):
    """Per-arm output dir + study prefix (every arm gets its own dir)."""
    return (PROJECT_ROOT / "runs" / f"cnn_optuna_{features}",
            f"cnn_{features}")


def open_study(run_dir, study_name, seed):
    """Open (or create) the study, retrying on creation races (46 pattern)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{run_dir / 'study.db'}"
    last = None
    for attempt in range(5):
        try:
            return optuna.create_study(
                study_name=study_name, storage=storage, load_if_exists=True,
                direction="maximize",
                sampler=TPESampler(seed=seed, multivariate=True),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=8,
                                                    n_warmup_steps=1))
        except (SQLOperationalError, optuna.exceptions.StorageInternalError,
                RuntimeError) as e:
            last = e
            print(f"  [study race/lock {attempt + 1}/5] {e}; retry in 30s")
            time.sleep(30)
    raise last


def run_tune(args):
    run_dir, prefix = arm_paths(args.features)
    outer = args.outer
    study_name = f"{prefix}_outer{outer}"
    h5_path, ds_name, labels, folds, emb_dim, window = load_data(args.features)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  outer={outer}  study={study_name}")

    if args.smoke:
        study_name = f"cnn_smoke_outer{outer}"      # separate smoke study
        study = open_study(run_dir, study_name, args.seed)
        smoke_epochs = 3
    else:
        study = open_study(run_dir, study_name, args.seed)

    buckets = list(range(N_CV_FOLDS + 1))
    inner_folds = [b for b in buckets if b != outer]
    if args.smoke:
        inner_folds = inner_folds[:2]

    if len(study.trials) == 0 and not args.smoke:      # fresh study: anchors first
        for anchor in ANCHORS:
            study.enqueue_trial(anchor)
        print(f"  enqueued {len(ANCHORS)} anchors "
              "(cnn_simple_k17 / cnn_2layer_k17_k17 twins)")

    def objective(trial):
        p = suggest(trial)
        if args.smoke:
            p["epochs"] = smoke_epochs
        p["base_seed"] = args.seed
        res = eval_params(p, h5_path, ds_name, labels, folds, outer,
                          inner_folds, emb_dim, device,
                          trial=trial, smoke=args.smoke)
        trial.set_user_attr("logloss_mean", res["logloss_mean"])
        trial.set_user_attr("auc_folds", res["auc_folds"])
        trial.set_user_attr("logloss_folds", res["logloss_folds"])
        print(f"  [trial {trial.number}] AUC={res['auc_mean']:.4f} "
              f"logloss={res['logloss_mean']:.4f}")
        return res["auc_mean"]

    n_trials = args.trials or (2 if args.smoke else 25)
    callbacks = [MaxTrialsCallback(args.max_trials,
                                   states=(TrialState.COMPLETE,))] \
        if args.max_trials else []
    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)

    tag = "smoke" if args.smoke else "tuned"
    df_t = study.trials_dataframe()
    df_t.to_csv(run_dir / f"trials_outer{outer}_{tag}.csv", index=False)
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    if done:                                  # best_trial RAISES on empty study
        out = run_dir / f"best_params_outer{outer}_{tag}.json"
        best = study.best_trial
        payload = {"params": {k: v for k, v in best.params.items()},
                   "auc_mean": best.value,
                   "logloss_mean": best.user_attrs.get("logloss_mean"),
                   "study_name": study_name, "features": args.features,
                   "outer": outer, "mode": tag}
        out.write_text(json.dumps(payload, indent=2))
        print(f"Best {tag} AUC={best.value:.4f}\n  params={best.params}")
        print(f"Best params -> {out}")
    print(f"Trials table -> {run_dir / f'trials_outer{outer}_{tag}.csv'}")


def run_refit(args):
    """After all 6 outer studies: per-outer winner refit on FULL outer-train
    (5 buckets), score outer bucket with --refit-seeds seeds -> honest nested
    estimate. Writes winner_stability.csv + nested_summary.json + overall
    best_params.json."""
    run_dir, prefix = arm_paths(args.features)
    h5_path, ds_name, labels, folds, emb_dim, window = load_data(args.features)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    buckets = list(range(N_CV_FOLDS + 1))

    rows, inner_aucs, outer_aucs = [], {}, []
    for outer in buckets:
        study_name = f"{prefix}_outer{outer}"
        try:
            study = optuna.load_study(study_name=study_name,
                                      storage=f"sqlite:///{run_dir / 'study.db'}")
        except KeyError:
            print(f"  WARNING: study {study_name} missing — skipped")
            continue
        done = [t for t in study.trials if t.state.name == "COMPLETE"]
        if not done:
            print(f"  WARNING: study {study_name} has no COMPLETE trials — skipped")
            continue
        best = study.best_trial
        inner_aucs[outer] = best.value
        p = {**best.params, "base_seed": args.seed}
        va_idx = np.where(folds == outer)[0]
        tr_idx = np.where(folds != outer)[0]
        vl = DataLoader(WindowDataset(h5_path, ds_name, va_idx, labels[va_idx]),
                        batch_size=p["batch_size"])
        y_ou = labels[va_idx].astype(int)
        seed_aucs = []
        for s in range(args.refit_seeds):
            p_s = {**p, "base_seed": args.seed + s}
            tl = DataLoader(WindowDataset(h5_path, ds_name, tr_idx, labels[tr_idx]),
                            batch_size=p["batch_size"], shuffle=True)
            net = build_trial_cnn(emb_dim, p_s)
            net, _, _ = train_fixed_budget(net, tl, vl, device, p_s,
                                           seed=args.seed + s)
            _, probs = predict_probs(net, vl, device)
            del net
            seed_aucs.append(roc_auc_score(y_ou, probs))
        outer_aucs.append(float(np.mean(seed_aucs)))
        rows.append({"outer": outer, "n_outer_test": len(y_ou),
                     "inner_auc": round(best.value, 4),
                     "outer_auc_mean": round(float(np.mean(seed_aucs)), 4),
                     "outer_auc_std": round(float(np.std(seed_aucs)), 4),
                     "outer_auc_seeds": [round(a, 4) for a in seed_aucs],
                     "best_params": json.dumps(best.params)})
        print(f"  outer {outer}: inner={best.value:.4f} "
              f"outer={np.mean(seed_aucs):.4f}±{np.std(seed_aucs):.4f} "
              f"({args.refit_seeds} seeds)")

    assert rows, "no completed outer studies found — run the campaign first"
    import csv as _csv
    with open(run_dir / "winner_stability.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    overall = max(inner_aucs, key=inner_aucs.get)
    overall_study = f"{prefix}_outer{overall}"
    study = optuna.load_study(study_name=overall_study,
                              storage=f"sqlite:///{run_dir / 'study.db'}")
    (run_dir / "best_params.json").write_text(json.dumps({
        "params": study.best_params, "inner_auc": study.best_value,
        "study_name": overall_study, "features": args.features,
        "source": "nested tune winner (highest mean inner AUC)",
        "note": "feed to 05a run / 05d plots for the standard report"}, indent=2))
    nested = {
        "features": args.features, "n_outer": len(rows),
        "outer_auc_mean": float(np.mean(outer_aucs)),
        "outer_auc_std": float(np.std(outer_aucs)),
        "outer_auc_values": [float(a) for a in outer_aucs],
        "refit_seeds": args.refit_seeds, "seed_base": args.seed,
        "per_outer": rows,
    }
    (run_dir / "nested_summary.json").write_text(json.dumps(nested, indent=2))
    print(f"\nNESTED estimate: outer AUC {np.mean(outer_aucs):.4f} "
          f"± {np.std(outer_aucs):.4f} over {len(rows)} outer buckets")
    print(f"  -> {run_dir / 'winner_stability.csv'}")
    print(f"  -> {run_dir / 'nested_summary.json'}")
    print(f"  -> {run_dir / 'best_params.json'} (overall winner, outer {overall})")


def parse_args():
    p = argparse.ArgumentParser(description="05e_cnn_optuna_tune: nested "
                                            "Optuna TPE tuning for the CNN")
    p.add_argument("--outer", type=int, default=0,
                   help="outer bucket this worker tunes (0..5; slurm array id)")
    p.add_argument("--trials", type=int, default=None,
                   help="trials this worker contributes (default 25, or 2 "
                        "for --smoke)")
    p.add_argument("--features", type=str, default="emb_esmc300m_win_padtoken",
                   help="Feature-set key from FEATURE_SETS; each arm gets "
                        "its own runs/cnn_optuna_<features>/ dir + studies")
    p.add_argument("--seed", type=int, default=RANDOM_STATE,
                   help="TPE sampler + torch seed; give each parallel worker "
                        "a different seed so startup trials don't duplicate")
    p.add_argument("--max-trials", type=int, default=None,
                   help="cap on COMPLETE trials in this outer study across "
                        "all workers (MaxTrialsCallback)")
    p.add_argument("--smoke", action="store_true",
                   help="wiring check: outer 0, 2 inner folds, 2 trials, "
                        "3 epochs, separate smoke study")
    p.add_argument("--refit", action="store_true",
                   help="post-campaign: per-outer winner refit (multi-seed) "
                        "+ winner_stability.csv + nested_summary.json")
    p.add_argument("--refit-seeds", type=int, default=3,
                   help="seeds per outer refit (default 3)")
    return p.parse_args()


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args = parse_args()
    validate_config()
    if args.refit:
        run_refit(args)
    else:
        run_tune(args)
