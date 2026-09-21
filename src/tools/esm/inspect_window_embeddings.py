#!/usr/bin/env python3
"""
inspect_window_embeddings.py — inspect + compare embedding HDF5 files.

Handles, in ONE run and WITHOUT assuming identical schemas:
  * new  ESM-IF fixed-29 windows : window_if_struct [N,29,512] + pad_mask
    (esm-if_test_{zero,padtoken,eosrepeat}.h5)
  * old  ESM-C legacy 3-region   : peptide_emb/n_flank_emb/c_flank_emb [N,1152]
  * old  ESM-IF legacy 3-region  : peptide_if_struct/... [N,512]
  * (future) single-context 2D   : context_emb / context_if_struct [N,D]

Comparisons across the window modes are row-aligned via row_indices
(verified, else intersected). Old files join the comparison table only —
they have different N / row orders (53072 / 53198 vs 46617).

Usage:
    python src/tools/esm/inspect_window_embeddings.py \
        data/processed/embeddings/esm-if_test_zero.h5 \
        data/processed/embeddings/esm-if_test_padtoken.h5 \
        data/processed/embeddings/esm-if_test_eosrepeat.h5 \
        data/processed/embeddings/esmc_protein_embeddings.h5 \
        data/processed/embeddings/esmif_structure_embeddings.h5 \
        data/processed/df_all.csv \
        --out_dir results/embedding_inspection/windows_test

    python src/tools/esm/inspect_window_embeddings.py <any subset> df_all.csv \
        --out_dir results/embedding_inspection/quick --chunk 2048
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
ap = argparse.ArgumentParser(description="Inspect + compare embedding H5 files")
ap.add_argument("h5_paths", nargs="+", help="Embedding HDF5 files (any schema)")
ap.add_argument("csv_path", help="CSV with label column (e.g. df_all.csv)")
ap.add_argument("--out_dir", default="results/embedding_inspection/windows")
ap.add_argument("--chunk", type=int, default=2048, help="HDF5 read chunk (rows)")
ap.add_argument("--pca_n", type=int, default=10000, help="Rows sampled for PCA")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Schema detection — each file probed independently, nothing assumed
# --------------------------------------------------------------------------
def detect_schema(f):
    keys = set(f.keys())
    if "window_if_struct" in keys:
        return "window29"
    if "peptide_emb" in keys:
        return "legacy_esmc"
    if "peptide_if_struct" in keys:
        return "legacy_esmif"
    if "context_emb" in keys or "context_if_struct" in keys:
        return "context2d"
    raise ValueError(f"Unknown schema, keys={sorted(keys)}")


def region_datasets(kind):
    if kind == "legacy_esmc":
        return {"peptide": "peptide_emb", "n_flank": "n_flank_emb",
                "c_flank": "c_flank_emb"}
    if kind == "legacy_esmif":
        return {"peptide": "peptide_if_struct", "n_flank": "n_flank_if_struct",
                "c_flank": "c_flank_if_struct"}
    if kind == "context2d":
        return None  # resolved per-file
    return None


def decode_bytes(arr, n=5):
    out = []
    for x in arr[:n]:
        out.append(x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x))
    return out


# --------------------------------------------------------------------------
# Load CSV labels once
# --------------------------------------------------------------------------
print("=" * 70)
print("  Window embedding inspection — multi-file")
print("=" * 70)
df = pd.read_csv(args.csv_path, usecols=["label"])
csv_labels = df["label"].values.astype(int)
print(f"\nCSV: {args.csv_path}  rows={len(df)}  "
      f"pos={(csv_labels == 1).sum()} neg={(csv_labels == 0).sum()}")

# --------------------------------------------------------------------------
# Per-file probe (streaming — never loads a full 2.6 GB array)
# --------------------------------------------------------------------------
files = []  # list of dicts with schema + streaming stats
for p in args.h5_paths:
    print("\n" + "=" * 70)
    print(f"  {Path(p).name}")
    print("=" * 70)
    info = {"path": p, "name": Path(p).stem}
    with h5py.File(p, "r") as f:
        kind = detect_schema(f)
        info["kind"] = kind
        print(f"  schema: {kind}")
        print("  attrs:")
        info["attrs"] = {}
        for k, v in f.attrs.items():
            info["attrs"][k] = v
            print(f"    {k:<18}: {v}")
        print("  datasets:")
        for k in sorted(f.keys()):
            d = f[k]
            print(f"    {k:<22}: shape={d.shape} dtype={d.dtype}")
        n_rows = f[list(f.keys())[0]].shape[0]
        info["n_rows"] = int(n_rows)

        # -- small metadata arrays (full read, tiny) --
        for mkey in ["status", "n_pad", "c_pad", "row_indices",
                     "fallback_flag", "start", "end"]:
            if mkey in f:
                info[mkey] = f[mkey][:]
        if "peptide_ids" in f:
            info["pep"] = f["peptide_ids"][:]
        elif "peptide_seqs" in f:
            info["pep"] = f["peptide_seqs"][:]
        if "uniprot_ids" in f:
            info["uid"] = f["uniprot_ids"][:]
        print(f"  id sample pep={decode_bytes(info.get('pep', ['?']))} "
              f"uid={decode_bytes(info.get('uid', ['?']))}")

        if "status" in info:
            u, c = np.unique(info["status"], return_counts=True)
            info["status_counts"] = dict(zip(u.tolist(), c.tolist()))
            print(f"  status: {info['status_counts']} "
                  f"(0=primary 1=af2 2=missing 3=notfound)")

        # -- row_indices validity vs CSV --
        if "row_indices" in info:
            ri = info["row_indices"]
            info["ri_in_range"] = bool(
                (ri >= 0).all() and (ri < len(df)).all())
            info["ri_unique"] = int(len(np.unique(ri)))
            print(f"  row_indices: unique={info['ri_unique']}/{len(ri)} "
                  f"range=[{ri.min()},{ri.max()}] "
                  f"in_csv_range={info['ri_in_range']}")
            if info["ri_in_range"]:
                lab = csv_labels[ri]
                info["labels"] = lab
                print(f"  labels via row_indices: "
                      f"pos={(lab == 1).sum()} neg={(lab == 0).sum()}")
            else:
                print("  WARN: row_indices out of CSV range — "
                      "label stats skipped for this file")
                info["labels"] = None
        else:
            print("  no row_indices — label stats skipped (key join needed)")
            info["labels"] = None

        # -- streaming embedding stats --
        n_chunk = args.chunk
        if kind == "window29":
            W = f["window_if_struct"]
            M = f["pad_mask"] if "pad_mask" in f else None
            n_nan = n_inf = n_zero_win = 0
            frob = []  # per-chunk sample of whole-window Frobenius norms
            slot_sum = np.zeros(29)
            slot_cnt = np.zeros(29)
            pad_norms = []
            real_norms = []
            for s in range(0, n_rows, n_chunk):
                e = s + min(n_chunk, n_rows - s)
                w = W[s:e].astype(np.float64)
                n_nan += int((~np.isfinite(w) & np.isnan(w)).sum()) \
                    if False else int(np.isnan(w).sum())
                n_inf += int(np.isinf(w).sum())
                fn = np.linalg.norm(w.reshape(e - s, -1), axis=1)
                n_zero_win += int((fn == 0).sum())
                frob.extend(fn[:: max(1, (e - s) // 256)].tolist())
                sn = np.linalg.norm(w, axis=2)  # [c,29]
                slot_sum += sn.sum(axis=0)
                slot_cnt += (sn > 0).sum(axis=0)
                if M is not None:
                    m = M[s:e][:]
                    pad_norms.extend(sn[m].tolist())
                    real_norms.extend(sn[~m].tolist())
            info.update({
                "emb_shape": tuple(W.shape), "emb_dim": int(W.shape[2]),
                "n_nan": n_nan, "n_inf": n_inf,
                "n_zero_windows": n_zero_win,
                "frob_median": float(np.median(frob)),
                "frob_mean": float(np.mean(frob)),
                "slot_mean_norm": slot_sum / np.maximum(n_rows, 1),
                "slot_coverage": slot_cnt / n_rows,
                "pad_slot_median": float(np.median(pad_norms))
                if pad_norms else None,
                "real_slot_median": float(np.median(real_norms))
                if real_norms else None,
            })
            print(f"  windows: NaN={n_nan} Inf={n_inf} "
                  f"zero_windows={n_zero_win}/{n_rows}")
            print(f"  whole-window Frobenius: median={info['frob_median']:.3f} "
                  f"mean={info['frob_mean']:.3f}")
            print(f"  real slots median L2={info['real_slot_median']:.3f}  "
                  f"padded slots median L2={info['pad_slot_median']:.3f}")
            print(f"  slot coverage (frac nonzero): "
                  f"{np.round(info['slot_coverage'], 3).tolist()}")
        else:
            regs = region_datasets(kind)
            if regs is None:  # context2d — resolve actual key
                regs = {k: k for k in
                        ("context_emb", "context_if_struct") if k in f}
            info["regions"] = {}
            for rname, dset in regs.items():
                if dset not in f:
                    continue
                d = f[dset]
                z = t = 0
                norms = []
                nan = inf = 0
                for s in range(0, n_rows, n_chunk):
                    e = s + min(n_chunk, n_rows - s)
                    w = d[s:e].astype(np.float64)
                    nan += int(np.isnan(w).sum())
                    inf += int(np.isinf(w).sum())
                    n = np.linalg.norm(w, axis=1)
                    z += int((n == 0).sum())
                    t += e - s
                    norms.extend(n[:: max(1, (e - s) // 256)].tolist())
                info["regions"][rname] = {
                    "shape": tuple(d.shape), "zero": z, "total": t,
                    "nan": nan, "inf": inf,
                    "median": float(np.median(norms)),
                    "mean": float(np.mean(norms)),
                }
                r = info["regions"][rname]
                print(f"  {rname}: zero={z}/{t} "
                      f"({100*z/max(t,1):.1f}%) NaN={nan} Inf={inf} "
                      f"norm med={r['median']:.3f} mean={r['mean']:.3f}")
    files.append(info)

# --------------------------------------------------------------------------
# Cross-window comparison (only kind==window29, chunk-aligned)
# --------------------------------------------------------------------------
wins = [fi for fi in files if fi["kind"] == "window29"]
if len(wins) >= 2:
    print("\n" + "=" * 70)
    print("  CROSS-MODE comparison (row-aligned via row_indices)")
    print("=" * 70)
    base_ri = wins[0].get("row_indices")
    aligned = all(
        w.get("row_indices") is not None and len(w["row_indices"]) == len(base_ri)
        and (w["row_indices"] == base_ri).all() for w in wins)
    if not aligned:
        # intersect on row_indices -> index maps per file
        print("  row orders differ — intersecting on row_indices...")
        key_to_pos = [{int(r): i for i, r in enumerate(w["row_indices"])}
                      for w in wins]
        common = sorted(set.intersection(
            *[set(m.keys()) for m in key_to_pos]))
        print(f"  common rows: {len(common)}/{min(len(w['row_indices']) for w in wins)}")
        maps = [np.array([m[k] for k in common], dtype=np.int64)
                for m in key_to_pos]
    else:
        print(f"  identical row order across {len(wins)} files "
              f"(N={len(base_ri)}) — direct chunk alignment")
        common, maps = None, None

    handles = [h5py.File(w["path"], "r") for w in wins]
    try:
        names = [w["name"] for w in wins]
        # status arrays (tiny) — pad-constancy checks use status==0 rows only,
        # since missing/notfound rows are all-zero windows in every mode
        statuses = []
        for h in handles:
            if "status" in h:
                st = h["status"][:]
                statuses.append(st)
            else:
                statuses.append(None)
        n = len(common) if common is not None else wins[0]["n_rows"]
        diff_acc = {(names[i], names[j]): []
                    for i in range(len(wins)) for j in range(i + 1, len(wins))}
        cos_acc = {k: [] for k in diff_acc}
        max_real_diff = 0.0
        masks_equal = True
        pad_zero_max = 0.0       # zero-mode padded slots should be ~0
        padtoken_std_max = 0.0   # padtoken padded slots should be constant
        n_cmp = args.chunk
        for s in range(0, n, n_cmp):
            e = s + min(n_cmp, n - s)
            if maps is not None:
                idx = [m[s:e] for m in maps]
                arrs = [handles[i]["window_if_struct"][idx[i]][:].astype(np.float64)
                        for i in range(len(wins))]
                mk = [handles[i]["pad_mask"][idx[i]][:]
                      for i in range(len(wins))]
            else:
                arrs = [h["window_if_struct"][s:e].astype(np.float64)
                        for h in handles]
                mk = [h["pad_mask"][s:e][:] for h in handles]
            if not all((mk[0] == m).all() for m in mk[1:]):
                masks_equal = False
            mask = mk[0]
            real = ~mask
            # real-slot identity: max abs diff on real positions
            for i in range(len(arrs)):
                for j in range(i + 1, len(arrs)):
                    d = np.abs(arrs[i] - arrs[j])
                    rd = d[real].max(initial=0.0)
                    max_real_diff = max(max_real_diff, float(rd))
                    fd = np.linalg.norm(
                        (arrs[i] - arrs[j]).reshape(e - s, -1), axis=1)
                    diff_acc[(names[i], names[j])].extend(fd.tolist())
                    # cosine of mean-pooled REAL slots
                    a = (arrs[i] * real[..., None]).sum(1) / \
                        np.maximum(real.sum(1, keepdims=True), 1)
                    b = (arrs[j] * real[..., None]).sum(1) / \
                        np.maximum(real.sum(1, keepdims=True), 1)
                    denom = (np.linalg.norm(a, axis=1) *
                             np.linalg.norm(b, axis=1))
                    ok = denom > 0
                    cos_acc[(names[i], names[j])].extend(
                        ((a[ok] * b[ok]).sum(1) / denom[ok]).tolist())
            # zero-mode check: which file is the zero mode?
            chunk_ok = None
            if any(s is not None for s in statuses):
                chunk_ok = np.ones(e - s, dtype=bool)
                for si, st in enumerate(statuses):
                    if st is None:
                        continue
                    if maps is not None:
                        chunk_ok &= st[maps[si][s:e]] == 0
                    else:
                        chunk_ok &= st[s:e] == 0
            for wi, (w, a) in enumerate(zip(wins, arrs)):
                use = mask if chunk_ok is None else (mask & chunk_ok[:, None])
                if w["attrs"].get("pad_mode") == "zero" and use.any():
                    pad_zero_max = max(
                        pad_zero_max, float(np.abs(a[use]).max(initial=0.0)))
                if w["attrs"].get("pad_mode") == "padtoken" and use.any():
                    # padded slots within/across rows should all equal pad_rep:
                    # per-dim std across all padded slots should be ~0
                    prow = a[use].reshape(-1, a.shape[-1])
                    padtoken_std_max = max(
                        padtoken_std_max, float(prow.std(axis=0).max()))
        print(f"  pad_mask identical across modes: {masks_equal}")
        print(f"  max abs diff on REAL slots: {max_real_diff:.6f} "
              f"(~0 ⇒ modes differ ONLY in padding ⇒ correct)")
        if any(w["attrs"].get("pad_mode") == "zero" for w in wins):
            print(f"  zero-mode padded slots max|.| = {pad_zero_max:.6f} "
                  f"(expect 0.0)")
        if any(w["attrs"].get("pad_mode") == "padtoken" for w in wins):
            print(f"  padtoken padded-slot per-dim std = "
                  f"{padtoken_std_max:.6f} (status==0 rows; ~0 ⇒ "
                  f"single pad vector)")
        pair_rows = []
        for k, v in diff_acc.items():
            v = np.array(v)
            c = np.array(cos_acc[k])
            print(f"  {' vs '.join(k)}: Frobenius med={np.median(v):.3f} "
                  f"mean={np.mean(v):.3f} max={v.max():.3f} | "
                  f"real-mean-cos med={np.median(c):.4f} "
                  f"min={c.min():.4f}")
            pair_rows.append({"pair": " vs ".join(k),
                              "frob_median": float(np.median(v)),
                              "frob_mean": float(np.mean(v)),
                              "frob_max": float(v.max()),
                              "real_cos_median": float(np.median(c)),
                              "real_cos_min": float(c.min())})
        pd.DataFrame(pair_rows).to_csv(
            out_dir / "pairwise_mode_diffs.csv", index=False)
        # keep one pooled sample for the diff histogram
        info_hist = {k: np.array(v)[:20000] for k, v in diff_acc.items()}
    finally:
        for h in handles:
            h.close()
else:
    print("\n<2 window files — cross-mode comparison skipped>")
    info_hist = {}

# --------------------------------------------------------------------------
# Plots (lightweight: no t-SNE, no per-dim tests)
# --------------------------------------------------------------------------
# 1. slot norm curves (window files overlaid)
if wins:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for w in wins:
        ax.plot(range(29), w["slot_mean_norm"], "o-", ms=3,
                label=w["name"])
    ax.axvspan(-0.5, 9.5, alpha=0.08, label="N-flank slots")
    ax.axvspan(9.5, 18.5, alpha=0.08, color="green")
    ax.axvspan(18.5, 28.5, alpha=0.08, color="orange")
    ax.set_xlabel("Window slot (0-9 N | 10-18 peptide | 19-28 C)")
    ax.set_ylabel("Mean slot L2 norm")
    ax.set_title("Per-slot embedding magnitude by padding mode")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "slot_norm_curve.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out_dir / 'slot_norm_curve.png'}")

# 2. pairwise diff histograms
if info_hist:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, v in info_hist.items():
        ax.hist(v, bins=80, alpha=0.5, density=True, label=" vs ".join(k))
    ax.set_xlabel("Per-row Frobenius norm of mode difference")
    ax.set_ylabel("Density")
    ax.set_title("How much do padding modes change each window?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pairwise_diff_hist.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out_dir / 'pairwise_diff_hist.png'}")

# 3. status breakdown bar
if any("status_counts" in w for w in files):
    labels = ["0 primary", "1 af2", "2 missing", "3 notfound"]
    x = np.arange(len(files))
    fig, ax = plt.subplots(figsize=(max(8, len(files) * 1.8), 4.5))
    bottom = np.zeros(len(files))
    for si in [0, 1, 2, 3]:
        vals = np.array([f.get("status_counts", {}).get(si, 0)
                         for f in files], dtype=float)
        ax.bar(x, vals, bottom=bottom, label=labels[si])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f["name"] for f in files], rotation=20, ha="right",
                       fontsize=8)
    ax.set_ylabel("Rows")
    ax.set_title("Structure coverage per file (status)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "status_breakdown.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out_dir / 'status_breakdown.png'}")

# 4. PCA-lite on mean-pooled real slots (first window file only, sampled)
if wins:
    print("\n  PCA-lite on mean-pooled real slots...")
    with h5py.File(wins[0]["path"], "r") as h:
        Wd, Md = h["window_if_struct"], h["pad_mask"]
        lab = wins[0].get("labels")
        n_all = wins[0]["n_rows"]
        take = rng.choice(n_all, size=min(args.pca_n, n_all), replace=False)
        take.sort()
        # read in chunks to bound RAM
        pooled = []
        for s in range(0, len(take), args.chunk):
            idx = take[s:s + args.chunk]
            w = Wd[idx].astype(np.float64)
            m = Md[idx][:]
            real = ~m
            pooled.append((w * real[..., None]).sum(1) /
                          np.maximum(real.sum(1, keepdims=True), 1))
        pooled = np.vstack(pooled)
        lab_s = lab[take] if lab is not None else None
    keep = np.linalg.norm(pooled, axis=1) > 0
    pca = PCA(n_components=min(20, pooled.shape[1], keep.sum()))
    pc = pca.fit_transform(pooled[keep])
    print(f"  PCA: PC1={pca.explained_variance_ratio_[0]*100:.1f}% "
          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}% "
          f"top5={pca.explained_variance_ratio_[:5].sum()*100:.1f}%")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1),
                 np.cumsum(pca.explained_variance_ratio_) * 100, "o-", ms=3)
    axes[0].set_xlabel("# PCs")
    axes[0].set_ylabel("Cumulative variance (%)")
    axes[0].set_title(f"PCA variance ({wins[0]['name']})")
    if lab_s is not None:
        ls = lab_s[keep]
        for v, c, n_ in [(0, "salmon", "Neg"), (1, "steelblue", "Pos")]:
            m = ls == v
            axes[1].scatter(pc[m, 0], pc[m, 1], c=c, alpha=0.15, s=5,
                            label=n_, rasterized=True)
        axes[1].legend(fontsize=8, markerscale=3)
    else:
        axes[1].scatter(pc[:, 0], pc[:, 1], alpha=0.15, s=5, rasterized=True)
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[1].set_title("PC1 vs PC2 (mean-pooled real slots)")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_windows.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out_dir / 'pca_windows.png'}")

# --------------------------------------------------------------------------
# Comparison table CSV + terminal summary
# --------------------------------------------------------------------------
rows = []
for w in files:
    r = {"file": w["name"], "schema": w["kind"], "n_rows": w["n_rows"]}
    if w["kind"] == "window29":
        r.update({
            "dim": w["emb_dim"], "zero_windows": w["n_zero_windows"],
            "zero_pct": 100 * w["n_zero_windows"] / w["n_rows"],
            "nan": w["n_nan"], "inf": w["n_inf"],
            "frob_median": round(w["frob_median"], 3),
            "real_slot_med": round(w["real_slot_median"], 3)
            if w["real_slot_median"] is not None else None,
            "pad_slot_med": round(w["pad_slot_median"], 3)
            if w["pad_slot_median"] is not None else None,
        })
    else:
        tot_z = sum(v["zero"] for v in w.get("regions", {}).values())
        tot_t = sum(v["total"] for v in w.get("regions", {}).values())
        r.update({
            "dim": ";".join(f"{k}:{v['shape'][1]}"
                            for k, v in w.get("regions", {}).items()),
            "zero_windows": tot_z,
            "zero_pct": 100 * tot_z / max(tot_t, 1),
            "nan": sum(v["nan"] for v in w.get("regions", {}).values()),
            "inf": sum(v["inf"] for v in w.get("regions", {}).values()),
            "frob_median": None, "real_slot_med": None,
            "pad_slot_med": None,
        })
    for si in [0, 1, 2, 3]:
        r[f"status_{si}"] = w.get("status_counts", {}).get(si, None)
    rows.append(r)
cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(out_dir / "comparison_table.csv", index=False)
print("\n" + "=" * 70)
print("  COMPARISON TABLE")
print("=" * 70)
print(cmp_df.to_string(index=False))
print(f"\nSaved to {out_dir}/ : "
      f"{sorted(p.name for p in out_dir.glob('*'))}")
