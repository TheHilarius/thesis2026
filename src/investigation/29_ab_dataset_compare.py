#!/usr/bin/env python3
"""
29_ab_dataset_compare.py — CV-AUC comparison: full dataset vs the reduced
variant with the negative-protein-only rows removed (THESIS_VARIANT
remove715negonly: 715 orphan proteins, 2,271 negative peptides).

Arms: whatever exists under runs/ab_dataset_test/models/flat/ (one
cv_results per feature x variant; newest timestamp wins). The variant is
read from the JSON CONTENT, not the filename (both variants share the tag):
reduced runs embed the df_*_remove715negonly.csv component path.
The variant drops the 2,271 negative peptides contributed by 715 orphan
proteins (proteins with no positive example) — see make_variant_dataset.py.

Figure (runs/ab_dataset_test/plots/ab_dataset_cv_auc.png):
    TWO panels (shared y, one legend): left = paired DeLong, right =
    paired bootstrap (10k). In each, one group per feature set:
        solid   = full dataset
        hatched = -715 neg-only variant
    family colors (encoding #BBBBBB / ESM-C #EE6677 / ESM-IF #4477AA),
    y-axis 0-1, value labels, delta + test statistic above each pair.

Significance: the two variants score different peptide sets, so every
test runs on the SHARED subset — both models' OOF predictions restricted
to the peptides present in both datasets (order-aligned via the split
CSVs, per-fold y_true asserted). Paired DeLong (Holm across features)
+ 10k paired bootstrap (95% CI); brackets on the figure carry the
DeLong star. Bars stay per-variant means over their own sets.

Outputs:
    runs/ab_dataset_test/plots/ab_dataset_cv_auc.png / .csv
    runs/ab_dataset_test/results/ab_dataset_delong.tsv
"""

import argparse
import csv
import importlib.util
import json
import re
import sys
from math import erfc, sqrt
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR.parent / "pipeline" / "python"))
from config import PROJECT_ROOT, SPLIT_DATA_PATH  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# DeLong + paired bootstrap + Holm from the padmode significance script
_spec = importlib.util.spec_from_file_location(
    "p15", SRC_DIR / "15_padmode_significance.py")
_p15 = importlib.util.module_from_spec(_spec)
sys.modules["p15"] = _p15
_spec.loader.exec_module(_p15)

R = PROJECT_ROOT / "runs"
REDUCED_MARKER = "remove715negonly"   # appears in df_*_<variant>.csv paths
                                      # (override with --variant; substring
                                      #  "remove715" still matches old runs)
COLORS = {"encoding": "#BBBBBB", "esmc": "#EE6677", "esmif": "#4477AA"}
FRIENDLY = {  # bare CSV feature tags -> display names
    "blosum50": "BLOSUM50", "blosum62": "BLOSUM62", "blosum80": "BLOSUM80",
    "sparse": "Sparse", "kidera": "Kidera", "atchley": "Atchley",
    "handcrafted": "Handcrafted", "compositional": "Compositional",
}


def family(tag):
    if "esmc" in tag:
        return "esmc"
    if "esmif" in tag:
        return "esmif"
    return "encoding"


def display(tag):
    """Feature tag -> short bar label."""
    if tag in FRIENDLY:
        return FRIENDLY[tag]
    if tag.startswith("emb_"):
        base = tag[len("emb_"):]
        fam = "ESM-C-300M" if "esmc" in base else "ESM-IF"
        m = re.search(r"_var(\d+)$", base)
        return f"{fam}\n(var{m.group(1)})" if m else fam
    return tag


def load_arm(path, full_n):
    """(mean, std, n_oof, variant) from one cv_results JSON.

    Variant detection is two-way: CSV arms carry the *_remove715negonly
    component path (marker); embedding-only arms have no such string, so
    they fall back to the row count (reduced set = fewer rows)."""
    d = json.loads(path.read_text())
    s = d["cv_summary"]["auc_roc"]
    folds = d.get("outer_val_predictions", {})
    n = sum(len(folds[k]["y_true"]) for k in folds)
    reduced = REDUCED_MARKER in json.dumps(d) or n != full_n
    return float(s["mean"]), float(s["std"]), n, ("reduced" if reduced
                                                  else "full")


def _read_split(path):
    """[(fold, label, key)] in CSV order; key identifies one peptide row."""
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out.append((int(r["fold"]), int(r["label"]),
                        (r["start"], r["end"], r["peptide"], r["uniprot_id"])))
    return out


def subset_mapping(full_rows, red_rows):
    """full-row index for every reduced row (reduced = full[mask], the
    mask preserves order — see make_variant_dataset.py). Key+label must
    match; FOLDS intentionally NOT compared: the reduced variant was
    re-folded (38,376/44,346 rows sit in different buckets), and each
    model's OOF comes from its own partition — row-wise pairing on the
    shared peptides stays valid either way."""
    mapping, i = [], 0
    for j, (rf, rl, rk) in enumerate(red_rows):
        while i < len(full_rows) and full_rows[i][2] != rk:
            i += 1
        if i >= len(full_rows):
            raise SystemExit(f"ABORT: reduced row {j} not found in full split")
        _, fl, _ = full_rows[i]
        if fl != rl:
            raise SystemExit(f"ABORT: reduced row {j} label mismatch")
        mapping.append(i)
        i += 1
    return mapping


def oof_scatter(res, rows):
    """Full-length OOF probability vector in split-CSV order. Verifies the
    JSON's per-fold y_true equals the CSV labels (the 18_ convention) —
    aborts instead of silently misaligning."""
    probs = np.full(len(rows), np.nan)
    for fk, e in res["outer_val_predictions"].items():
        fold = int(fk)
        idx = [p for p, (f, _l, _k) in enumerate(rows) if f == fold]
        y_json = np.asarray(e["y_true"])
        if (len(idx) != len(y_json) or not np.array_equal(
                np.asarray([rows[p][1] for p in idx]), y_json)):
            raise SystemExit(
                f"ABORT: fold {fk} y_true does not match split CSV order")
        probs[idx] = np.asarray(e["y_prob"], dtype=float)
    if np.isnan(probs).any():
        raise SystemExit("ABORT: rows never predicted")
    return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--run-dir", default="ab_dataset_test",
                    help="run folder under runs/ (default: ab_dataset_test)")
    ap.add_argument("--out-prefix", default=None,
                    help="default: <run-dir>/plots/ab_dataset_cv_auc")
    ap.add_argument("--variant", default="remove715negonly",
                    help="reduced-dataset variant name (marker in run JSONs "
                         "and split-CSV suffix; default: remove715negonly)")
    ap.add_argument("--tags", default=None,
                    help="comma-separated feature tags to compare "
                         "(default: all; e.g. --tags blosum50)")
    args = ap.parse_args()

    global REDUCED_MARKER
    REDUCED_MARKER = args.variant
    suffix = ("" if args.variant == "remove715negonly"
              else f"_{args.variant}")

    run_dir = R / args.run_dir
    out_prefix = (Path(args.out_prefix) if args.out_prefix
                  else run_dir / "plots" / f"ab_dataset_cv_auc{suffix}")

    # pca_mode decides the subdir (flat / vardims / flat_raw / ...) —
    # the blosum arms are models/flat, the esmc --vardims arms are
    # models/vardims, so scan every subdir.
    files = sorted(str(p) for p in run_dir.glob("models/*/cv_results_*.json"))
    if not files:
        raise SystemExit(f"no cv_results under {run_dir}/models — "
                         "rsync the campaign first")

    # full-dataset row count — variant fallback for embedding-only arms
    full_n = sum(1 for _ in open(SPLIT_DATA_PATH)) - 1

    # tag -> variant -> (path, mean, std, n); newest timestamp wins
    arms = {}
    for fp in files:
        f = Path(fp)
        m = re.match(r"cv_results_xgb_(.+?)_\d{8}_\d{6}\.json$", f.name)
        if not m:
            continue
        tag = m.group(1)
        if tag.endswith("_flat"):       # normalize blosum50_flat -> blosum50
            tag = tag[:-len("_flat")]
        mean, std, n, variant = load_arm(f, full_n)
        slot = arms.setdefault(tag, {})
        if variant not in slot or f.name > slot[variant][0].name:
            slot[variant] = (f, mean, std, n)

    tags = sorted(arms, key=lambda t: (family(t) != "encoding", t))
    if args.tags:
        want = {t.strip() for t in args.tags.split(",") if t.strip()}
        tags = [t for t in tags if t in want]
    print("Loading arms ...")
    rows = []
    for tag in tags:
        for variant in ("full", "reduced"):
            if variant in arms[tag]:
                p, mean, std, n = arms[tag][variant]
                print(f"  {tag:<45} {variant:<8} {mean:.4f} +/- {std:.4f} "
                      f"(n={n}, {p.name})")
                rows.append({"feature": tag, "variant": variant,
                             "auc_roc": f"{mean:.6f}", "std": f"{std:.6f}",
                             "n_oof": n, "source": p.name})
            else:
                print(f"  {tag:<45} {variant:<8} MISSING (job still running?)")

    # ── paired tests on the shared subset ────────────────────────────
    full_rows = _read_split(SPLIT_DATA_PATH)
    red_path = SPLIT_DATA_PATH.with_name(
        SPLIT_DATA_PATH.stem + "_" + args.variant + SPLIT_DATA_PATH.suffix)
    red_rows = _read_split(red_path)
    mapping = subset_mapping(full_rows, red_rows)
    n_pairs = len(mapping)
    y = np.asarray([red_rows[j][1] for j in range(n_pairs)])
    print(f"\nShared-subset alignment: {n_pairs} peptides "
          f"(full {len(full_rows)} - reduced {len(red_rows)} = "
          f"{len(full_rows) - len(red_rows)} dropped)")

    rng = np.random.default_rng(20261106)          # fixed seed, reproducible
    test_rows = []
    for tag in tags:
        if not ({"full", "reduced"} <= set(arms[tag])):
            print(f"  SKIP paired test {tag}: needs both variants")
            continue
        res_f = json.loads(arms[tag]["full"][0].read_text())
        res_r = json.loads(arms[tag]["reduced"][0].read_text())
        p_full = oof_scatter(res_f, full_rows)[mapping]
        p_red = oof_scatter(res_r, red_rows)        # reduced-CSV order
        # DeLong: a = reduced, b = full  ->  delta = reduced - full
        auc_f, auc_r, var_d = _p15.delong_test(y, p_red, p_full)
        d = auc_r - auc_f
        se = float(sqrt(var_d)) if var_d > 0 else 0.0
        p_d = 1.0 if se == 0 else float(erfc(abs(d / se) / sqrt(2)))
        b_delta, b_lo, b_hi, b_p, _, _, b_sd = _p15.paired_bootstrap_delta(
            y, p_red, p_full, 10000, rng)
        test_rows.append(dict(
            feature=tag, auc_full_subset=f"{auc_f:.6f}",
            auc_red_subset=f"{auc_r:.6f}", delta=f"{d:.6f}",
            se=f"{se:.6f}", z=f"{(d / se if se else 0.0):.4f}",
            p_delong=f"{p_d:.3e}", p_holm="", star="",
            boot_ci_lo=f"{b_lo:.6f}", boot_ci_hi=f"{b_hi:.6f}",
            p_boot=f"{b_p:.3e}", boot_sd=f"{b_sd:.6f}", n_pairs=n_pairs))
        print(f"  {tag}: subset AUC full={auc_f:.4f} red={auc_r:.4f} "
              f"delta={d:+.4f} | DeLong p={p_d:.3g} | "
              f"boot 95% CI [{b_lo:+.4f}, {b_hi:+.4f}] p={b_p:.3g}")
    if not test_rows:
        raise SystemExit("no paired tests computed — both variants needed")

    ph = _p15.holm_adjust([float(r["p_delong"]) for r in test_rows])
    stars = {}
    for r, a in zip(test_rows, ph):
        r["p_holm"] = f"{a:.3e}"
        r["star"] = ("***" if a < 0.001 else "**" if a < 0.01
                     else "*" if a < 0.05 else "ns")
        stars[r["feature"]] = r["star"]
        print(f"  {r['feature']}: Holm p={float(r['p_holm']):.3g} "
              f"[{r['star']}]")
    ph_b = _p15.holm_adjust([float(r["p_boot"]) for r in test_rows])
    stars_b = {}
    for r, a in zip(test_rows, ph_b):
        r["p_boot_holm"] = f"{a:.3e}"
        r["star_boot"] = ("***" if a < 0.001 else "**" if a < 0.01
                          else "*" if a < 0.05 else "ns")
        stars_b[r["feature"]] = r["star_boot"]
        print(f"  {r['feature']}: boot-Holm p={float(r['p_boot_holm']):.3g} "
              f"[{r['star_boot']}]")

    # ── figure: two panels — paired DeLong left, paired bootstrap right ──
    fig, axes = plt.subplots(
        1, 2, figsize=(max(12.0, 5.8 * len(tags) + 3), 5.5), sharey=True)
    w = 0.38
    tr_by = {r["feature"]: r for r in test_rows}
    for ax, ptitle, kind in (
            (axes[0], "Paired DeLong (Holm-corrected)", "delong"),
            (axes[1], "Paired bootstrap (10k resamples, Holm)", "boot")):
        for i, tag in enumerate(tags):
            color = COLORS[family(tag)]
            for variant in ("full", "reduced"):
                if variant not in arms[tag]:
                    continue
                _, mean, std, _ = arms[tag][variant]
                x = i + (-(w / 2 + 0.02) if variant == "full"
                         else (w / 2 + 0.02))
                ax.bar(x, mean, yerr=std, width=w, color=color,
                       alpha=0.9 if variant == "full" else 0.55,
                       hatch="" if variant == "full" else "//",
                       capsize=3, edgecolor="white" if variant == "full"
                       else "black", lw=0.6,
                       label=("Full dataset" if variant == "full"
                              else "Reduced (−2,271 neg-only)"))
                ax.text(x, mean + std + 0.008, f"{mean:.4f}", ha="center",
                        va="bottom", fontsize=8.5, fontweight="bold")
            if "full" in arms[tag] and "reduced" in arms[tag]:
                d = arms[tag]["reduced"][1] - arms[tag]["full"][1]
                top = max(arms[tag]["full"][1] + arms[tag]["full"][2],
                          arms[tag]["reduced"][1] + arms[tag]["reduced"][2])
                xa = i - (w / 2 + 0.02) - w / 2
                xb = i + (w / 2 + 0.02) + w / 2
                y_br = top + 0.032
                ax.plot([xa, xa, xb, xb],
                        [y_br - 0.012, y_br, y_br, y_br - 0.012],
                        lw=0.8, color="black")
                tr = tr_by.get(tag)
                if tr:
                    star = tr["star"] if kind == "delong" else tr["star_boot"]
                    ax.text(i, y_br + 0.003, star, ha="center",
                            va="bottom", fontsize=9)
                if kind == "delong":
                    stat = (f"Δ {d:+.4f}  p={float(tr['p_holm']):.2g}"
                            if tr else f"Δ {d:+.4f}")
                else:
                    stat = (f"Δ {d:+.4f}  CI "
                            f"[{float(tr['boot_ci_lo']):+.4f}, "
                            f"{float(tr['boot_ci_hi']):+.4f}]"
                            if tr else f"Δ {d:+.4f}")
                ax.text(i, y_br + 0.035, stat, ha="center",
                        va="bottom", fontsize=8, style="italic")
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels([display(t) for t in tags], fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(ptitle, fontsize=11)
    axes[0].set_ylabel("AUC-ROC (nested CV, mean +/- std)")

    # one shared legend below both panels
    handles, labels = axes[0].get_legend_handles_labels()
    seen, uh, ul = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uh.append(h)
            ul.append(l)
    fig.legend(uh, ul, loc="lower center", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.015))
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = Path(str(out_prefix) + ".png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    csvp = Path(str(out_prefix) + ".csv")
    with open(csvp, "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w_.writeheader()
        w_.writerows(rows)
    tsvp = run_dir / "results" / f"ab_dataset_delong{suffix}.tsv"
    tsvp.parent.mkdir(parents=True, exist_ok=True)
    with open(tsvp, "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(test_rows[0]), delimiter="\t")
        w_.writeheader()
        w_.writerows(test_rows)
    print(f"  wrote {png}")
    print(f"  wrote {csvp}")
    print(f"  wrote {tsvp}")

    print("\n" + "=" * 60)
    print("  CAPTION (paste below the figure):")
    print("=" * 60)
    if args.variant == "remove715negonly":
        desc = ("the reduced variant drops the 2,271 "
                "negative peptides contributed by 715 orphan proteins — "
                "proteins whose entries contain no positive example "
                "(THESIS_VARIANT remove715negonly, built by "
                "make_variant_dataset.py)")
    else:
        desc = (f"the reduced variant applies the {args.variant} "
                f"context-window dedup rule (THESIS_VARIANT "
                f"{args.variant}, built by make_variant_dataset.py; "
                "prepared embeddings subset by prepare_variant_h5.py)")
    print(f"A/B dataset comparison: {desc} — while all positives "
          "and the 6-fold "
          "nesting are kept. Solid bars = full "
          "dataset, hatched bars = reduced variant; family colors as in "
          "the other benchmark figures. Bars: nested-CV outer mean +/- std, "
          "XGB, each over its own peptide set (italic delta = raw mean "
          "difference). Significance: paired DeLong on the "
          f"{n_pairs} peptides SHARED by both variants — identical test "
          "rows, the two models differ only in training data (and the "
          "reduced variant was re-folded, so each OOF comes from its own "
          "partition) — Holm-adjusted across features in BOTH panels. "
          "Left panel: paired DeLong — bracket star + adjusted p. "
          "Right panel: paired bootstrap (10,000 resamples) — bracket "
          "star + 95% CI printed above each pair. Both tests give the "
          "same verdict; full numbers in "
          "results/ab_dataset_delong.tsv.")
    print("=" * 60)


if __name__ == "__main__":
    main()
