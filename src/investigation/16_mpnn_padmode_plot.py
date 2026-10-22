#!/usr/bin/env python3
"""
16_mpnn_padmode_plot.py
ProteinMPNN comparison figures — padmode_choice figure convention:
AUC-ROC bars + DeLong significance brackets (raw p + Holm p_adj), one
figure for all peptides and a _subset duplicate for padded peptides only.

Campaigns (--campaign):
    padmode  zero-pad vs boundary-repeat @ v_48_020
             -> runs/mpnn_padmode_v48020/plots/mpnn_padmode_auc_roc[_subset]
    ckpt     4 train-noise checkpoints (zeropad) + the ESM-IF comparison
             arm -> runs/mpnn_ckpt_zeropad/plots/mpnn_ckpt_auc_roc[_subset]
             Hub brackets: the default checkpoint (bars[0], v48002) vs
             EVERY other bar; pairs are merged from the ckpt_<tk>[_full]
             and ckpt_vs_esmif[_full] families of paired_tests.csv.

Single data source: 15_padmode_significance.py's paired_tests.csv (per-arm
auc + delta + DeLong p/Holm) — bars, brackets and CSV sidecar stay mutually
consistent (no mixing with cv_summary).

Usage:
    python src/investigation/16_mpnn_padmode_plot.py --smoke
    python src/investigation/16_mpnn_padmode_plot.py
    python src/investigation/16_mpnn_padmode_plot.py --campaign ckpt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
PIPELINE_DIR = SRC_DIR.parent / "pipeline" / "python"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# padmode_choice reference: reuse its star/p-text formatting so bracket
# wording is identical across campaigns
import plot_padmode_data_compare as pdp

from config import PROJECT_ROOT  # noqa: E402

# Paul Tol bright (colour-blind safe); green = MPNN group, blue = ESM-IF
MPNN_COLOR = "#44AA99"
ESMIF_COLOR = "#4477AA"
GROUP_OF_COLOR = {MPNN_COLOR: "ProteinMPNN 90%", ESMIF_COLOR: "ESM-IF"}

# CSV schema: 15_ stores model_a/model_b per pair row -> auc_a/auc_b.
# Bars: (model key in paired_tests.csv, label, feature-set key, colour).
CAMPAIGNS = {
    "padmode": {
        "run": PROJECT_ROOT / "runs" / "mpnn_padmode_v48020",
        "stem": "mpnn_padmode_auc_roc",
        # (suffix, [paired_tests.csv groups merged], scope)
        "groups": [("", ["padmode_proteinmpnn_full"], "all peptides"),
                   ("_subset", ["padmode_proteinmpnn"],
                    "padded peptides only")],
        "bars": [("zero", "zero-pad", "emb_mpnn_v48020_zeropad", MPNN_COLOR),
                 ("boundary", "boundary-repeat", "emb_mpnn_v48020_boundary",
                  MPNN_COLOR)],
    },
    "ckpt": {
        "run": PROJECT_ROOT / "runs" / "mpnn_ckpt_zeropad",
        "stem": "mpnn_ckpt_auc_roc",
        "groups": [
            ("", ["ckpt_proteinmpnn_full", "ckpt_vs_esmif_full"],
             "all peptides"),
            ("_subset", ["ckpt_proteinmpnn", "ckpt_vs_esmif"],
             "padded peptides only")],
        "bars": [
            ("v48002", "v_48_002\n(0.02 Å)", "emb_mpnn_v48002_zeropad",
             MPNN_COLOR),
            ("v48010", "v_48_010\n(0.10 Å)", "emb_mpnn_v48010_zeropad",
             MPNN_COLOR),
            ("v48020", "v_48_020\n(0.20 Å)", "emb_mpnn_v48020_zeropad",
             MPNN_COLOR),
            ("v48030", "v_48_030\n(0.30 Å)", "emb_mpnn_v48030_zeropad",
             MPNN_COLOR),
            ("eos_bos_repeat@4308", "ESM-IF\nEOS/BOS 90%\n(k=4308)",
             "emb_esmif_eos_bos_repeat", ESMIF_COLOR),
            ("eos_bos_repeat@632", "ESM-IF\nEOS/BOS 60%\n(k=632)",
             "emb_esmif_eos_bos_repeat", ESMIF_COLOR)],
    },
}


def load_groups(paired_csv: Path, test_groups: list[str], arms: list[str]) -> dict:
    """Merged view of one figure's test groups -> per-arm AUCs + per-pair
    DeLong stats.

    Within one figure all groups share the same test rows, so an arm's AUC
    must agree across groups (asserted); pairs from every group merge into
    one lookup keyed by frozenset of the two arm names.
    """
    arm_auc, pairs = {}, {}
    with open(paired_csv, newline="") as f:
        for r in csv.DictReader(f):
            tg = r["test_group"].strip()
            if tg not in test_groups:
                continue
            for side in ("a", "b"):
                arm, auc = r[f"model_{side}"].strip(), float(r[f"auc_{side}"])
                if arm not in arms:
                    continue
                if arm in arm_auc:
                    assert abs(arm_auc[arm] - auc) < 1e-12, (arm, arm_auc[arm], auc)
                arm_auc[arm] = auc
            pairs[frozenset((r["model_a"].strip(), r["model_b"].strip()))] = {
                "delta": float(r["delta_auc"]),
                "p": float(r["p_delong"]),
                "p_adj": float(r["p_delong_holm"]),
                "n_rows": int(float(r["n_rows"])),
            }
    missing = [a for a in arms if a not in arm_auc]
    if missing:
        raise SystemExit(f"{test_groups}: no rows for arms {missing} in {paired_csv}")
    return {"auc": arm_auc, "pairs": pairs}


def plot_bars(g: dict, bars: list[tuple], title: str, outfile: Path) -> None:
    """AUC bars + hub DeLong brackets (bars[0] vs every other bar) —
    padmode_choice style: widest span on top, text = raw p + Holm p_adj."""
    labels = [lbl for _, lbl, _, _ in bars]
    values = [g["auc"][key] for key, _, _, _ in bars]

    x = np.arange(len(bars))
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.bar(x, values, color=[c for _, _, _, c in bars], width=0.62,
           edgecolor="black", linewidth=0.5)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8)
    ax.set_ylim(0, 1.0)

    # hub brackets: bars[0] (default arm) vs every other bar with a pair
    hub, brackets = 0, []
    for j in range(1, len(bars)):
        pair = g["pairs"].get(frozenset((bars[hub][0], bars[j][0])))
        if pair is not None:
            brackets.append((j, pair))
    brackets.sort(key=lambda b: -abs(b[0] - hub))   # widest span on top
    y0 = max(values) * 1.08
    lvl_step = min(max(values) * 0.075,
                   (0.94 - y0 - max(values) * 0.03) / max(len(brackets), 1))
    for lvl, (j, pair) in enumerate(reversed(brackets)):
        y_br = y0 + lvl * lvl_step
        ax.plot([x[hub], x[hub], x[j], x[j]],
                [y_br * 0.985, y_br, y_br, y_br * 0.985],
                color="black", lw=0.9)
        star_src = pair["p_adj"] if pair["p_adj"] is not None else pair["p"]
        ax.text((x[hub] + x[j]) / 2, y_br + lvl_step * 0.12,
                f"{pdp.fmt_p_pair(pair['p'], pair['p_adj'])} "
                f"({pdp.stars(star_src)})",
                ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUC-ROC")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    # divider + legend between colour groups (padmode_choice convention)
    colors = [c for _, _, _, c in bars]
    for i in range(1, len(colors)):
        if colors[i] != colors[i - 1]:
            ax.axvline(i - 0.5, color="grey", linestyle="--", linewidth=0.8)
    present = [c for c in GROUP_OF_COLOR if c in colors]
    if len(present) > 1:
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black")
                   for c in present]
        ax.legend(handles, [GROUP_OF_COLOR[c] for c in present],
                  fontsize=11, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  frameon=False)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {outfile}")

    csv_path = outfile.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "feature_set", "label", "metric", "value"])
        for key, lbl, feat, color in bars:
            w.writerow([GROUP_OF_COLOR[color], feat, lbl.replace("\n", " "),
                        "auc_roc", g["auc"][key]])
    print(f"  saved: {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ProteinMPNN comparison figures (padmode_choice style).")
    ap.add_argument("--campaign", default="padmode", choices=sorted(CAMPAIGNS),
                    help="padmode = zero vs boundary @ v_48_020; "
                         "ckpt = 4 checkpoints + ESM-IF comparison arm")
    ap.add_argument("--run", default=None,
                    help="override campaign run folder")
    ap.add_argument("--out-dir", default=None,
                    help="default: <run>/plots")
    ap.add_argument("--smoke", action="store_true",
                    help="print the loaded rows and exit before plotting")
    args = ap.parse_args()

    camp = CAMPAIGNS[args.campaign]
    run_dir = Path(args.run) if args.run else Path(camp["run"])
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    paired_csv = run_dir / "results" / "paired_tests.csv"
    bars = camp["bars"]
    arms = [key for key, _, _, _ in bars]

    groups = [(suffix, scope, load_groups(paired_csv, sources, arms),
               sources)
              for suffix, sources, scope in camp["groups"]]
    for (_, scope, g, _), (_, sources, _) in zip(groups, camp["groups"]):
        auc_str = "  ".join(f"{a}={g['auc'][a]:.4f}" for a in arms)
        n_rows = next(iter(g["pairs"].values()))["n_rows"]
        print(f"  {sources} (n={n_rows:,}): {auc_str}")
        for other in arms[1:]:
            pair = g["pairs"].get(frozenset((arms[0], other)))
            if pair:
                print(f"    {arms[0]} vs {other:<14} delta={pair['delta']:+.4f} "
                      f"p={pair['p']:.3g} p_adj={pair['p_adj']:.3g}")
    if args.smoke:
        return

    for suffix, scope, g, _ in groups:
        n_rows = next(iter(g["pairs"].values()))["n_rows"]
        plot_bars(
            g, bars,
            f"ProteinMPNN — held-out AUC-ROC, {scope}\n"
            f"nested CV (OOF), n={n_rows:,}; DeLong brackets: "
            f"{bars[0][1].splitlines()[0]} vs each arm, raw p + Holm p_adj",
            out_dir / f"{camp['stem']}{suffix}.png",
        )
    print(f"\n  plots in: {out_dir}")


if __name__ == "__main__":
    main()
