#!/usr/bin/env python3
"""
32_fake_protein_netmhcpan_logo.py — fake proteins -> NetMHCpan 4.2 binders
-> context windows -> XGB BLOSUM50 (_flank5_rmdup) ensemble -> decile logos.

Replicates how the training data was built (04_/05_ R scripts): negatives are
NetMHCpan-predicted binders (HLA-A02:01, 9-mer, NB==1 i.e. Rank<2%). Here the
proteins are synthetic, so every scored window is a plausible binder that no
MS has ever seen — the model's top/bottom deciles then show what it learned
beyond "binds MHC".

Stages (run separately):
  --smoke           generate 200 proteins, run LOCAL netMHCpan, report the
                    measured binder rate + runtime, print the protein count
                    needed for 1M binders
  --generate N      generate N unique fake proteins (uniform 20 AA, lengths
                    sampled from the real per-protein length pool), write
                    500-protein FASTA batches for the cluster array job
  --collect         parse runs/fake_protein_netmhcpan/netmhcpan/*.csv, keep
                    NB==1 9-mers, cut 19-mer windows (5+5 flanks, X pads at
                    termini, same as extract_flanks_safe), dedup windows,
                    exclude training full_context, cap at exactly 1M, score
                    with the 30-pickle _flank5_rmdup ensemble, write
                    top/bottom deciles + PPMs + score hist

Cluster: sbatch src/run/run_netmhcpan_fake_slurm.sh (array job, one batch per
task; binary path exported there — netMHCpan is not in the cluster PATH).

Outputs -> runs/fake_protein_netmhcpan/. Logos afterwards:
  Rscript src/investigation/31_random_peptide_logos.R runs/fake_proptide_netmhcpan
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parent.parent / "pipeline" / "python"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from config import PROJECT_ROOT

# reuse encoding/scoring machinery from the 31_ script (module name starts
# with a digit -> load via importlib)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "rpl", Path(__file__).resolve().parent / "31_random_peptide_logo.py")
rpl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rpl)

OUT = PROJECT_ROOT / "runs" / "fake_protein_netmhcpan"
FASTA_DIR = OUT / "fasta"
NMP_DIR = OUT / "netmhcpan"
SMOKE_DIR = OUT / "smoke"

TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "df_all_flank5_rmdup.csv"
LOCAL_NETMHCPAN = Path("/home/hilarius/netMHCpan-4.2/netMHCpan")  # tcsh wrapper (sets $NETMHCpan)

SEED = 20261126
TARGET_BINDERS = 1_000_000
HEADROOM = 1.2
BATCH_PROTEINS = 500
FLANK = 5
ALLELE = "HLA-A02:01"
AA20 = np.array([a for a in rpl.AA_ORDER if a != "X"])


def length_pool() -> np.ndarray:
    """Per-protein lengths of the real training proteins."""
    df = pd.read_csv(TRAIN_CSV, usecols=["uniprot_id", "protein_length"])
    return df.drop_duplicates("uniprot_id")["protein_length"].to_numpy()


def real_protein_set() -> set[str]:
    df = pd.read_csv(TRAIN_CSV, usecols=["sequence"])
    return set(df["sequence"].unique())


def generate_proteins(n: int, fasta_dir: Path, batch_size: int = BATCH_PROTEINS):
    """n unique uniform-random proteins -> FASTA batches of batch_size."""
    rng = np.random.default_rng(SEED)
    lens = length_pool()
    real = real_protein_set()
    print(f"length pool: {len(lens)} proteins, mean {lens.mean():.0f} aa")
    fasta_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    n_batches = (n + batch_size - 1) // batch_size
    t0 = time.time()
    idx = 0
    for b in range(n_batches):
        lines = []
        for _ in range(min(batch_size, n - idx)):
            while True:
                seq = "".join(AA20[rng.integers(0, 20, size=int(rng.choice(lens)))])
                if seq not in seen and seq not in real:
                    break
            seen.add(seq)
            idx += 1
            lines.append(f">fake_{idx:07d}\n{seq}\n")
        (fasta_dir / f"batch_{b:03d}.fasta").write_text("".join(lines))
    print(f"generated {idx} unique proteins in {n_batches} batches "
          f"({time.time() - t0:.1f}s) -> {fasta_dir}")


def run_netmhcpan(fasta: Path, out_csv: Path, binary: Path = LOCAL_NETMHCPAN) -> float:
    t0 = time.time()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(binary), "-a", ALLELE, "-l", "9", "-xls",
                    "-xlsfile", str(out_csv), "-f", str(fasta)],
                   check=True, capture_output=True)
    return time.time() - t0


def parse_binders(csv_path: Path) -> list[tuple[str, int, str, float]]:
    """NB==1 rows -> (ID, Pos, Peptide, Rank). Format matches functions.R
    load_netmhcpan_batches: 2 preamble lines, then whitespace columns."""
    rows = []
    with open(csv_path) as fh:
        for _ in range(2):
            next(fh)
        for line in fh:
            f = line.split()
            if len(f) != 9 or f[8] != "1":
                continue
            rows.append((f[2], int(f[0]), f[1], float(f[7])))
    return rows


def cut_window(protein: str, pos: int, peptide: str) -> str | None:
    """19-mer context. Binders within FLANK of a terminus are skipped —
    no X pads: synthetic windows must be fully inside the protein."""
    s = pos - 1                                   # 0-based peptide start
    assert protein[s:s + 9] == peptide, (pos, peptide)
    if s < FLANK or s + 9 + FLANK > len(protein):
        return None
    return protein[s - FLANK:s] + peptide + protein[s + 9:s + 9 + FLANK]


def smoke() -> None:
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    fasta = SMOKE_DIR / "smoke_200.fasta"
    rng = np.random.default_rng(SEED)
    lens = length_pool()
    lines = []
    for i in range(1, 201):
        seq = "".join(AA20[rng.integers(0, 20, size=int(rng.choice(lens)))])
        lines.append(f">smoke_{i:04d}\n{seq}\n")
    fasta.write_text("".join(lines))

    dt = run_netmhcpan(fasta, SMOKE_DIR / "smoke_200.csv")
    rows = parse_binders(SMOKE_DIR / "smoke_200.csv")
    lens_drawn = np.array([len(l.split("\n")[1]) for l in
                           fasta.read_text().split(">")[1:]])
    n_9mers = int((lens_drawn - 8).clip(min=0).sum())
    rate = len(rows) / n_9mers
    print(f"smoke: 200 proteins, {n_9mers} 9-mers, {len(rows)} binders "
          f"(rate {rate:.4%}), netMHCpan {dt:.1f}s "
          f"({n_9mers / dt:,.0f} 9-mers/s)")
    mean_len = lens.mean()
    need = TARGET_BINDERS * HEADROOM / (rate * (mean_len - 8))
    print(f"for 1M binders: ~{need:,.0f} proteins "
          f"(rate {rate:.3%}, mean len {mean_len:.0f}, headroom {HEADROOM})")


def load_proteins() -> dict[str, str]:
    prots = {}
    for f in sorted(glob.glob(str(FASTA_DIR / "batch_*.fasta"))) + \
               sorted(glob.glob(str(OUT / "smoke" / "smoke_200.fasta"))):
        for line in Path(f).read_text().splitlines():
            if line.startswith(">"):
                pid = line[1:].split()[0]
            elif line:
                prots[pid] = line
    return prots


def collect() -> None:
    csvs = sorted(glob.glob(str(NMP_DIR / "*.csv")))
    if not csvs:
        raise SystemExit(f"no netmhcpan CSVs in {NMP_DIR} — run the array job")
    prots = load_proteins()
    print(f"proteins loaded: {len(prots)} | netmhcpan batches: {len(csvs)}")

    train_windows: set[str] = set()
    with open(TRAIN_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            train_windows.add(row["full_context"])

    windows: set[str] = set()
    t0 = time.time()
    for c in csvs:
        for pid, pos, pep, _rank in parse_binders(Path(c)):
            w = cut_window(prots[pid], pos, pep)
            if w is not None and w not in train_windows:
                windows.add(w)
    print(f"binder windows: {len(windows):,} "
          f"(after train-exclusion + dedup, {time.time() - t0:.0f}s)")

    if len(windows) < TARGET_BINDERS:
        print(f"WARNING: below the 1M target ({len(windows):,}); continuing")
        seqs = list(windows)
    else:
        rng = np.random.default_rng(SEED)
        seqs = list(windows)
        keep = rng.choice(len(seqs), TARGET_BINDERS, replace=False)
        seqs = [seqs[i] for i in sorted(keep)]
        print(f"capped to exactly {len(seqs):,} (subsampled, seeded)")

    regime = rpl.MODEL_REGIMES["flank5_rmdup"]
    feature_names, positions = rpl.load_feature_names(regime)
    models = rpl.load_models(regime)

    probs = rpl.score(models, seqs)
    with gzip.open(OUT / "scores.csv.gz", "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sequence", "prob"])
        w.writerows(zip(seqs, probs))

    k = len(seqs) // 10
    order = np.argsort(probs)
    bottom_idx, top_idx = order[:k], order[-k:]
    top_seqs = [seqs[i] for i in top_idx]
    bot_seqs = [seqs[i] for i in bottom_idx]
    pd.DataFrame({"sequence": top_seqs, "prob": probs[top_idx]}) \
        .to_csv(OUT / "top10.csv", index=False)
    pd.DataFrame({"sequence": bot_seqs, "prob": probs[bottom_idx]}) \
        .to_csv(OUT / "bottom10.csv", index=False)
    rpl.ppm(top_seqs, positions).to_csv(OUT / "ppm_top10.csv")
    rpl.ppm(bot_seqs, positions).to_csv(OUT / "ppm_bottom10.csv")

    train_probs = rpl.score_training(
        models, feature_names,
        PROJECT_ROOT / "data" / "processed" / "df_blosum50_flank5_rmdup.csv")
    with gzip.open(OUT / "train_scores.csv.gz", "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prob"])
        w.writerows((f"{p:.6f}",) for p in train_probs)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(probs, bins=100, density=True, alpha=0.6,
            label=f"fake binders ({len(seqs):,})")
    ax.hist(train_probs, bins=100, density=True, alpha=0.6,
            label=f"training ({len(train_probs):,})")
    ax.axvline(probs[top_idx[0]], color="k", ls="--", lw=1,
               label=f"top10% cutoff = {probs[top_idx[0]]:.3f}")
    ax.axvline(probs[bottom_idx[-1]], color="k", ls=":", lw=1,
               label=f"bottom10% cutoff = {probs[bottom_idx[-1]]:.3f}")
    ax.set_xlabel("Predicted P(presented)")
    ax.set_ylabel("Density")
    ax.set_title("XGB BLOSUM50 (_flank5_rmdup) — NetMHCpan binder windows "
                 "from fake proteins vs training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "score_hist.png", dpi=200)
    print(f"done -> {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--generate", type=int, metavar="N")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.generate:
        generate_proteins(args.generate, FASTA_DIR)
    elif args.collect:
        collect()
    else:
        ap.error("choose --smoke / --generate N / --collect")


if __name__ == "__main__":
    main()
