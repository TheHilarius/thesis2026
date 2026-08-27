"""
python src/fetch/report_structure_coverage.py \
    --df data/processed/df_all.csv \
    --h5 data/processed/embeddings/esmif_structure_embeddings.h5 \
    --missing data/processed/structures/logs/missing_structures.tsv \
    --out results/structure_coverage
"""
import argparse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import h5py


# -----------------------------
# Helpers
# -----------------------------

def load_df(path):
    print(f"Loading df_all.csv: {path}")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df)}")
    print(f"  Proteins (unique uniprot_id): {df['uniprot_id'].nunique()}")
    return df


def load_h5(h5_path):
    print(f"\nLoading HDF5: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        uniprot_ids = f["uniprot_ids"][:].astype(str)
        peptide_emb = f["peptide_if_struct"][:]
        n_flank_emb = f["n_flank_if_struct"][:]
        c_flank_emb = f["c_flank_if_struct"][:]

    zero = lambda x: np.all(x == 0, axis=1)

    return {
        "uniprot_ids": uniprot_ids,
        "peptide_zero": zero(peptide_emb),
        "nflank_zero": zero(n_flank_emb),
        "cflank_zero": zero(c_flank_emb),
        "n_samples": len(uniprot_ids),
    }


def load_missing_structures(missing_path):
    if missing_path is None or not os.path.exists(missing_path):
        return set()

    df = pd.read_csv(missing_path, sep="\t", header=None)
    # assume first column is uniprot_id
    return set(df.iloc[:, 0].astype(str))


# -----------------------------
# Protein-level coverage
# -----------------------------

def protein_coverage(df, h5_uniprot_ids, missing_structures_set):
    df_prots = set(df["uniprot_id"].astype(str).unique())
    h5_prots = set(h5_uniprot_ids)

    # proteins that actually have embeddings
    available_prots = df_prots & h5_prots

    # proteins explicitly missing structure
    missing_prots = df_prots - h5_prots

    # also include known missing structure list (if provided)
    missing_prots = missing_prots | missing_structures_set

    stats = {
        "total_proteins_in_df": len(df_prots),
        "proteins_with_any_embedding": len(available_prots),
        "proteins_missing_structure": len(missing_prots),
        "coverage_rate": len(available_prots) / len(df_prots),
    }

    return stats, available_prots, missing_prots


# -----------------------------
# Peptide-level coverage
# -----------------------------

def peptide_coverage(df, h5_data):
    n_total = len(df)

    peptide_zero = h5_data["peptide_zero"]
    n_zero = h5_data["nflank_zero"]
    c_zero = h5_data["cflank_zero"]

    stats = {
        "peptides_total": n_total,
        "peptide_missing_rate": float(peptide_zero.mean()),
        "nflank_missing_rate": float(n_zero.mean()),
        "cflank_missing_rate": float(c_zero.mean()),
    }

    return stats


# -----------------------------
# Protein-level peptide missingness
# -----------------------------

def per_protein_missingness(df, h5_data):
    """
    For each protein:
    compute fraction of its peptides with missing structure
    """

    df = df.copy()
    df["peptide_missing"] = h5_data["peptide_zero"]
    df["nflank_missing"] = h5_data["nflank_zero"]
    df["cflank_missing"] = h5_data["cflank_zero"]

    grouped = df.groupby("uniprot_id").agg(
        peptide_missing_rate=("peptide_missing", "mean"),
        nflank_missing_rate=("nflank_missing", "mean"),
        cflank_missing_rate=("cflank_missing", "mean"),
        n_peptides=("peptide_missing", "size"),
    )

    summary = {
        "proteins_all_peptides_missing_structure": int((grouped["peptide_missing_rate"] == 1.0).sum()),
        "proteins_any_structure": int((grouped["peptide_missing_rate"] < 1.0).sum()),
        "proteins_mean_peptide_missing_rate": float(grouped["peptide_missing_rate"].mean()),
    }

    return summary, grouped


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--missing", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load inputs
    # -------------------------
    df = load_df(args.df)
    h5 = load_h5(args.h5)
    missing_set = load_missing_structures(args.missing)

    # -------------------------
    # Protein-level stats
    # -------------------------
    protein_stats, available_prots, missing_prots = protein_coverage(
        df, h5["uniprot_ids"], missing_set
    )

    # -------------------------
    # Peptide-level stats
    # -------------------------
    peptide_stats = peptide_coverage(df, h5)

    # -------------------------
    # Protein peptide breakdown
    # -------------------------
    protein_pep_stats, protein_table = per_protein_missingness(df, h5)

    # -------------------------
    # Combine report
    # -------------------------
    report = {
        "protein_level": protein_stats,
        "peptide_level": peptide_stats,
        "protein_peptide_level": protein_pep_stats,
    }

    # -------------------------
    # Save outputs
    # -------------------------
    print("\nSaving results...")

    with open(out_dir / "structure_coverage_report.json", "w") as f:
        json.dump(report, f, indent=2)

    protein_table.to_csv(out_dir / "protein_level_missingness.csv")

    # -------------------------
    # Print summary
    # -------------------------
    print("\n==============================")
    print("STRUCTURE COVERAGE SUMMARY")
    print("==============================")

    print("\n[Protein level]")
    for k, v in protein_stats.items():
        print(f"  {k}: {v}")

    print("\n[Peptide level]")
    for k, v in peptide_stats.items():
        print(f"  {k}: {v:.4f}")

    print("\n[Protein peptide structure]")
    for k, v in protein_pep_stats.items():
        print(f"  {k}: {v}")

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
