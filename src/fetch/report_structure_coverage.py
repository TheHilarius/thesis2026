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
        if 'context_if_struct' in f:
            ctx_emb = f["context_if_struct"][:]
        elif 'context_emb' in f:
            ctx_emb = f["context_emb"][:]
        else:
            raise KeyError(f"No context embedding dataset found in {h5_path}")

    zero = lambda x: np.all(x == 0, axis=1)

    return {
        "uniprot_ids": uniprot_ids,
        "ctx_zero": zero(ctx_emb),
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

    ctx_zero = h5_data["ctx_zero"]

    stats = {
        "peptides_total": n_total,
        "ctx_missing_rate": float(ctx_zero.mean()),
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
    df["ctx_missing"] = h5_data["ctx_zero"]

    grouped = df.groupby("uniprot_id").agg(
        ctx_missing_rate=("ctx_missing", "mean"),
        n_peptides=("ctx_missing", "size"),
    )

    summary = {
        "proteins_all_peptides_missing_structure": int((grouped["ctx_missing_rate"] == 1.0).sum()),
        "proteins_any_structure": int((grouped["ctx_missing_rate"] < 1.0).sum()),
        "proteins_mean_ctx_missing_rate": float(grouped["ctx_missing_rate"].mean()),
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
