#!/usr/bin/env python3
"""Fetch UniProt FASTAs for Assarsson peptides lacking mol_parent_iri.

Maps the paper's VACWR locus codes (from the `orf` column) to UniProt
accessions via the UniProt REST API, downloads FASTAs for accessions not
already in data/raw/fasta/assarsson_protein_fastas/, and writes the
orf -> accession lookup to data/raw/external/assarsson/assarsson_orf_accession.csv.

Usage: python3 src/util/fetch_assarsson_fastas.py
"""

import csv
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MERGED = ROOT / "data/raw/external/assarsson/merged_assarsson_data.csv"
FASTA_DIR = ROOT / "data/raw/fasta/assarsson_protein_fastas"
OUT_MAP = ROOT / "data/raw/external/assarsson/assarsson_orf_accession.csv"
ORGANISM = 10254  # Vaccinia virus (strain Western Reserve)


def query_uniprot(vacwr: str) -> str | None:
    q = urllib.parse.urlencode(
        {"query": f"(gene_exact:{vacwr}) AND (organism_id:{ORGANISM})",
         "format": "fasta", "fields": "accession"}
    )
    url = f"https://rest.uniprot.org/uniprotkb/search?{q}"
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode()
    m = re.search(r"\|([A-Z0-9]+)\|", text.splitlines()[0]) if text.strip() else None
    return m.group(1) if m else None


def fetch_fasta(accession: str) -> str | None:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read().decode()
    except Exception as e:  # noqa: BLE001
        print(f"  FASTA fetch failed for {accession}: {e}", file=sys.stderr)
        return None


def main() -> None:
    rows = list(csv.DictReader(MERGED.open()))
    # ORF codes only for rows that lack mol_parent_iri
    orfs = sorted(
        {m.group(1) for r in rows
         if (r["mol_parent_iri"] in ("", "NA"))
         and (m := re.search(r"(VACWR\d+)", r["orf"]))}
    )
    print(f"Unique VACWR codes needing mapping: {len(orfs)}")

    mapping: dict[str, str] = {}
    for v in orfs:
        acc = query_uniprot(v)
        print(f"  {v} -> {acc}")
        mapping[v] = acc or ""
        time.sleep(0.3)  # be nice to the REST API

    FASTA_DIR.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in FASTA_DIR.glob("*.fasta")}
    for acc in sorted(set(mapping.values()) - {""}):
        if acc in existing:
            print(f"  {acc} already present, skipping download")
            continue
        fasta = fetch_fasta(acc)
        if fasta:
            (FASTA_DIR / f"{acc}.fasta").write_text(fasta)
            print(f"  saved {acc}.fasta")

    with OUT_MAP.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vacwr_code", "uniprot_id"])
        for k, v in mapping.items():
            w.writerow([k, v])
    print(f"Wrote {OUT_MAP} ({len(mapping)} entries)")


if __name__ == "__main__":
    main()
