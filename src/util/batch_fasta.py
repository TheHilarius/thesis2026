#!/usr/bin/env python3

import sys
from pathlib import Path


def read_fasta_entries(filepath):
    """Read a multi-entry FASTA file and return list of (header, sequence) tuples"""
    entries = []
    with open(filepath) as f:
        header, seq_lines = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seq_lines)))
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            entries.append((header, "".join(seq_lines)))
    return entries


def main():
    if len(sys.argv) not in [3, 4]:
        print("Usage: python batch_fasta.py <input_fasta> <output_dir> [batch_size]")
        print("Example: python src/util/batch_fasta.py data/raw/fasta/combined.fasta data/raw/fasta/fasta_batches 50")
        sys.exit(1)

    input_fasta = Path(sys.argv[1])
    output_dir  = Path(sys.argv[2])
    batch_size  = int(sys.argv[3]) if len(sys.argv) == 4 else 500

    if not input_fasta.exists():
        print(f"[ERROR] Input file not found: {input_fasta}")
        sys.exit(1)

    if output_dir.exists():
        print(f"[ERROR] Output directory already exists: {output_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True)

    print(f"Reading: {input_fasta}")
    entries = read_fasta_entries(input_fasta)
    print(f"Total sequences: {len(entries)}")

    n_batches  = (len(entries) + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch        = entries[i * batch_size:(i + 1) * batch_size]
        batch_num    = str(i + 1).zfill(3)
        output_path  = output_dir / f"batch_{batch_num}.fasta"

        with open(output_path, "w") as f:
            for header, seq in batch:
                f.write(f"{header}\n{seq}\n")

        print(f"[OK] Batch {batch_num}: {len(batch)} sequences → {output_path}")

    print(f"\nDone. {n_batches} batches of max {batch_size} sequences created in {output_dir}")


if __name__ == "__main__":
    main()

"""
python src/util/batch_fasta.py \
    data/raw/fasta/combined.fasta \
    data/raw/fasta/fasta_batches
"""
