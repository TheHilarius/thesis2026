# ProteinMPNN (vendored)

Pinned upstream: **dauparas/ProteinMPNN @ `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`** (`main`).

The clone under `vendor/ProteinMPNN/` is **git-ignored** (216M: nested `.git` + all
checkpoints + example weights). It is reproduced, not committed. The only local
change is the `ProteinMPNN.encode()` method (encoder-only per-residue embeddings,
no decoder / no sequence leak), captured in `vendor/ProteinMPNN_encoder.patch`.

## Reproduce

```bash
# from repo root
git clone https://github.com/dauparas/ProteinMPNN.git vendor/ProteinMPNN
cd vendor/ProteinMPNN
git checkout 8907e6671bfbfc92303b5f79c4b5e6ce47cdef57
git apply ../../vendor/ProteinMPNN_encoder.patch
cd ../..
```

## What the patch adds

`encode(self, X, mask, residue_idx, chain_encoding_all)` on the `ProteinMPNN`
class (near the top of `forward()`). It runs only the feature/encoder block and
returns `h_V [B, L, 128]` (post-LayerNorm, masked). `forward()` / `sample()` are
untouched, so upstream behavior is unchanged.

## Checkpoints used

Encoder embeddings in this thesis use `vanilla_model_weights/`:
- `v_48_020.pt` — 0.20 Å training noise (default; robust to AF2 error) → `mpnn_v48020_*`
- `v_48_002.pt` — 0.02 Å training noise (sensitivity) → `mpnn_v48002_*`

`v_48_010.pt` / `v_48_030.pt` are present but unused. `soluble_model_weights/`,
`ca_model_weights/`, `ligandmpnn` (separate repo) are unused.

## Runtime deps

`torch`, `numpy` only for the encoder path (`protein_mpnn_utils.py`). `pdbx`,
`scipy`, `dateutil` appear in helper/training scripts that are **not** imported by
`src/tools/proteinmpnn/embed_peptides_mpnn.py`. `h5py`, `pandas`, `biopython` are
needed by the embedding/validation scripts themselves (installed in `env_mpnn` /
`env_mpnn_server`).
