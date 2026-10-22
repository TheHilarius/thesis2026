"""
05_cnn_models.py:c
Simplest 1D-CNN for fixed-29 window embeddings. Mirrors MODEL_REGISTRY
style in config.py (display_name + params dict).

Input convention: (N, D, L=29) float32, mask (N, L) bool where True = PAD.
"""

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - import-time hint only
    raise ImportError("torch required: use env_esmc_server or GPU env")

CNN_REGISTRY = {
    "cnn_simple": {
        "display_name": "Simple CNN (1x Conv k=3 + GAP)",
        "params": {"channels": 32, "kernel": 3},
    },
    "cnn_2layer": {
        "display_name": "CNN 2-layer (2x Conv k=3 + GAP, receptive field 5)",
        "params": {"channels": 32, "kernel": 3},
    },
}

DEFAULT_CNN_MODEL = "cnn_simple"


def get_cnn_config(model_key):
    if model_key not in CNN_REGISTRY:
        valid = ", ".join(sorted(CNN_REGISTRY.keys()))
        raise KeyError(f"Unknown CNN model '{model_key}'. Valid: {valid}")
    return CNN_REGISTRY[model_key]


class SimpleCNN(nn.Module):
    """One Conv1d over L=29 + ReLU + masked mean + linear.

    Per peptide: 29 D-vectors in, 1 logit out. Kernel k=3 looks at
    3 neighbour slots at a time, same weights everywhere.

    Masking is post-conv only: the filler (padtoken / BOS-EOS vectors)
    stays as-is into the conv, pads are zeroed after conv and excluded
    from the pool via renormalized mean. No pre-zeroing.
    """

    def __init__(self, in_channels, channels=32, kernel=3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, channels, kernel,
                              padding=kernel // 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(channels, 1)

    def forward(self, x, mask):
        # x: (N, D, L); mask: (N, L) True=PAD
        h = self.relu(self.conv(x))  # (N, C, L)
        real = (~mask).float().unsqueeze(1)  # (N, 1, L)
        h = h * real
        denom = real.sum(-1).clamp(min=1.0)  # (N, 1)
        h = h.sum(-1) / denom  # masked mean over real slots only
        return self.fc(h).squeeze(-1)


class TwoLayerCNN(nn.Module):
    """2x Conv1d k=3 + ReLU, receptive field 5, then masked mean + linear.

    Same post-conv-only masking as SimpleCNN. Layer 2 can compare a
    peptide trigram against flanking trigrams; still translation
    equivariant, so no explicit peptide-vs-context position signal.
    """

    def __init__(self, in_channels, channels=32, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_channels, channels, kernel, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(channels, 1)

    def forward(self, x, mask):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        real = (~mask).float().unsqueeze(1)
        h = h * real
        denom = real.sum(-1).clamp(min=1.0)
        h = h.sum(-1) / denom
        return self.fc(h).squeeze(-1)


def build_cnn(model_key, in_channels, window=29):
    cfg = get_cnn_config(model_key)["params"]
    if model_key == "cnn_2layer":
        return TwoLayerCNN(in_channels, cfg["channels"], cfg["kernel"])
    return SimpleCNN(in_channels, cfg["channels"], cfg["kernel"])
