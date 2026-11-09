"""
05_cnn_models.py:c
Simplest 1D-CNN for fixed-29 window embeddings. Mirrors MODEL_REGISTRY
style in config.py (display_name + params dict).

Input convention: (N, D, L=29) float32. Naive GAP over all 29 slots.
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
    "cnn_simple_k5": {
        "display_name": "Simple CNN (1x Conv k=5 + GAP)",
        "params": {"channels": 32, "kernel": 5},
    },
    "cnn_simple_k7": {
        "display_name": "Simple CNN (1x Conv k=7 + GAP)",
        "params": {"channels": 32, "kernel": 7},
    },
    "cnn_simple_k9": {
        "display_name": "Simple CNN (1x Conv k=9 + GAP)",
        "params": {"channels": 32, "kernel": 9},
    },
    "cnn_simple_k11": {
        "display_name": "Simple CNN (1x Conv k=11 + GAP)",
        "params": {"channels": 32, "kernel": 11},
    },
    "cnn_simple_k13": {
        "display_name": "Simple CNN (1x Conv k=13 + GAP)",
        "params": {"channels": 32, "kernel": 13},
    },
    "cnn_simple_k15": {
        "display_name": "Simple CNN (1x Conv k=15 + GAP)",
        "params": {"channels": 32, "kernel": 15},
    },
    "cnn_simple_k17": {
        "display_name": "Simple CNN (1x Conv k=17 + GAP)",
        "params": {"channels": 32, "kernel": 17},
    },
    "cnn_simple_k19": {
        "display_name": "Simple CNN (1x Conv k=19 + GAP)",
        "params": {"channels": 32, "kernel": 19},
    },
    "cnn_simple_k21": {
        "display_name": "Simple CNN (1x Conv k=21 + GAP)",
        "params": {"channels": 32, "kernel": 21},
    },
    "cnn_simple_k23": {
        "display_name": "Simple CNN (1x Conv k=23 + GAP)",
        "params": {"channels": 32, "kernel": 23},
    },
    "cnn_simple_k25": {
        "display_name": "Simple CNN (1x Conv k=25 + GAP)",
        "params": {"channels": 32, "kernel": 25},
    },
    "cnn_simple_k27": {
        "display_name": "Simple CNN (1x Conv k=27 + GAP)",
        "params": {"channels": 32, "kernel": 27},
    },
    "cnn_simple_k29": {
        "display_name": "Simple CNN (1x Conv k=29 + GAP)",
        "params": {"channels": 32, "kernel": 29},
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
    """One Conv1d over L=29 + ReLU + naive mean + linear.

    Per peptide: 29 D-vectors in, 1 logit out. Kernel k=3 looks at
    3 neighbour slots at a time, same weights everywhere.

    Naive GAP: plain mean over all 29 slots, no pad mask. The filler
    (padtoken / BOS-EOS vectors) stays in the average. Justified
    because only ~3.5% rows have any pads, and a masked vs naive
    comparison showed identical performance.
    """

    def __init__(self, in_channels, channels=32, kernel=3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, channels, kernel,
                              padding=kernel // 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        # x: (N, D, L)
        h = self.relu(self.conv(x))  # (N, C, L)
        h = h.mean(-1)  # naive mean over all slots
        return self.fc(h).squeeze(-1)


class TwoLayerCNN(nn.Module):
    """2x Conv1d k=3 + ReLU, receptive field 5, then naive mean + linear.

    Same naive GAP as SimpleCNN. Layer 2 can compare a
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

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = h.mean(-1)
        return self.fc(h).squeeze(-1)


def build_cnn(model_key, in_channels, window=29, channels_override=None):
    """Build by registry key; channels_override (channel sweep) beats registry C."""
    cfg = get_cnn_config(model_key)["params"]
    channels = cfg["channels"] if channels_override is None else int(channels_override)
    if model_key == "cnn_2layer":
        return TwoLayerCNN(in_channels, channels, cfg["kernel"])
    return SimpleCNN(in_channels, channels, cfg["kernel"])
