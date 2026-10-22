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
    """

    def __init__(self, in_channels, channels=32, kernel=3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, channels, kernel,
                              padding=kernel // 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(channels, 1)

    def forward(self, x, mask=None):
        # x: (N, D, L); mask: (N, L) True=PAD
        if mask is not None:
            real = (~mask).float().unsqueeze(1)  # (N, 1, L)
            x = x * real
        h = self.relu(self.conv(x))  # (N, C, L)
        if mask is not None:
            h = h * real
            denom = real.sum(-1).clamp(min=1.0)  # (N, 1)
            h = h.sum(-1) / denom  # masked mean over real slots
        else:
            h = h.mean(-1)
        return self.fc(h).squeeze(-1)


def build_cnn(model_key, in_channels, window=29):
    cfg = get_cnn_config(model_key)["params"]
    return SimpleCNN(in_channels, cfg["channels"], cfg["kernel"])
