"""
05_cnn_models.py
Minimal 1D-CNN zoo for fixed-29 window embeddings. Mirrors MODEL_REGISTRY
style in config.py (display_name + params dict).

Input convention: (N, D, L=29) float32, mask (N, L) bool where True = PAD.
"""

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - import-time hint only
    raise ImportError("torch required: use env_esmc_server or GPU env")

CNN_REGISTRY = {
    "cnn_k3": {
        "display_name": "CNN k=3 (baseline)",
        "params": {"channels": 64, "depth": 2, "kernel": 3, "dropout": 0.2},
    },
    "cnn_k5": {
        "display_name": "CNN k=5",
        "params": {"channels": 64, "depth": 2, "kernel": 5, "dropout": 0.2},
    },
    "cnn_k7": {
        "display_name": "CNN k=7",
        "params": {"channels": 64, "depth": 2, "kernel": 7, "dropout": 0.2},
    },
    "cnn_deep": {
        "display_name": "CNN deep (3 layers)",
        "params": {"channels": 64, "depth": 3, "kernel": 3, "dropout": 0.2},
    },
    "mlp_flat": {
        "display_name": "MLP flat control (no conv)",
        "params": {"hidden": 256, "dropout": 0.2},
    },
}

DEFAULT_CNN_MODEL = "cnn_k3"


def get_cnn_config(model_key):
    if model_key not in CNN_REGISTRY:
        valid = ", ".join(sorted(CNN_REGISTRY.keys()))
        raise KeyError(f"Unknown CNN model '{model_key}'. Valid: {valid}")
    return CNN_REGISTRY[model_key]


class SeqCNN(nn.Module):
    """Conv1d over L=29, masked GAP + linear. Pads re-zeroed each layer."""

    def __init__(self, in_channels, channels=64, depth=2, kernel=3, dropout=0.2):
        super().__init__()
        pad = kernel // 2
        layers = []
        c_in = in_channels
        for _ in range(depth):
            layers += [nn.Conv1d(c_in, channels, kernel, padding=pad),
                       nn.ReLU(), nn.Dropout(dropout)]
            c_in = channels
        self.convs = nn.Sequential(*layers)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x, mask=None):
        # x: (N, D, L); mask: (N, L) True=PAD
        if mask is not None:
            real = (~mask).float().unsqueeze(1)  # (N, 1, L)
            x = x * real
        h = self.convs(x)
        if mask is not None:
            h = h * real
            denom = real.sum(-1).clamp(min=1.0)  # (N, 1)
            h = h.sum(-1) / denom
        else:
            h = h.mean(-1)
        return self.fc(h).squeeze(-1)


class FlatMLP(nn.Module):
    """Flat control: same raw dims as 04 flat_raw, no convolution."""

    def __init__(self, in_channels, window=29, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * window, hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, mask=None):
        return self.net(x).squeeze(-1)


def build_cnn(model_key, in_channels, window=29):
    cfg = get_cnn_config(model_key)["params"]
    if model_key == "mlp_flat":
        return FlatMLP(in_channels, window, cfg["hidden"], cfg["dropout"])
    return SeqCNN(in_channels, cfg["channels"], cfg["depth"],
                  cfg["kernel"], cfg["dropout"])
