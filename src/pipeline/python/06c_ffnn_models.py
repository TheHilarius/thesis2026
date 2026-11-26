"""
06c_ffnn_models.py
Simplest FFNNs on flattened window embeddings. Mirrors 05c_cnn_models.py
registry style (display_name + params dict). Stage letters: a = modelling
(06a_ffnn_modelling.py), c = model zoo (this file), d = plotting (shared
05d_plotting.py).

Input convention: (N, L*D) float32 — each window (L, D) flattened
row-major (slot-major: slot 0 dims, slot 1 dims, ...). Same prepared H5
as the CNN (emb_esmc300m_win_padtoken, flank5_rmdup 19mer), so FFNN and
CNN see literally the same rows, only reshaped.

Baseline ffnn_simple is the twin of the course notebook SimpleFFNN:
Linear(in->w) -> ReLU -> Linear(w->1). Width sweep runs on the SAME key
via --width override (mirrors CNN --channels override), so no width
zoo entries are needed; fixed keys exist only for structural variants
(2 hidden layers, dropout).
"""

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - import-time hint only
    raise ImportError("torch required: use env_esmc_server or GPU env")

FFNN_REGISTRY = {
    "ffnn_simple": {
        "display_name": "Simple FFNN (1x hidden, ReLU)",
        "params": {"width": 32, "layers": 1, "dropout": 0.0},
    },
    "ffnn_2layer": {
        "display_name": "FFNN 2-layer (2x hidden, ReLU)",
        "params": {"width": 32, "layers": 2, "dropout": 0.0},
    },
    "ffnn_simple_dropout05": {
        "display_name": "Simple FFNN (1x hidden, ReLU, dropout 0.5)",
        "params": {"width": 32, "layers": 1, "dropout": 0.5},
    },
}

DEFAULT_FFNN_MODEL = "ffnn_simple"


def get_ffnn_config(model_key):
    if model_key not in FFNN_REGISTRY:
        valid = ", ".join(sorted(FFNN_REGISTRY.keys()))
        raise KeyError(f"Unknown FFNN model '{model_key}'. Valid: {valid}")
    return FFNN_REGISTRY[model_key]


class SimpleFFNN(nn.Module):
    """Linear(in->w) -> ReLU -> Linear(w->1). Course-notebook twin.

    Per window: L*D flattened dims in, 1 logit out. Width w=32 default
    mirrors the CNN channel baseline (C=32) for a like-for-like
    parameter-count starting point.
    """

    def __init__(self, in_dim, width=32, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, width)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(width, 1)

    def forward(self, x):
        # x: (N, L*D)
        h = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(h).squeeze(-1)


class TwoLayerFFNN(nn.Module):
    """2x hidden: Linear(in->w) -> ReLU -> Linear(w->w) -> ReLU -> Linear(w->1).

    Depth counterpart of TwoLayerCNN: more params, no weight sharing,
    no translation equivariance — FFNN must learn anchor positions
    itself from the flattened slot order.
    """

    def __init__(self, in_dim, width=32, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout(self.relu(self.fc1(x)))
        h = self.dropout(self.relu(self.fc2(h)))
        return self.fc3(h).squeeze(-1)


def build_ffnn(model_key, in_dim, width_override=None):
    """Build by registry key; width_override (width sweep) beats registry w."""
    cfg = get_ffnn_config(model_key)["params"]
    width = cfg["width"] if width_override is None else int(width_override)
    layers = cfg.get("layers", 1)
    dropout = cfg.get("dropout", 0.0)
    if layers == 2:
        return TwoLayerFFNN(in_dim, width, dropout)
    return SimpleFFNN(in_dim, width, dropout)
