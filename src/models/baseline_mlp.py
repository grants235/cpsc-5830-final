"""
Baselines B1 (Random Forest) and B2 (MLP) for edge/flow classification.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier


class RandomForestBaseline:
    """B1 — Random Forest on Tier-A shared features."""

    def __init__(self, n_estimators=200, max_depth=20, n_jobs=-1, random_state=0):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            class_weight="balanced",
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class MLP(nn.Module):
    """B2 — MLP on quantile-normalized features."""

    def __init__(self, in_features: int, hidden: int = 128, num_classes: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class EnsembleMLP(nn.Module):
    """
    Per-training-dataset MLP encoders; outputs are mean-pooled for test.
    Each encoder projects its own feature space to a shared hidden dim,
    then a shared classifier head produces logits.
    """

    def __init__(self, feat_dims: list, hidden: int = 128, num_classes: int = 2):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden), nn.ReLU(), nn.LayerNorm(hidden)
            )
            for d in feat_dims
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward_single(self, x, encoder_idx: int):
        h = self.encoders[encoder_idx](x)
        return self.classifier(h)

    def forward_ensemble(self, xs: list):
        """xs[i] is the feature tensor for encoder i."""
        hs = [enc(x) for enc, x in zip(self.encoders, xs)]
        h = torch.stack(hs, dim=0).mean(dim=0)
        return self.classifier(h)
