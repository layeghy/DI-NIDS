"""PyTorch networks used by the supplied DI-NIDS implementation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.autograd import Function

from dinids.constants import FEATURE_COLUMNS


class GradientReversal(Function):
    """Identity in the forward pass and sign-reversing in the backward pass."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient.neg() * ctx.alpha, None


class FeatureExtractor(nn.Module):
    """Map the 39 non-identifier NetFlow-v2 fields to ten features.

    The layer layout intentionally matches the checkpoints in the original archive.
    """

    def __init__(self, input_features: int = len(FEATURE_COLUMNS), output_features: int = 10):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(input_features, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(10, output_features),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.extractor(inputs)


class LabelClassifier(nn.Module):
    """Binary source-label classifier."""

    def __init__(self, input_features: int = 10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs)


class DomainClassifier(nn.Module):
    """Binary source/target domain classifier with gradient reversal."""

    def __init__(self, input_features: int = 10):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_features, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(10, 1),
        )

    def forward(self, inputs: torch.Tensor, alpha: float) -> torch.Tensor:
        return self.discriminator(GradientReversal.apply(inputs, alpha))


def load_encoder(path: str | Path, device: torch.device) -> FeatureExtractor:
    """Load a state-dictionary checkpoint without permitting arbitrary objects."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("Expected a PyTorch state dictionary")
    model = FeatureExtractor().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
