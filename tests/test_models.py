from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dinids.models import (  # noqa: E402
    DomainClassifier,
    FeatureExtractor,
    LabelClassifier,
    load_encoder,
)


def test_model_shapes_match_supplied_checkpoints() -> None:
    batch = torch.zeros((8, 39))
    extractor = FeatureExtractor()
    classifier = LabelClassifier()
    discriminator = DomainClassifier()
    features = extractor(batch)
    assert features.shape == (8, 10)
    assert classifier(features).shape == (8, 1)
    assert discriminator(features, alpha=1.0).shape == (8, 1)


def test_supplied_checkpoints_match_reconstructed_architecture() -> None:
    checkpoint_dir = Path(__file__).parents[1] / "models" / "original_checkpoints"
    device = torch.device("cpu")
    for source in ("cic2018", "unsw_nb15"):
        encoder = load_encoder(checkpoint_dir / f"encoder_source_{source}.pt", device)
        assert encoder(torch.zeros((2, 39))).shape == (2, 10)

        classifier = LabelClassifier()
        state = torch.load(
            checkpoint_dir / f"classifier_source_{source}.pt",
            map_location=device,
            weights_only=True,
        )
        classifier.load_state_dict(state, strict=True)
        assert classifier(torch.zeros((2, 10))).shape == (2, 1)
