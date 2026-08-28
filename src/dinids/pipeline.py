"""DI-NIDS feature extraction and One-Class SVM evaluation pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader, TensorDataset

from dinids.data import EvaluationData
from dinids.metrics import one_class_metrics, to_one_class_labels
from dinids.models import FeatureExtractor, load_encoder


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve auto, cpu, cuda, cuda:N, or mps to a usable torch device."""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return device


def evaluate_pair(
    data: EvaluationData,
    *,
    source_name: str,
    target_name: str,
    encoder_path: str | Path | None = None,
    device: torch.device | None = None,
    feature_batch_size: int = 8192,
    nu: float = 0.001,
    kernel: str = "poly",
    preprocessing: str = "legacy-independent",
    max_benign_train_rows: int | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """Fit OSVM on benign source flows and evaluate source and target domains."""

    if not 0 < nu <= 1:
        raise ValueError("nu must be in (0, 1]")
    if feature_batch_size < 1:
        raise ValueError("feature_batch_size must be positive")

    selected_device = device or resolve_device()
    encoder: FeatureExtractor | None = None
    checkpoint: dict[str, str] | None = None
    if encoder_path is not None:
        encoder = load_encoder(encoder_path, selected_device)
        resolved_checkpoint = Path(encoder_path).expanduser().resolve()
        checkpoint = {
            "file": resolved_checkpoint.name,
            "sha256": _sha256(resolved_checkpoint),
        }

    source_train_x = _project(encoder, data.source.x_train, selected_device, feature_batch_size)
    source_test_x = _project(encoder, data.source.x_test, selected_device, feature_batch_size)
    target_x = _project(encoder, data.target_x, selected_device, feature_batch_size)

    benign_mask = data.source.y_train == 0
    if not np.any(benign_mask):
        raise ValueError("The source training split contains no benign rows")
    benign_source = source_train_x[benign_mask]
    available_benign_rows = len(benign_source)
    if max_benign_train_rows is not None:
        if max_benign_train_rows < 1:
            raise ValueError("max_benign_train_rows must be positive")
        if max_benign_train_rows < available_benign_rows:
            generator = np.random.default_rng(seed)
            selected = generator.choice(
                available_benign_rows,
                size=max_benign_train_rows,
                replace=False,
            )
            benign_source = benign_source[selected]

    detector = OneClassSVM(nu=nu, kernel=kernel)
    detector.fit(benign_source)

    source_expected = to_one_class_labels(data.source.y_test)
    source_predicted = detector.predict(source_test_x)
    source_scores = detector.decision_function(source_test_x)
    target_expected = to_one_class_labels(data.target_y)
    target_predicted = detector.predict(target_x)
    target_scores = detector.decision_function(target_x)

    return {
        "method": "DI-NIDS" if encoder is not None else "OSVM",
        "source": source_name,
        "target": target_name,
        "preprocessing": preprocessing,
        "feature_count": int(source_train_x.shape[1]),
        "device": str(selected_device),
        "encoder": checkpoint,
        "osvm": {
            "nu": nu,
            "kernel": kernel,
            "available_benign_training_rows": int(available_benign_rows),
            "used_benign_training_rows": int(len(benign_source)),
        },
        "source_holdout": one_class_metrics(
            source_expected,
            source_predicted,
            decision_scores=source_scores,
        ),
        "target_complete": one_class_metrics(
            target_expected,
            target_predicted,
            decision_scores=target_scores,
        ),
        "row_counts": {
            "source_train": int(len(data.source.y_train)),
            "source_holdout": int(len(data.source.y_test)),
            "target_complete": int(len(data.target_y)),
        },
    }


def write_results(results: dict[str, object], destination: str | Path) -> Path:
    """Write evaluation results as deterministic, readable JSON."""

    output_path = Path(destination).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def _project(
    encoder: FeatureExtractor | None,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if encoder is None:
        return features
    loader = DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=batch_size)
    chunks: list[np.ndarray] = []
    encoder.eval()
    with torch.inference_mode():
        for (batch,) in loader:
            chunks.append(encoder(batch.to(device)).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
