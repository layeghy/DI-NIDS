"""Training loops for the source classifier and domain-adversarial network."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from dinids.data import DomainSplit, TrainingData
from dinids.models import DomainClassifier, FeatureExtractor, LabelClassifier


@dataclass(frozen=True)
class TrainConfig:
    source_epochs: int = 10
    dann_epochs: int = 10
    batch_size: int = 1024
    learning_rate: float = 0.001
    momentum: float = 0.9
    seed: int = 0
    num_workers: int = 0
    optimisation: str = "legacy"
    shuffle_batches: bool = False


def train_dann_models(
    data: TrainingData,
    *,
    output_dir: str | Path,
    device: torch.device,
    config: TrainConfig,
) -> dict[str, object]:
    """Run source pretraining followed by DANN training and save all checkpoints."""

    _validate_config(config)
    _set_seed(config.seed)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    extractor = FeatureExtractor().to(device)
    classifier = LabelClassifier().to(device)
    discriminator = DomainClassifier().to(device)

    source_train = _labelled_loader(
        data.source.x_train,
        data.source.y_train,
        batch_size=config.batch_size,
        shuffle=config.shuffle_batches,
        workers=config.num_workers,
        seed=config.seed,
    )
    target_train = _labelled_loader(
        data.target.x_train,
        data.target.y_train,
        batch_size=config.batch_size,
        shuffle=config.shuffle_batches,
        workers=config.num_workers,
        seed=config.seed + 1,
    )

    source_history = _train_source_only(
        extractor,
        classifier,
        source_train,
        device=device,
        config=config,
    )
    source_metrics = _evaluate_label_classifier(
        extractor,
        classifier,
        data.source,
        device,
        batch_size=config.batch_size,
    )
    torch.save(extractor.state_dict(), destination / "encoder_source_only.pt")
    torch.save(classifier.state_dict(), destination / "classifier_source_only.pt")

    dann_history = _train_dann(
        extractor,
        classifier,
        discriminator,
        source_train,
        target_train,
        device=device,
        config=config,
    )
    source_after_dann = _evaluate_label_classifier(
        extractor,
        classifier,
        data.source,
        device,
        batch_size=config.batch_size,
    )
    target_after_dann = _evaluate_label_classifier(
        extractor,
        classifier,
        data.target,
        device,
        batch_size=config.batch_size,
    )

    torch.save(extractor.state_dict(), destination / "encoder_dann.pt")
    torch.save(classifier.state_dict(), destination / "classifier_dann.pt")
    torch.save(discriminator.state_dict(), destination / "discriminator_dann.pt")

    summary: dict[str, object] = {
        "config": asdict(config),
        "device": str(device),
        "source_only": {
            "loss": source_history,
            "source_holdout": source_metrics,
        },
        "dann": {
            "loss": dann_history,
            "source_holdout": source_after_dann,
            "target_holdout": target_after_dann,
        },
        "checkpoints": {
            "source_encoder": "encoder_source_only.pt",
            "source_classifier": "classifier_source_only.pt",
            "dann_encoder": "encoder_dann.pt",
            "dann_classifier": "classifier_dann.pt",
            "dann_discriminator": "discriminator_dann.pt",
        },
    }
    with (destination / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def _train_source_only(
    extractor: FeatureExtractor,
    classifier: LabelClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    config: TrainConfig,
) -> list[float]:
    criterion = nn.BCEWithLogitsLoss()
    optimiser = SGD(
        [*extractor.parameters(), *classifier.parameters()],
        lr=config.learning_rate,
        momentum=config.momentum,
    )
    total_steps = max(1, config.source_epochs * len(loader))
    global_step = 0
    history: list[float] = []

    for _ in range(config.source_epochs):
        extractor.train()
        classifier.train()
        losses: list[float] = []
        for features, labels in loader:
            progress = global_step / total_steps
            if config.optimisation in {"legacy", "scheduled"}:
                _set_learning_rate(optimiser, config.learning_rate, progress)
            features = features.to(device)
            labels = labels.to(device)
            optimiser.zero_grad(set_to_none=True)
            logits = classifier(extractor(features)).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
            global_step += 1
        history.append(float(np.mean(losses)))
    return history


def _train_dann(
    extractor: FeatureExtractor,
    classifier: LabelClassifier,
    discriminator: DomainClassifier,
    source_loader: DataLoader,
    target_loader: DataLoader,
    *,
    device: torch.device,
    config: TrainConfig,
) -> list[dict[str, float]]:
    criterion = nn.BCEWithLogitsLoss()
    optimiser = SGD(
        [*extractor.parameters(), *classifier.parameters(), *discriminator.parameters()],
        lr=config.learning_rate,
        momentum=config.momentum,
    )
    steps_per_epoch = min(len(source_loader), len(target_loader))
    total_steps = max(1, config.dann_epochs * steps_per_epoch)
    global_step = 0
    history: list[dict[str, float]] = []

    for _ in range(config.dann_epochs):
        extractor.train()
        classifier.train()
        discriminator.train()
        class_losses: list[float] = []
        domain_losses: list[float] = []

        for (source_x, source_y), (target_x, _) in zip(source_loader, target_loader, strict=False):
            progress = global_step / total_steps
            alpha = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
            if config.optimisation == "scheduled":
                _set_learning_rate(optimiser, config.learning_rate, progress)

            source_x = source_x.to(device)
            source_y = source_y.to(device)
            target_x = target_x.to(device)
            combined_x = torch.cat((source_x, target_x), dim=0)
            domain_y = torch.cat(
                (
                    torch.zeros(source_x.shape[0], device=device),
                    torch.ones(target_x.shape[0], device=device),
                )
            )

            optimiser.zero_grad(set_to_none=True)
            source_logits = classifier(extractor(source_x)).squeeze(1)
            class_loss = criterion(source_logits, source_y)
            domain_logits = discriminator(extractor(combined_x), alpha).squeeze(1)
            domain_loss = criterion(domain_logits, domain_y)
            (class_loss + domain_loss).backward()
            optimiser.step()

            class_losses.append(float(class_loss.detach().cpu()))
            domain_losses.append(float(domain_loss.detach().cpu()))
            global_step += 1

        history.append(
            {
                "class_loss": float(np.mean(class_losses)),
                "domain_loss": float(np.mean(domain_losses)),
            }
        )
    return history


def _evaluate_label_classifier(
    extractor: FeatureExtractor,
    classifier: LabelClassifier,
    split: DomainSplit,
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, float]:
    extractor.eval()
    classifier.eval()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(split.x_test),
            torch.from_numpy(split.y_test),
        ),
        batch_size=batch_size,
    )
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    with torch.inference_mode():
        for features, expected in loader:
            expected = expected.to(device)
            probabilities = torch.sigmoid(classifier(extractor(features.to(device))).squeeze(1))
            predicted = (probabilities >= 0.5).to(torch.int8)
            correct += int((predicted == expected).sum().cpu())
            total += int(expected.numel())
            true_positive += int(((predicted == 1) & (expected == 1)).sum().cpu())
            false_positive += int(((predicted == 1) & (expected == 0)).sum().cpu())
            false_negative += int(((predicted == 0) & (expected == 1)).sum().cpu())

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": correct / total if total else 0.0,
        "attack_precision": float(precision),
        "attack_recall": float(recall),
        "attack_f1": float(f1),
    }


def _labelled_loader(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(labels.astype(np.float32, copy=False)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        generator=generator,
    )


def _set_learning_rate(optimiser: SGD, base_rate: float, progress: float) -> None:
    rate = base_rate / (1.0 + 10.0 * progress) ** 0.75
    for group in optimiser.param_groups:
        group["lr"] = rate


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_config(config: TrainConfig) -> None:
    if config.source_epochs < 1 or config.dann_epochs < 1:
        raise ValueError("Both epoch counts must be positive")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if config.optimisation not in {"legacy", "scheduled", "constant"}:
        raise ValueError("optimisation must be legacy, scheduled, or constant")
