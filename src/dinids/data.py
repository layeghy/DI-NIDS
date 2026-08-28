"""Loading, validation, splitting and scaling for NetFlow-v2 datasets."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from dinids.constants import ATTACK_COLUMN, FEATURE_COLUMNS, LABEL_COLUMN


@dataclass(frozen=True)
class DomainSplit:
    """A train/test split after preprocessing."""

    x_train: np.ndarray
    y_train: np.ndarray
    attacks_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    attacks_test: np.ndarray


@dataclass(frozen=True)
class EvaluationData:
    """Prepared source training/holdout data and a complete target domain."""

    source: DomainSplit
    target_x: np.ndarray
    target_y: np.ndarray
    target_attacks: np.ndarray
    source_scaler: MinMaxScaler
    target_scaler: MinMaxScaler | None


@dataclass(frozen=True)
class TrainingData:
    """Prepared source and target splits for DANN training."""

    source: DomainSplit
    target: DomainSplit
    source_scaler: MinMaxScaler
    target_scaler: MinMaxScaler | None


def read_dataframe(path: str | Path, *, allow_unsafe_pickle: bool = False) -> pd.DataFrame:
    """Read a supported tabular format.

    Pickle is disabled by default because loading an untrusted pickle can execute code.
    """

    data_path = Path(path).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    name = data_path.name.lower()
    if name.endswith((".csv", ".csv.gz", ".csv.bz2", ".csv.xz")):
        frame = pd.read_csv(data_path, low_memory=False)
    elif name.endswith((".parquet", ".pq")):
        frame = pd.read_parquet(data_path)
    elif name.endswith((".pkl", ".pickle")):
        if not allow_unsafe_pickle:
            raise ValueError(
                "Pickle loading is disabled. Use --allow-unsafe-pickle only for a file you trust."
            )
        frame = pd.read_pickle(data_path)  # noqa: S301 - explicitly gated above
    else:
        raise ValueError(
            f"Unsupported dataset format for {data_path.name}. "
            "Use CSV, compressed CSV, Parquet, or a trusted pickle."
        )

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame in {data_path.name}")
    return frame


def clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and return finite numeric model inputs."""

    required = set(FEATURE_COLUMNS) | {ATTACK_COLUMN, LABEL_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(f"Dataset is missing {len(missing)} required columns: {preview}{suffix}")

    cleaned = frame.loc[:, [*FEATURE_COLUMNS, ATTACK_COLUMN, LABEL_COLUMN]].copy()
    for column in FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned[LABEL_COLUMN] = pd.to_numeric(cleaned[LABEL_COLUMN], errors="coerce")
    cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)

    before = len(cleaned)
    cleaned.dropna(subset=[*FEATURE_COLUMNS, ATTACK_COLUMN, LABEL_COLUMN], inplace=True)
    dropped = before - len(cleaned)
    if dropped:
        warnings.warn(
            f"Dropped {dropped:,} rows containing missing or non-finite values.",
            stacklevel=2,
        )

    cleaned[LABEL_COLUMN] = cleaned[LABEL_COLUMN].astype(np.int8)
    observed_labels = set(cleaned[LABEL_COLUMN].unique().tolist())
    if not observed_labels.issubset({0, 1}):
        raise ValueError(
            f"Label must contain only 0 (benign) and 1 (attack), found {observed_labels}"
        )
    if cleaned.empty:
        raise ValueError("No usable rows remain after validation")

    cleaned[ATTACK_COLUMN] = cleaned[ATTACK_COLUMN].astype(str)
    cleaned.reset_index(drop=True, inplace=True)
    return cleaned


def load_domain(
    path: str | Path,
    *,
    allow_unsafe_pickle: bool = False,
    max_rows: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Load, clean and optionally stratify-sample a domain."""

    frame = clean_dataframe(read_dataframe(path, allow_unsafe_pickle=allow_unsafe_pickle))
    if max_rows is None or max_rows >= len(frame):
        return frame
    if max_rows < 2:
        raise ValueError("max_rows must be at least 2")

    stratify = _safe_stratify(frame)
    try:
        sampled, _ = train_test_split(
            frame,
            train_size=max_rows,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        sampled = frame.sample(n=max_rows, random_state=seed)
    return sampled.reset_index(drop=True)


def prepare_training_data(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    test_size: float = 0.3,
    seed: int = 0,
    preprocessing: str = "legacy-independent",
) -> TrainingData:
    """Prepare both domains for DANN training."""

    if preprocessing not in {"legacy-independent", "source-train"}:
        raise ValueError(f"Unknown preprocessing mode: {preprocessing}")
    source_train, source_test = _split_frame(source, test_size=test_size, seed=seed)
    target_train, target_test = _split_frame(
        target,
        test_size=test_size,
        seed=seed,
        stratify=preprocessing == "legacy-independent",
    )

    if preprocessing == "legacy-independent":
        source_scaler = MinMaxScaler().fit(source.loc[:, list(FEATURE_COLUMNS)])
        target_scaler = MinMaxScaler().fit(target.loc[:, list(FEATURE_COLUMNS)])
    elif preprocessing == "source-train":
        source_scaler = MinMaxScaler().fit(source_train.loc[:, list(FEATURE_COLUMNS)])
        target_scaler = None

    return TrainingData(
        source=_to_domain_split(source_train, source_test, source_scaler),
        target=_to_domain_split(target_train, target_test, target_scaler or source_scaler),
        source_scaler=source_scaler,
        target_scaler=target_scaler,
    )


def prepare_evaluation_data(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    test_size: float = 0.3,
    seed: int = 0,
    preprocessing: str = "legacy-independent",
) -> EvaluationData:
    """Prepare a source split and complete target domain for OSVM evaluation."""

    source_train, source_test = _split_frame(source, test_size=test_size, seed=seed)
    if preprocessing == "legacy-independent":
        source_scaler = MinMaxScaler().fit(source.loc[:, list(FEATURE_COLUMNS)])
        target_scaler = MinMaxScaler().fit(target.loc[:, list(FEATURE_COLUMNS)])
    elif preprocessing == "source-train":
        source_scaler = MinMaxScaler().fit(source_train.loc[:, list(FEATURE_COLUMNS)])
        target_scaler = None
    else:
        raise ValueError(f"Unknown preprocessing mode: {preprocessing}")

    target_features = (target_scaler or source_scaler).transform(
        target.loc[:, list(FEATURE_COLUMNS)]
    )
    return EvaluationData(
        source=_to_domain_split(source_train, source_test, source_scaler),
        target_x=np.asarray(target_features, dtype=np.float32),
        target_y=target[LABEL_COLUMN].to_numpy(dtype=np.int8),
        target_attacks=target[ATTACK_COLUMN].to_numpy(),
        source_scaler=source_scaler,
        target_scaler=target_scaler,
    )


def _to_domain_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    scaler: MinMaxScaler,
) -> DomainSplit:
    return DomainSplit(
        x_train=np.asarray(scaler.transform(train.loc[:, list(FEATURE_COLUMNS)]), dtype=np.float32),
        y_train=train[LABEL_COLUMN].to_numpy(dtype=np.int8),
        attacks_train=train[ATTACK_COLUMN].to_numpy(),
        x_test=np.asarray(scaler.transform(test.loc[:, list(FEATURE_COLUMNS)]), dtype=np.float32),
        y_test=test[LABEL_COLUMN].to_numpy(dtype=np.int8),
        attacks_test=test[ATTACK_COLUMN].to_numpy(),
    )


def _split_frame(
    frame: pd.DataFrame,
    *,
    test_size: float,
    seed: int,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=_safe_stratify(frame) if stratify else None,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _safe_stratify(frame: pd.DataFrame) -> pd.Series | None:
    attack_counts = frame[ATTACK_COLUMN].value_counts()
    if len(attack_counts) > 1 and attack_counts.min() >= 2:
        return frame[ATTACK_COLUMN]
    label_counts = frame[LABEL_COLUMN].value_counts()
    if len(label_counts) > 1 and label_counts.min() >= 2:
        return frame[LABEL_COLUMN]
    return None
