from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dinids.constants import FEATURE_COLUMNS
from dinids.data import (
    clean_dataframe,
    prepare_evaluation_data,
    prepare_training_data,
    read_dataframe,
)


def test_clean_dataframe_rejects_missing_feature(netflow_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing 1 required columns"):
        clean_dataframe(netflow_frame.drop(columns=[FEATURE_COLUMNS[0]]))


def test_clean_dataframe_drops_non_finite_rows(netflow_frame: pd.DataFrame) -> None:
    frame = netflow_frame.copy()
    frame.loc[0, FEATURE_COLUMNS[0]] = np.inf
    with pytest.warns(UserWarning, match="Dropped 1 rows"):
        cleaned = clean_dataframe(frame)
    assert len(cleaned) == len(frame) - 1


def test_legacy_training_preparation_has_expected_shapes(netflow_frame: pd.DataFrame) -> None:
    source = clean_dataframe(netflow_frame)
    target = clean_dataframe(netflow_frame.sample(frac=1, random_state=4))
    prepared = prepare_training_data(
        source,
        target,
        test_size=0.25,
        seed=2,
        preprocessing="legacy-independent",
    )
    assert prepared.source.x_train.shape == (30, 39)
    assert prepared.source.x_test.shape == (10, 39)
    assert prepared.target.x_train.shape == (30, 39)
    assert prepared.target_scaler is not None


def test_source_train_evaluation_uses_one_scaler(netflow_frame: pd.DataFrame) -> None:
    source = clean_dataframe(netflow_frame)
    target = clean_dataframe(netflow_frame)
    prepared = prepare_evaluation_data(
        source,
        target,
        test_size=0.25,
        seed=2,
        preprocessing="source-train",
    )
    assert prepared.target_scaler is None
    assert prepared.target_x.shape == (40, 39)


def test_pickle_loading_requires_explicit_trust(
    tmp_path,
    netflow_frame: pd.DataFrame,
) -> None:
    path = tmp_path / "domain.pickle"
    netflow_frame.to_pickle(path)
    with pytest.raises(ValueError, match="Pickle loading is disabled"):
        read_dataframe(path)
    loaded = read_dataframe(path, allow_unsafe_pickle=True)
    assert loaded.shape == netflow_frame.shape
