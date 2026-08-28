from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dinids.constants import ATTACK_COLUMN, FEATURE_COLUMNS, LABEL_COLUMN


@pytest.fixture
def netflow_frame() -> pd.DataFrame:
    rows = 40
    values: dict[str, object] = {}
    for index, column in enumerate(FEATURE_COLUMNS):
        values[column] = np.linspace(index, index + 1, rows)
    labels = np.array(([0] * 16 + [1] * 4) * 2, dtype=np.int8)
    values[ATTACK_COLUMN] = np.where(labels == 0, "Benign", "Attack")
    values[LABEL_COLUMN] = labels
    return pd.DataFrame(values)
