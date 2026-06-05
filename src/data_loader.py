from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def load_project_data(
    data_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load x_train.txt, y_train.txt, and x_test.txt from data_dir.

    Defaults to the data/ folder one level above this file when
    data_dir is None.  Returns (X_train, y_train, X_test).
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

    def _read(filename: str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(data_dir, filename), sep=r"\s+", engine="python")

    X_train = _read("x_train.txt")
    y_train = _read("y_train.txt")
    X_test = _read("x_test.txt")
    logger.info(
        "Data loaded — X_train: %s, y_train: %s, X_test: %s",
        X_train.shape, y_train.shape, X_test.shape,
    )
    return X_train, y_train, X_test
