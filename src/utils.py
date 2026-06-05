"""Shared helpers used across all pipeline scripts.

Each function here existed as a verbatim duplicate in multiple run_*.py files.
Centralising them here enforces a single source of truth.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import calculate_profit

logger = logging.getLogger(__name__)


def cv_splits(y: pd.Series, upper_bound: int = 5) -> int:
    """Return a valid number of stratified CV folds for target vector *y*."""
    n = min(upper_bound, int(y.value_counts().min()))
    if n < 2:
        raise ValueError("At least 2 samples per class are required for CV.")
    return n


def oof_predict_proba(
    model_factory: callable,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> np.ndarray:
    """Out-of-fold class-1 probabilities via stratified k-fold CV.

    model_factory must be a zero-argument callable that returns a fresh,
    unfitted binary classifier on every call.
    """
    folds = StratifiedKFold(n_splits=cv_splits(y), shuffle=True, random_state=random_state)
    oof = np.zeros(len(y), dtype=float)
    for train_idx, val_idx in folds.split(X, y):
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return oof


def find_best_threshold(
    y: pd.Series,
    y_pred_proba: npt.ArrayLike,
    num_vars: int,
    threshold_grid: npt.ArrayLike,
    max_offers: int = 1000,
) -> tuple[float, float]:
    """Return (threshold, profit) that maximise campaign profit on *threshold_grid*."""
    best_t, best_p = float(threshold_grid[0]), -np.inf  # type: ignore[index]
    for t in threshold_grid:
        p = calculate_profit(y, y_pred_proba, threshold=float(t), num_vars=num_vars, max_offers=max_offers)
        if p > best_p:
            best_p, best_t = p, float(t)
    return best_t, best_p


def logistic_pipeline(
    C: float = 0.2,
    class_weight: str | None = "balanced",
    max_iter: int = 3000,
    random_state: int = 42,
) -> Pipeline:
    """StandardScaler + L2-regularised LogisticRegression as a single Pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=C, max_iter=max_iter,
            class_weight=class_weight, solver="lbfgs", random_state=random_state,
        )),
    ])


def calibrate(estimator: object, n_folds: int) -> CalibratedClassifierCV:
    """Wrap estimator with Platt (sigmoid) probability calibration."""
    return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=n_folds)


def write_submission(
    obs_indices: npt.ArrayLike,
    var_names: list[str],
    project_root: str,
    team_name: str,
    suffix: str = "",
) -> None:
    """Write ``*_obs.txt`` and ``*_vars.txt`` submission files to *project_root*.

    Parameters
    ----------
    obs_indices:
        1-based customer indices selected for the campaign.
    var_names:
        Feature names of the form ``'V<n>'`` (e.g. ``['V11', 'V160']``).
    suffix:
        Tag appended between *team_name* and the extension
        (e.g. ``'ensemble'`` -> ``'ids_ensemble_obs.txt'``).
    """
    raw_vars = [int(v.replace("V", "")) for v in var_names]
    stem = f"{team_name}_{suffix}" if suffix else team_name
    obs_path = os.path.join(project_root, f"{stem}_obs.txt")
    vars_path = os.path.join(project_root, f"{stem}_vars.txt")
    pd.Series(obs_indices).to_csv(obs_path, index=False, header=False)
    pd.DataFrame(raw_vars).to_csv(vars_path, index=False, header=False)
    logger.info("Observations : %s (%d customers)", obs_path, len(obs_indices))
    logger.info("Variables    : %s (%d features)", vars_path, len(raw_vars))
