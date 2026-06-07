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


def select_offer_indices(
    y_pred_proba: npt.ArrayLike,
    threshold: float = 0.333,
    max_offers: int = 1000,
) -> np.ndarray:
    """Select campaign recipients using business-aware ranking.

    Keeps only probabilities above threshold, ranks them descending,
    and returns at most max_offers indices.
    """
    scores = np.array(y_pred_proba).ravel()

    if max_offers <= 0:
        return np.array([], dtype=int)

    eligible = np.where(scores > threshold)[0]
    if len(eligible) == 0:
        return np.array([], dtype=int)

    ranked = eligible[np.argsort(scores[eligible])[::-1]]
    return ranked[:max_offers]


def calculate_profit(
    y_true: npt.ArrayLike,
    y_pred_proba: npt.ArrayLike,
    threshold: float = 0.333,
    num_vars: int = 0,
    max_offers: int = 1000,
    feature_cost: int = 200,
    debug: bool = False,
) -> float:
    """Calculate campaign profit (EUR) under the scoring formula.

    ``Profit = (TP * 10) - (FP * 5) - (num_vars * feature_cost)``

    Parameters
    ----------
    y_true:
        True binary labels.
    y_pred_proba:
        Model-predicted class-1 probabilities.
    threshold:
        Minimum probability to be considered for an offer.
    num_vars:
        Number of purchased variables (each costs *feature_cost* EUR).
    max_offers:
        Hard cap on offers sent.
    feature_cost:
        EUR penalty per purchased variable (default 200).
    debug:
        When ``True``, logs a breakdown of the profit components.
    """
    labels = np.array(y_true).ravel()
    scores = np.array(y_pred_proba).ravel()

    selected = select_offer_indices(scores, threshold=threshold, max_offers=max_offers)

    tp = int(np.sum(labels[selected] == 1)) if len(selected) else 0
    fp = int(np.sum(labels[selected] == 0)) if len(selected) else 0

    customer_profit = tp * 10 - fp * 5
    variable_cost = num_vars * feature_cost
    total = customer_profit - variable_cost

    if debug:
        logger.info("Selected customers: %d", len(selected))
        logger.info("Hits (TP): %d  (+%d EUR)", tp, tp * 10)
        logger.info("Misses (FP): %d  (-%d EUR)", fp, fp * 5)
        logger.info("Variable cost (%d vars): -%d EUR", num_vars, variable_cost)
        logger.info("FINAL SCORE: %d EUR", total)

    return float(total)




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
    margin: float = 0.0,
) -> tuple[float, float]:
    """Return ``(threshold, profit)`` that maximise campaign profit on *threshold_grid*.

    *margin* shifts the selected threshold upward (e.g. 0.02) to reduce the
    false-positive rate on unseen data at the cost of a slightly lower TP count.
    """
    best_t, best_p = float(threshold_grid[0]), -np.inf  # type: ignore[index]
    for t in threshold_grid:
        p = calculate_profit(y, y_pred_proba, threshold=float(t), num_vars=num_vars, max_offers=max_offers)
        if p > best_p:
            best_p, best_t = p, float(t)
    if margin:
        best_t = float(np.clip(best_t + margin, 0.0, 1.0))
        best_p = calculate_profit(y, y_pred_proba, threshold=best_t, num_vars=num_vars, max_offers=max_offers)
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


def calibrate(estimator: object, n_folds: int | None = None) -> CalibratedClassifierCV:
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
