from __future__ import annotations

import logging
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import ParameterSampler, StratifiedKFold

from .config import Config
from .data_loader import load_project_data
from .evaluation import select_offer_indices
from .feature_selection import ProfitDrivenFeatureSelector
from .utils import calibrate, cv_splits, find_best_threshold, write_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

cfg = Config()

# Platt calibration adds computational cost; enable with CSM_TUNE_WITH_CALIBRATION=1.
_TUNE_WITH_CALIBRATION = os.getenv("CSM_TUNE_WITH_CALIBRATION", "0") == "1"

HGB_PARAM_SPACE = {
    "learning_rate":     np.linspace(0.02, 0.20, 10),
    "max_leaf_nodes":    [15, 31, 63, 127],
    "min_samples_leaf":  [5, 10, 20, 40, 80],
    "max_iter":          [120, 180, 240, 300],
    "l2_regularization": [0.0, 0.1, 0.5, 1.0],
}


def _evaluate_hgb(
    X: object,
    y: object,
    params: dict,
    num_vars: int,
) -> tuple[float, float]:
    """Return ``(threshold, profit)`` for one HGB config, optionally calibrated."""
    folds = StratifiedKFold(
        n_splits=cv_splits(y), shuffle=True, random_state=cfg.random_state
    )
    oof = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in folds.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        base = HistGradientBoostingClassifier(random_state=cfg.random_state, **params)
        inner_folds = min(3, cv_splits(y_tr))
        model = (
            calibrate(base, inner_folds)
            if _TUNE_WITH_CALIBRATION and inner_folds >= 2
            else base
        )
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]

    return find_best_threshold(y, oof, num_vars, cfg.threshold_grid, cfg.max_offers)


def main() -> None:
    """Calibrated HGB pipeline: feature selection → param search → threshold search."""
    logger.info(
        "threshold-search pipeline | fast=%s | param_iter=%d | thr_grid=%d | calibration=%s",
        cfg.fast_mode, cfg.param_search_iter, len(cfg.threshold_grid), _TUNE_WITH_CALIBRATION,
    )

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    X_train, y_train, X_test = load_project_data(data_dir=data_dir)
    y = y_train.iloc[:, 0]

    selector = ProfitDrivenFeatureSelector(**cfg.selector_kwargs)
    selector.fit(X_train, y)
    features = selector.selected_features_
    if not features:
        logger.warning("No profitable features selected. Exiting.")
        return

    X_tr = X_train[features]
    X_te = X_test[features]

    candidates = list(
        ParameterSampler(HGB_PARAM_SPACE, n_iter=cfg.param_search_iter, random_state=cfg.random_state)
    )
    best_params, best_threshold, best_profit = None, None, -np.inf
    for i, params in enumerate(candidates, start=1):
        threshold, profit = _evaluate_hgb(X_tr, y, params, num_vars=len(features))
        logger.info(
            "[%02d/%02d] profit=%0.0f | threshold=%0.3f | params=%s",
            i, len(candidates), profit, threshold, params,
        )
        if profit > best_profit:
            best_profit, best_threshold, best_params = profit, threshold, params

    logger.info("Best: profit=%0.0f | threshold=%0.3f", best_profit, best_threshold)

    final = calibrate(
        HistGradientBoostingClassifier(random_state=cfg.random_state, **best_params),
        n_folds=cv_splits(y),
    )
    final.fit(X_tr, y)
    top_indices = select_offer_indices(
        final.predict_proba(X_te)[:, 1], threshold=best_threshold, max_offers=cfg.max_offers,
    )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    write_submission(top_indices + 1, features, project_root, cfg.team_name, suffix="threshold")


if __name__ == "__main__":
    main()
