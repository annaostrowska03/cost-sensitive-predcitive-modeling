"""Random Forest-only pipeline experiment.

Runs the 4-stage feature selection, then tunes Random Forest
hyperparameters via randomised search scored by OOF CV profit.
"""
from __future__ import annotations

import logging
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler

from ..config import Config
from ..feature_selection import ProfitDrivenFeatureSelector
from ..utils import (
    find_best_threshold,
    load_project_data,
    oof_predict_proba,
    select_offer_indices,
    write_submission,
)

logger = logging.getLogger(__name__)

cfg = Config()

RF_PARAM_SPACE = {
    "n_estimators":      [200, 300, 500, 700],
    "max_depth":         [None, 4, 6, 8, 12],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8, 12],
    "max_features":      ["sqrt", None, 0.5, 0.75],
    "class_weight":      [None, "balanced", "balanced_subsample"],
}


def main() -> None:
    """Random Forest-only pipeline with hyperparameter and threshold search."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info(
        "RF-only pipeline | fast=%s | param_iter=%d | thr_grid=%d",
        cfg.fast_mode, cfg.param_search_iter, len(cfg.threshold_grid),
    )

    X_train, y_train, X_test = load_project_data()
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
        ParameterSampler(RF_PARAM_SPACE, n_iter=cfg.param_search_iter, random_state=cfg.random_state)
    )
    best_params, best_threshold, best_profit = None, None, -np.inf

    for i, params in enumerate(candidates, start=1):
        oof = oof_predict_proba(
            lambda p=params: RandomForestClassifier(
                random_state=cfg.random_state, n_jobs=-1, **p
            ),
            X_tr, y,
        )
        threshold, profit = find_best_threshold(
            y, oof, len(features), cfg.threshold_grid, cfg.max_offers
        )
        logger.info(
            "[%02d/%02d] profit=%0.0f | threshold=%0.3f | params=%s",
            i, len(candidates), profit, threshold, params,
        )
        if profit > best_profit:
            best_profit, best_threshold, best_params = profit, threshold, params

    logger.info("Best RF: profit=%0.0f | threshold=%0.3f", best_profit, best_threshold)

    final = RandomForestClassifier(
        random_state=cfg.random_state, n_jobs=-1, **best_params
    )
    final.fit(X_tr, y)
    top_indices = select_offer_indices(
        final.predict_proba(X_te)[:, 1], threshold=best_threshold, max_offers=cfg.max_offers,
    )

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    write_submission(top_indices + 1, features, project_root, cfg.team_name, suffix="rf_only")


if __name__ == "__main__":
    main()
