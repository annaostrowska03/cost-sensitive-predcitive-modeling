from __future__ import annotations

import logging
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import ParameterSampler

from .config import Config
from .data_loader import load_project_data
from .evaluation import select_offer_indices
from .feature_selection import ProfitDrivenFeatureSelector
from .utils import (
    calibrate, cv_splits, find_best_threshold,
    logistic_pipeline, oof_predict_proba, write_submission,
)
from .visualization import plot_profit_curve

# Alias kept for notebook compatibility.
build_calibrated_model = calibrate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

cfg = Config()

HGB_PARAM_SPACE = {
    "learning_rate":     np.linspace(0.01, 0.18, 12),
    "max_leaf_nodes":    [7, 15, 31, 63, 127],
    "min_samples_leaf":  [5, 10, 20, 40, 80, 120],
    "max_iter":          [120, 180, 240, 320, 420],
    "l2_regularization": [0.0, 0.05, 0.1, 0.5, 1.0, 2.0],
    "max_depth":         [None, 3, 5, 7],
}

RF_CONFIG = {
    "n_estimators":      cfg.rf_estimators,
    "max_depth":         None,
    "min_samples_split": 10,
    "min_samples_leaf":  8,
    "max_features":      0.5,
    "class_weight":      None,
}

LR_C = 0.2


def tune_hgb_for_profit(
    X: object,
    y: object,
    num_vars: int,
    config: Config | None = None,
) -> tuple[dict, float, float]:
    """Randomised search over HGB hyperparameters scored by OOF CV profit.

    Returns ``(best_params, best_threshold, best_profit)``.
    """
    c = config or cfg
    candidates = list(ParameterSampler(
        HGB_PARAM_SPACE, n_iter=c.param_search_iter, random_state=c.random_state,
    ))

    best_params, best_threshold, best_profit = None, None, -np.inf

    for i, params in enumerate(candidates, start=1):
        oof = oof_predict_proba(
            lambda p=params: HistGradientBoostingClassifier(random_state=c.random_state, **p),
            X, y, random_state=c.random_state,
        )
        threshold, profit = find_best_threshold(y, oof, num_vars, c.threshold_grid, c.max_offers)
        logger.info(
            "[HGB %02d/%02d] CV profit=%0.0f | threshold=%0.3f | params=%s",
            i, len(candidates), profit, threshold, params,
        )
        if profit > best_profit:
            best_profit, best_threshold, best_params = profit, threshold, params

    logger.info(
        "Best HGB: CV profit=%0.0f | threshold=%0.3f | params=%s",
        best_profit, best_threshold, best_params,
    )
    return best_params, best_threshold, best_profit


def select_features(
    X_train: object,
    y: object,
    config: Config | None = None,
) -> ProfitDrivenFeatureSelector:
    """Run the four-stage selector and save the profit-curve plot."""
    c = config or cfg
    selector = ProfitDrivenFeatureSelector(**c.selector_kwargs)
    selector.fit(X_train, y)

    plot_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "docs", "profit_curve.png",
    )
    plot_profit_curve(selector, output_path=plot_path)
    return selector


def _evaluate_ensemble(
    X: object,
    y: object,
    best_hgb_params: dict,
    num_vars: int,
    config: Config | None = None,
) -> tuple[str, float, float]:
    """Compare HGB-only vs soft-voting ensemble using OOF profit."""
    c = config or cfg
    hgb_oof = oof_predict_proba(
        lambda: HistGradientBoostingClassifier(random_state=c.random_state, **best_hgb_params),
        X, y, random_state=c.random_state,
    )
    rf_oof = oof_predict_proba(
        lambda: RandomForestClassifier(random_state=c.random_state, n_jobs=-1, **RF_CONFIG),
        X, y, random_state=c.random_state,
    )
    lr_oof = oof_predict_proba(
        lambda: logistic_pipeline(C=LR_C, class_weight="balanced"),
        X, y, random_state=c.random_state,
    )

    hgb_t, hgb_p = find_best_threshold(y, hgb_oof, num_vars, c.threshold_grid, c.max_offers)

    w = c.ensemble_weights
    ens_oof = w["hgb"] * hgb_oof + w["rf"] * rf_oof + w["lr"] * lr_oof
    ens_t, ens_p = find_best_threshold(y, ens_oof, num_vars, c.threshold_grid, c.max_offers)

    logger.info("Strategy HGB-only  | CV profit=%0.0f | threshold=%0.3f", hgb_p, hgb_t)
    logger.info("Strategy Ensemble  | CV profit=%0.0f | threshold=%0.3f", ens_p, ens_t)

    if ens_p > hgb_p:
        return "ensemble", ens_t, ens_p
    return "hgb", hgb_t, hgb_p


def _predict_test(
    X_train: object,
    y: object,
    X_test: object,
    best_hgb_params: dict,
    strategy: str,
    config: Config | None = None,
) -> np.ndarray:
    """Fit the chosen strategy on full training data and return test probabilities."""
    c = config or cfg
    hgb = HistGradientBoostingClassifier(random_state=c.random_state, **best_hgb_params)
    rf = RandomForestClassifier(random_state=c.random_state, n_jobs=-1, **RF_CONFIG)
    lr = logistic_pipeline(C=LR_C, class_weight="balanced")

    if strategy == "ensemble":
        hgb.fit(X_train, y)
        rf.fit(X_train, y)
        lr.fit(X_train, y)
        w = c.ensemble_weights
        return (
            w["hgb"] * hgb.predict_proba(X_test)[:, 1]
            + w["rf"] * rf.predict_proba(X_test)[:, 1]
            + w["lr"] * lr.predict_proba(X_test)[:, 1]
        )

    final = calibrate(hgb, n_folds=cv_splits(y))
    final.fit(X_train, y)
    return final.predict_proba(X_test)[:, 1]


def main() -> None:
    """Main pipeline: feature selection → HGB tuning → strategy selection → submission."""
    logger.info(
        "Starting pipeline | fast_mode=%s | filter_n=%d | embedded_n=%d | param_iter=%d",
        cfg.fast_mode, cfg.filter_top_n, cfg.embedded_target_n, cfg.param_search_iter,
    )

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    X_train, y_train, X_test = load_project_data(data_dir=data_dir)
    y = y_train.iloc[:, 0]

    selector = select_features(X_train, y)
    features = selector.selected_features_
    if not features:
        logger.warning("No profitable features found. Exiting.")
        return

    X_tr = X_train[features]
    X_te = X_test[features]

    best_params, _, _ = tune_hgb_for_profit(X_tr, y, num_vars=len(features))
    strategy, threshold, cv_profit = _evaluate_ensemble(X_tr, y, best_params, num_vars=len(features))
    logger.info(
        "Selected strategy: %s | CV profit=%0.0f | threshold=%0.3f",
        strategy, cv_profit, threshold,
    )

    y_test_proba = _predict_test(X_tr, y, X_te, best_params, strategy)
    top_indices = select_offer_indices(y_test_proba, threshold=threshold, max_offers=cfg.max_offers)
    submission_indices = top_indices + 1

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    write_submission(submission_indices, features, project_root, cfg.team_name)
    logger.info("Selected %d customers.", len(submission_indices))


if __name__ == "__main__":
    main()
