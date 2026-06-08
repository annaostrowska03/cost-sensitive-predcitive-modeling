from __future__ import annotations

import logging
import os
import sys

import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import Config
from .feature_selection import ProfitDrivenFeatureSelector
from .run_optimization import tune_hgb_for_profit
from .utils import calibrate, cv_splits, load_project_data, select_offer_indices, write_submission

cfg = Config()

logger = logging.getLogger("experiments")

# Decision threshold at the break-even point for uniform class priors.
_PROFIT_THRESHOLD = 1.0 / 3.0

EXPERIMENTS = [
    {"name": "Standard (filter=50, embedded=15, corr=0.85)", "filter_n": 50,  "embedded_n": 15, "corr_thresh": 0.85},
    {"name": "Wide net (filter=100, embedded=30, corr=0.90)", "filter_n": 100, "embedded_n": 30, "corr_thresh": 0.90},
    {"name": "Strict net (filter=30, embedded=10, corr=0.80)", "filter_n": 30,  "embedded_n": 10, "corr_thresh": 0.80},
]


def run_experiment(
    X_train: object,
    y_train: object,
    filter_n: int,
    embedded_n: int,
    corr_thresh: float,
    name: str,
) -> dict:
    """Run one feature-selection + HGB-tuning experiment.

    Returns a dict with keys: ``name``, ``profit``, ``features``, ``params``.
    """
    logger.info("Starting experiment: %s", name)

    selector = ProfitDrivenFeatureSelector(
        filter_top_n=filter_n,
        embedded_target_n=embedded_n,
        feature_cost=cfg.feature_cost,
        max_offers=cfg.max_offers,
        corr_threshold=corr_thresh,
    )
    selector.fit(X_train, y_train)
    features = selector.selected_features_

    logger.info(
        "[%s] %d features selected: %s | SFS profit: %.0f EUR",
        name, len(features), features, selector.expected_profit_,
    )

    if not features:
        return {"name": name, "profit": 0, "features": [], "params": {}}

    best_params, _, tuned_profit = tune_hgb_for_profit(
        X_train[features], y_train, num_vars=len(features), config=cfg,
    )
    logger.info("[%s] Tuned CV profit: %.0f EUR", name, tuned_profit)
    return {"name": name, "profit": tuned_profit, "features": features, "params": best_params}


def main() -> None:
    """Compare feature-selection configs, then train a final ensemble on the winner."""
    log_file = os.path.join(os.path.dirname(__file__), "..", "experiments.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    X_train, y_train, X_test = load_project_data(data_dir=data_dir)
    y = y_train.iloc[:, 0]

    results = [run_experiment(X_train, y, **exp) for exp in EXPERIMENTS]

    logger.info("Experiment summary:")
    for r in results:
        logger.info(
            "  %s | features=%d | profit=%.0f EUR",
            r["name"], len(r["features"]), r["profit"],
        )

    best = max(results, key=lambda r: r["profit"])
    logger.info(
        "Winner: %s | profit=%.0f EUR | features=%s",
        best["name"], best["profit"], best["features"],
    )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pd.DataFrame(results).to_csv(
        os.path.join(project_root, "experiment_results.csv"), index=False
    )

    features = best["features"]
    X_tr = X_train[features]
    X_te = X_test[features]

    ensemble = VotingClassifier(
        estimators=[
            ("hgb", HistGradientBoostingClassifier(random_state=cfg.random_state, **best["params"])),
            ("rf",  RandomForestClassifier(
                random_state=cfg.random_state, n_estimators=300, max_depth=8, class_weight="balanced",
            )),
            ("lr",  make_pipeline(
                StandardScaler(), LogisticRegression(random_state=cfg.random_state, max_iter=1000, C=0.01),
            )),
            ("mlp", make_pipeline(
                StandardScaler(), MLPClassifier(random_state=cfg.random_state, hidden_layer_sizes=(64, 32), max_iter=500),
            )),
            ("svm", make_pipeline(
                StandardScaler(), SVC(probability=True, random_state=cfg.random_state, class_weight="balanced"),
            )),
        ],
        voting="soft",
    )

    n_folds = cv_splits(y)
    final_model = calibrate(ensemble, n_folds) if n_folds >= 2 else ensemble
    final_model.fit(X_tr, y)

    top_indices = select_offer_indices(
        final_model.predict_proba(X_te)[:, 1],
        threshold=_PROFIT_THRESHOLD,
        max_offers=cfg.max_offers,
    )
    write_submission(top_indices + 1, features, project_root, cfg.team_name)
    logger.info("Done.")


if __name__ == "__main__":
    main()
