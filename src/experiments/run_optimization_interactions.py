from __future__ import annotations

import logging
import os

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from ..config import Config
from ..data_loader import load_project_data
from ..evaluation import select_offer_indices
from ..feature_selection import ProfitDrivenFeatureSelector
from ..free_feature_engineering import build_free_engineered_features
from ..utils import find_best_threshold, logistic_pipeline, oof_predict_proba, write_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

cfg = Config()


def _candidate_factories() -> dict[str, callable]:
    """Model factories for the free-feature-engineering comparison.

    ``hgb_base`` runs on the original 6 selected features to isolate how much
    the engineered space helps.  All other models run on the full engineered set.
    ``hgb_cautious`` is more regularised; ``hgb_expressive`` allows deeper trees.
    """
    factories: dict[str, callable] = {
        "hgb_base": lambda: HistGradientBoostingClassifier(
            random_state=cfg.random_state, max_iter=240, learning_rate=0.05,
            min_samples_leaf=20, max_leaf_nodes=63,
        ),
        "hgb_cautious": lambda: HistGradientBoostingClassifier(
            random_state=cfg.random_state, max_iter=320, learning_rate=0.03,
            min_samples_leaf=40, max_leaf_nodes=63,
        ),
        "hgb_expressive": lambda: HistGradientBoostingClassifier(
            random_state=cfg.random_state, max_iter=280, learning_rate=0.05,
            min_samples_leaf=12, max_leaf_nodes=127,
        ),
        "rf": lambda: RandomForestClassifier(
            random_state=cfg.random_state,
            n_estimators=cfg.rf_estimators // 2 if cfg.fast_mode else cfg.rf_estimators,
            min_samples_leaf=8, min_samples_split=10, max_depth=6, max_features=0.5, n_jobs=-1,
        ),
        "lr": lambda: logistic_pipeline(C=0.5, class_weight="balanced", random_state=cfg.random_state),
    }
    if cfg.fast_mode:
        factories.pop("lr")
        factories.pop("hgb_expressive")
    return factories


def main() -> None:
    """Free feature engineering experiment: derived features vs. base model."""
    logger.info(
        "interactions pipeline | fast=%s | pca=%d | clusters=%d | thr_grid=%d",
        cfg.fast_mode, cfg.pca_components, cfg.cluster_count, len(cfg.threshold_grid),
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

    X_tr_base = X_train[features]
    X_te_base = X_test[features]
    X_tr_eng, X_te_eng = build_free_engineered_features(
        X_tr_base, X_te_base, features,
        n_pca_components=cfg.pca_components, n_clusters=cfg.cluster_count,
    )
    logger.info("Base features: %d | engineered: %d", X_tr_base.shape[1], X_tr_eng.shape[1])

    # hgb_base runs on original features; all others on the engineered space.
    inputs = {
        "hgb_base":       (X_tr_base, X_te_base),
        "hgb_cautious":   (X_tr_eng,  X_te_eng),
        "hgb_expressive": (X_tr_eng,  X_te_eng),
        "rf":             (X_tr_eng,  X_te_eng),
        "lr":             (X_tr_eng,  X_te_eng),
    }

    best_name, best_threshold, best_profit = None, None, -float("inf")
    for name, factory in _candidate_factories().items():
        X_in, _ = inputs[name]
        oof = oof_predict_proba(factory, X_in, y)
        threshold, profit = find_best_threshold(
            y, oof, len(features), cfg.threshold_grid, cfg.max_offers
        )
        logger.info("Model %s | CV profit=%0.0f | threshold=%0.3f", name, profit, threshold)
        if profit > best_profit:
            best_profit, best_threshold, best_name = profit, threshold, name

    logger.info(
        "Best: %s | CV profit=%0.0f | threshold=%0.3f", best_name, best_profit, best_threshold
    )

    X_tr_final, X_te_final = inputs[best_name]
    final = _candidate_factories()[best_name]()
    final.fit(X_tr_final, y)
    top_indices = select_offer_indices(
        final.predict_proba(X_te_final)[:, 1],
        threshold=best_threshold,
        max_offers=cfg.max_offers,
    )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    write_submission(top_indices + 1, features, project_root, cfg.team_name, suffix="interactions")


if __name__ == "__main__":
    main()
