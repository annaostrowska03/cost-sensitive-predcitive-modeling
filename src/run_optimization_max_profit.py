from __future__ import annotations

import logging
import os
from itertools import combinations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from .config import Config
from .data_loader import load_project_data
from .evaluation import calculate_profit, select_offer_indices
from .feature_selection import ProfitDrivenFeatureSelector
from .free_feature_engineering import build_free_engineered_features
from .utils import logistic_pipeline, oof_predict_proba, write_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

cfg = Config()

# Near-zero threshold ranks all customers by score without filtering — used to
# maximise ranking profit rather than precision-based threshold selection.
_RANKING_THRESHOLD: float = -1e-6


def _score(y: object, y_pred: object, num_vars: int) -> float:
    """Profit scored by allowing all 5 000 customers to be ranked."""
    return calculate_profit(
        y, y_pred,
        threshold=_RANKING_THRESHOLD,
        num_vars=num_vars,
        max_offers=cfg.max_offers,
        feature_cost=cfg.feature_cost,
    )


def _model_zoo() -> dict[str, object]:
    """Catalogue of candidate models evaluated in the search."""
    n_trees = cfg.rf_estimators // 2 if cfg.fast_mode else cfg.rf_estimators

    def rf(max_depth: int | None, min_leaf: int) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=n_trees, max_depth=max_depth,
            min_samples_split=10, min_samples_leaf=min_leaf,
            max_features=0.5, n_jobs=-1, random_state=cfg.random_state,
        )

    return {
        "lr_c0.05":     logistic_pipeline(C=0.05, class_weight="balanced"),
        "lr_c0.10":     logistic_pipeline(C=0.10, class_weight="balanced"),
        "lr_c0.20":     logistic_pipeline(C=0.20, class_weight="balanced"),
        "lr_c0.50":     logistic_pipeline(C=0.50, class_weight="balanced"),
        "lr_c1.00":     logistic_pipeline(C=1.00, class_weight="balanced"),
        "lr_c0.20_uw":  logistic_pipeline(C=0.20, class_weight=None),
        "lr_c0.50_uw":  logistic_pipeline(C=0.50, class_weight=None),
        "rf_d6_l8":     rf(max_depth=6,  min_leaf=8),
        "rf_d8_l4":     rf(max_depth=8,  min_leaf=4),
        "rf_d6_l12":    rf(max_depth=6,  min_leaf=12),
        "hgb_cautious": HistGradientBoostingClassifier(
            random_state=cfg.random_state, max_iter=320, learning_rate=0.03,
            min_samples_leaf=40, max_leaf_nodes=63,
        ),
        "hgb_expressive": HistGradientBoostingClassifier(
            random_state=cfg.random_state, max_iter=280, learning_rate=0.05,
            min_samples_leaf=12, max_leaf_nodes=127,
        ),
    }


def _best_blend(
    model_results: list[tuple[str, float]],
    oof_preds: dict[str, np.ndarray],
    y: object,
    num_vars: int,
) -> tuple[str, float, np.ndarray]:
    """Grid-search weighted blends for the top-5 model pairs."""
    best_name, best_profit, best_scores = "", -np.inf, np.array([])
    top_names = [name for name, _ in model_results[:5]]
    weight_grid = np.arange(0.0, 1.0 + cfg.blend_step, cfg.blend_step)

    for left, right in combinations(top_names, 2):
        for w in weight_grid:
            blended = w * oof_preds[left] + (1 - w) * oof_preds[right]
            profit = _score(y, blended, num_vars)
            if profit > best_profit:
                best_profit = profit
                best_name = f"blend::{left}::{right}::{w:.2f}"
                best_scores = blended

    return best_name, best_profit, best_scores


def main() -> None:
    """Model-zoo search with pairwise blend optimisation on engineered features."""
    logger.info(
        "max-profit pipeline | fast=%s | blend_step=%0.2f | pca=%d | clusters=%d",
        cfg.fast_mode, cfg.blend_step, cfg.pca_components, cfg.cluster_count,
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
    logger.info("Base: %d features | engineered: %d", X_tr_base.shape[1], X_tr_eng.shape[1])

    models = _model_zoo()
    oof_preds: dict[str, np.ndarray] = {}
    model_results: list[tuple[str, float]] = []
    for name, model in models.items():
        oof = oof_predict_proba(lambda m=model: clone(m), X_tr_eng, y)
        profit = _score(y, oof, num_vars=len(features))
        oof_preds[name] = oof
        model_results.append((name, profit))
        logger.info("Model %s | CV profit=%0.0f", name, profit)

    model_results.sort(key=lambda x: x[1], reverse=True)
    best_single_name, best_single_profit = model_results[0]
    logger.info("Best single: %s | profit=%0.0f", best_single_name, best_single_profit)

    blend_name, blend_profit, _ = _best_blend(model_results, oof_preds, y, len(features))
    logger.info("Best blend: %s | profit=%0.0f", blend_name, blend_profit)

    if blend_profit > best_single_profit:
        _, left_name, right_name, lw_str = blend_name.split("::")
        lw = float(lw_str)
        left_model = clone(models[left_name])
        right_model = clone(models[right_name])
        left_model.fit(X_tr_eng, y)
        right_model.fit(X_tr_eng, y)
        y_test = (
            lw * left_model.predict_proba(X_te_eng)[:, 1]
            + (1 - lw) * right_model.predict_proba(X_te_eng)[:, 1]
        )
        final_name = blend_name
    else:
        final_model = clone(models[best_single_name])
        final_model.fit(X_tr_eng, y)
        y_test = final_model.predict_proba(X_te_eng)[:, 1]
        final_name = best_single_name

    top_indices = select_offer_indices(
        y_test, threshold=_RANKING_THRESHOLD, max_offers=cfg.max_offers
    )
    logger.info("Strategy: %s | customers selected: %d", final_name, len(top_indices))

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    write_submission(top_indices + 1, features, project_root, cfg.team_name, suffix="max_profit")


if __name__ == "__main__":
    main()
