"""Model-zoo search with pairwise blending experiment.

Evaluates a catalogue of LR, RF, and HGB variants on engineered features,
then grid-searches weighted blends of the top-5 model pairs.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from ..config import Config
from ..feature_selection import ProfitDrivenFeatureSelector
from ..utils import (
    calculate_profit,
    load_project_data,
    logistic_pipeline,
    oof_predict_proba,
    select_offer_indices,
    write_submission,
)
from .free_feature_engineering import build_free_engineered_features

logger = logging.getLogger(__name__)

cfg = Config()

# Near-zero threshold ranks all customers by score without filtering - used to
# maximise ranking profit rather than precision-based threshold selection.
_RANKING_THRESHOLD: float = -1e-6


@dataclass(frozen=True)
class _BlendSpec:
    """Structured description of a two-model weighted blend."""
    left: str
    right: str
    weight: float

    def __str__(self) -> str:
        return f"blend({self.left}, {self.right}, w={self.weight:.2f})"


def _score(y: object, y_pred: object, num_vars: int) -> float:
    """Profit scored by allowing all 5 000 customers to be ranked."""
    return calculate_profit(
        y, y_pred,
        threshold=_RANKING_THRESHOLD,
        num_vars=num_vars,
        max_offers=cfg.max_offers,
        feature_cost=cfg.feature_cost,
    )


def _model_zoo() -> dict[str, Callable[[], object]]:
    """Catalogue of candidate model factories for the max-profit search."""
    n_trees = cfg.rf_estimators // 2 if cfg.fast_mode else cfg.rf_estimators

    def _rf(max_depth: int | None, min_leaf: int) -> Callable[[], object]:
        return lambda: RandomForestClassifier(
            n_estimators=n_trees, max_depth=max_depth,
            min_samples_split=10, min_samples_leaf=min_leaf,
            max_features=0.5, n_jobs=-1, random_state=cfg.random_state,
        )

    return {
        "lr_c0.05":     lambda: logistic_pipeline(C=0.05, class_weight="balanced"),
        "lr_c0.10":     lambda: logistic_pipeline(C=0.10, class_weight="balanced"),
        "lr_c0.20":     lambda: logistic_pipeline(C=0.20, class_weight="balanced"),
        "lr_c0.50":     lambda: logistic_pipeline(C=0.50, class_weight="balanced"),
        "lr_c1.00":     lambda: logistic_pipeline(C=1.00, class_weight="balanced"),
        "lr_c0.20_uw":  lambda: logistic_pipeline(C=0.20, class_weight=None),
        "lr_c0.50_uw":  lambda: logistic_pipeline(C=0.50, class_weight=None),
        "rf_d6_l8":     _rf(max_depth=6,  min_leaf=8),
        "rf_d8_l4":     _rf(max_depth=8,  min_leaf=4),
        "rf_d6_l12":    _rf(max_depth=6,  min_leaf=12),
        "hgb_cautious": lambda: HistGradientBoostingClassifier(
            random_state=cfg.random_state, max_iter=320, learning_rate=0.03,
            min_samples_leaf=40, max_leaf_nodes=63,
        ),
        "hgb_expressive": lambda: HistGradientBoostingClassifier(
            random_state=cfg.random_state, max_iter=280, learning_rate=0.05,
            min_samples_leaf=12, max_leaf_nodes=127,
        ),
    }


def _best_blend(
    model_results: list[tuple[str, float]],
    oof_preds: dict[str, np.ndarray],
    y: object,
    num_vars: int,
) -> tuple[_BlendSpec | None, float, np.ndarray]:
    """Grid-search weighted blends for the top-5 model pairs."""
    best_spec: _BlendSpec | None = None
    best_profit, best_scores = -np.inf, np.array([])
    top_names = [name for name, _ in model_results[:5]]
    weight_grid = np.arange(0.0, 1.0 + cfg.blend_step, cfg.blend_step)

    for left, right in combinations(top_names, 2):
        for w in weight_grid:
            blended = w * oof_preds[left] + (1 - w) * oof_preds[right]
            profit = _score(y, blended, num_vars)
            if profit > best_profit:
                best_profit = profit
                best_spec = _BlendSpec(left=left, right=right, weight=float(w))
                best_scores = blended

    return best_spec, best_profit, best_scores


def main() -> None:
    """Model-zoo search with pairwise blend optimisation on engineered features."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info(
        "max-profit pipeline | fast=%s | blend_step=%0.2f | pca=%d | clusters=%d",
        cfg.fast_mode, cfg.blend_step, cfg.pca_components, cfg.cluster_count,
    )

    X_train, y_train, X_test = load_project_data()
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

    factories = _model_zoo()
    oof_preds: dict[str, np.ndarray] = {}
    model_results: list[tuple[str, float]] = []
    for name, factory in factories.items():
        oof = oof_predict_proba(factory, X_tr_eng, y)
        profit = _score(y, oof, num_vars=len(features))
        oof_preds[name] = oof
        model_results.append((name, profit))
        logger.info("Model %s | CV profit=%0.0f", name, profit)

    model_results.sort(key=lambda x: x[1], reverse=True)
    best_single_name, best_single_profit = model_results[0]
    logger.info("Best single: %s | profit=%0.0f", best_single_name, best_single_profit)

    blend_spec, blend_profit, _ = _best_blend(model_results, oof_preds, y, len(features))
    logger.info("Best blend: %s | profit=%0.0f", blend_spec, blend_profit)

    if blend_spec is not None and blend_profit > best_single_profit:
        left_model = factories[blend_spec.left]()
        right_model = factories[blend_spec.right]()
        left_model.fit(X_tr_eng, y)
        right_model.fit(X_tr_eng, y)
        y_test = (
            blend_spec.weight * left_model.predict_proba(X_te_eng)[:, 1]
            + (1 - blend_spec.weight) * right_model.predict_proba(X_te_eng)[:, 1]
        )
        final_name = str(blend_spec)
    else:
        final_model = factories[best_single_name]()
        final_model.fit(X_tr_eng, y)
        y_test = final_model.predict_proba(X_te_eng)[:, 1]
        final_name = best_single_name

    top_indices = select_offer_indices(
        y_test, threshold=_RANKING_THRESHOLD, max_offers=cfg.max_offers
    )
    logger.info("Strategy: %s | customers selected: %d", final_name, len(top_indices))

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    write_submission(top_indices + 1, features, project_root, cfg.team_name, suffix="max_profit")


if __name__ == "__main__":
    main()
