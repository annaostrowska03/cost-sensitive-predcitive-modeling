from __future__ import annotations

import logging
import os

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from ..config import Config
from ..data_loader import load_project_data
from ..evaluation import select_offer_indices
from ..feature_selection import ProfitDrivenFeatureSelector
from ..utils import find_best_threshold, logistic_pipeline, oof_predict_proba, write_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

cfg = Config()


def _model_factories() -> dict[str, callable]:
    return {
        "hgb": lambda: HistGradientBoostingClassifier(
            random_state=cfg.random_state, max_iter=220, learning_rate=0.05,
            min_samples_leaf=20, max_leaf_nodes=63,
        ),
        "rf": lambda: RandomForestClassifier(
            n_estimators=cfg.rf_estimators, min_samples_leaf=8,
            random_state=cfg.random_state, n_jobs=-1,
        ),
        "lr": lambda: logistic_pipeline(C=0.2, class_weight="balanced", random_state=cfg.random_state),
    }


def main() -> None:
    """Profit-proportional soft-voting ensemble with CV-based threshold search."""
    logger.info(
        "ensemble pipeline | fast=%s | rf_estimators=%d | thr_grid=%d",
        cfg.fast_mode, cfg.rf_estimators, len(cfg.threshold_grid),
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

    factories = _model_factories()
    oof_preds: dict[str, object] = {}
    weights: dict[str, float] = {}
    for name, factory in factories.items():
        oof = oof_predict_proba(factory, X_tr, y)
        oof_preds[name] = oof
        _, solo_profit = find_best_threshold(
            y, oof, num_vars=0, threshold_grid=cfg.threshold_grid, max_offers=cfg.max_offers,
        )
        # Clamp to 1.0 so models with negative standalone profit still contribute.
        weights[name] = max(solo_profit, 1.0)
        logger.info("Model %s | standalone CV profit=%0.0f", name, solo_profit)

    weight_sum = sum(weights.values())
    # Guard against the degenerate case where every model has near-zero profit.
    if weight_sum == 0:
        weight_sum = float(len(weights))
    weights = {k: v / weight_sum for k, v in weights.items()}
    logger.info("Ensemble weights: %s", weights)

    ensemble_oof = sum(w * oof_preds[k] for k, w in weights.items())
    best_t, best_p = find_best_threshold(
        y, ensemble_oof, num_vars=len(features),
        threshold_grid=cfg.threshold_grid, max_offers=cfg.max_offers,
    )
    logger.info("Ensemble CV profit=%0.0f | threshold=%0.3f", best_p, best_t)

    fitted = {name: factory() for name, factory in factories.items()}
    for model in fitted.values():
        model.fit(X_tr, y)

    y_test = sum(weights[k] * m.predict_proba(X_te)[:, 1] for k, m in fitted.items())
    top_indices = select_offer_indices(y_test, threshold=best_t, max_offers=cfg.max_offers)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    write_submission(top_indices + 1, features, project_root, cfg.team_name, suffix="ensemble")


if __name__ == "__main__":
    main()
