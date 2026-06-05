from __future__ import annotations

import logging
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import ParameterSampler, StratifiedKFold

from .config import Config
from .data_loader import load_project_data
from .evaluation import select_offer_indices
from .feature_selection import ProfitDrivenFeatureSelector
from .utils import (
    calibrate,
    cv_splits,
    find_best_threshold,
    logistic_pipeline,
    oof_predict_proba,
    write_submission,
)
from .visualization import plot_profit_curve


def build_calibrated_model(estimator: object, n_folds: int | None = None, cv_splits: int | None = None) -> object:
    """Notebook-compatibility wrapper for :func:`calibrate`.

    Accepts either ``n_folds=`` or ``cv_splits=`` so existing notebooks that
    use either spelling continue to work without modification.
    """
    return calibrate(estimator, n_folds=n_folds or cv_splits)

try:
    from lightgbm import LGBMClassifier as _LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

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


def _ensemble_factories(best_hgb_params: dict, config: Config) -> dict[str, callable]:
    """All model factories for the ensemble, including LightGBM when available."""
    factories: dict[str, callable] = {
        "hgb": lambda: HistGradientBoostingClassifier(
            random_state=config.random_state, **best_hgb_params
        ),
        "rf": lambda: RandomForestClassifier(
            random_state=config.random_state, n_jobs=-1, **RF_CONFIG
        ),
        "lr": lambda: logistic_pipeline(C=LR_C, class_weight="balanced"),
    }
    if _LGBM_AVAILABLE:
        n_est = 200 if config.fast_mode else 400
        factories["lgbm"] = lambda: _LGBMClassifier(
            random_state=config.random_state,
            n_estimators=n_est,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            verbose=-1,
        )
        logger.info("LightGBM available — added to ensemble.")
    return factories


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
        threshold, profit = find_best_threshold(
            y, oof, num_vars, c.threshold_grid, c.max_offers, margin=c.threshold_margin,
        )
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
) -> tuple[str, float, float, dict[str, float]]:
    """Compare HGB-only vs profit-proportional soft ensemble using OOF profit.

    Returns ``(strategy, threshold, cv_profit, weights)``.
    """
    c = config or cfg
    factories = _ensemble_factories(best_hgb_params, c)

    oof_preds: dict[str, np.ndarray] = {}
    solo_profits: dict[str, float] = {}
    for name, factory in factories.items():
        oof = oof_predict_proba(factory, X, y, random_state=c.random_state)
        oof_preds[name] = oof
        _, p = find_best_threshold(
            y, oof, num_vars, c.threshold_grid, c.max_offers, margin=c.threshold_margin,
        )
        solo_profits[name] = p
        logger.info("Model %-6s | standalone CV profit=%0.0f", name, p)

    # HGB-only benchmark (already computed above).
    hgb_t, hgb_p = find_best_threshold(
        y, oof_preds["hgb"], num_vars, c.threshold_grid, c.max_offers, margin=c.threshold_margin,
    )

    # Profit-proportional weights — models that perform better get more weight.
    raw_w = {k: max(v, 1.0) for k, v in solo_profits.items()}
    w_sum = sum(raw_w.values()) or float(len(raw_w))
    norm_w = {k: v / w_sum for k, v in raw_w.items()}

    ens_oof = sum(w * oof_preds[k] for k, w in norm_w.items())
    ens_t, ens_p = find_best_threshold(
        y, ens_oof, num_vars, c.threshold_grid, c.max_offers, margin=c.threshold_margin,
    )

    logger.info("Strategy HGB-only  | CV profit=%0.0f | threshold=%0.3f", hgb_p, hgb_t)
    logger.info(
        "Strategy Ensemble  | CV profit=%0.0f | threshold=%0.3f | weights=%s",
        ens_p, ens_t, {k: round(v, 3) for k, v in norm_w.items()},
    )

    if ens_p > hgb_p:
        return "ensemble", ens_t, ens_p, norm_w
    return "hgb", hgb_t, hgb_p, {"hgb": 1.0}


def _predict_test(
    X_train: object,
    y: object,
    X_test: object,
    best_hgb_params: dict,
    strategy: str,
    weights: dict[str, float],
    config: Config | None = None,
) -> np.ndarray:
    """Fit the chosen strategy on full training data and return test probabilities."""
    c = config or cfg
    factories = _ensemble_factories(best_hgb_params, c)

    if strategy == "ensemble":
        result = np.zeros(len(X_test))
        for name, w in weights.items():
            model = factories[name]()
            model.fit(X_train, y)
            result += w * model.predict_proba(X_test)[:, 1]
        return result

    hgb = factories["hgb"]()
    final = calibrate(hgb, n_folds=cv_splits(y))
    final.fit(X_train, y)
    return final.predict_proba(X_test)[:, 1]


def nested_cv_profit_estimate(
    X: object,
    y: object,
    n_outer_folds: int = 3,
    config: Config | None = None,
) -> float:
    """Unbiased profit estimate via nested CV.

    Each outer fold runs the full 4-stage feature selection on training data
    only, then evaluates on the held-out fold — so Stage 1-2 (supervised RF
    filter) never sees the validation labels.  This eliminates the leakage
    present when feature selection is done once on the full dataset.

    Returns mean profit scaled to the full dataset size.

    Warning: ~n_outer_folds times slower than a single pipeline run.
    """
    c = config or cfg
    outer_cv = StratifiedKFold(
        n_splits=n_outer_folds, shuffle=True, random_state=c.random_state,
    )
    fold_profits: list[float] = []

    for fold_i, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        logger.info(
            "Nested CV fold %d/%d — running full feature selection on %d samples…",
            fold_i, n_outer_folds, len(train_idx),
        )
        selector = ProfitDrivenFeatureSelector(**c.selector_kwargs)
        selector.fit(X_tr, y_tr)
        features = selector.selected_features_

        if not features:
            logger.warning("Fold %d: no features selected — profit=0.", fold_i)
            fold_profits.append(0.0)
            continue

        model = HistGradientBoostingClassifier(
            random_state=c.random_state, max_iter=200,
        )
        model.fit(X_tr[features], y_tr)
        preds = model.predict_proba(X_val[features])[:, 1]

        # Scale max_offers proportionally to the fold size.
        fold_max_offers = int(c.max_offers * len(val_idx) / len(y))
        _, fold_profit = find_best_threshold(
            y_val, preds, len(features), c.threshold_grid,
            max_offers=fold_max_offers, margin=c.threshold_margin,
        )
        # Scale profit back to full-dataset units.
        scaled = fold_profit * n_outer_folds
        fold_profits.append(scaled)
        logger.info(
            "Fold %d/%d: features=%s | fold profit=%.0f EUR (scaled=%.0f EUR)",
            fold_i, n_outer_folds, features, fold_profit, scaled,
        )

    mean_p = float(np.mean(fold_profits))
    std_p = float(np.std(fold_profits))
    logger.info(
        "Nested CV unbiased profit estimate: %.0f ± %.0f EUR (folds=%s)",
        mean_p, std_p, [round(p) for p in fold_profits],
    )
    return mean_p


def main() -> None:
    """Main pipeline: feature selection → HGB tuning → strategy selection → submission."""
    logger.info(
        "Starting pipeline | fast_mode=%s | filter_n=%d | embedded_n=%d"
        " | param_iter=%d | sfs_repeats=%d | margin=%.2f",
        cfg.fast_mode, cfg.filter_top_n, cfg.embedded_target_n,
        cfg.param_search_iter, cfg.sfs_n_repeats, cfg.threshold_margin,
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

    # Unbiased profit estimate: feature selection re-run inside each outer fold
    # so Stage 1-2 (supervised RF filter) never leaks validation labels.
    logger.info(
        "Running nested CV for unbiased profit estimate (3 outer folds)…"
        " This takes ~3x longer than a single run."
    )
    nested_cv_profit_estimate(X_train, y, n_outer_folds=3)

    best_params, _, _ = tune_hgb_for_profit(X_tr, y, num_vars=len(features))
    strategy, threshold, cv_profit, weights = _evaluate_ensemble(
        X_tr, y, best_params, num_vars=len(features),
    )
    logger.info(
        "Selected strategy: %s | CV profit=%0.0f | threshold=%0.3f",
        strategy, cv_profit, threshold,
    )

    y_test_proba = _predict_test(X_tr, y, X_te, best_params, strategy, weights)
    top_indices = select_offer_indices(y_test_proba, threshold=threshold, max_offers=cfg.max_offers)
    submission_indices = top_indices + 1

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    write_submission(submission_indices, features, project_root, cfg.team_name)
    logger.info("Selected %d customers.", len(submission_indices))


if __name__ == "__main__":
    main()
