import logging
import os
import sys
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_project_data
from src.evaluation import calculate_profit, select_offer_indices
from src.feature_selection import ProfitDrivenFeatureSelector
from src.visualization import plot_learning_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TEAM_NAME = os.getenv("TEAM_NAME", "ids")
RANDOM_STATE = 42
MAX_OFFERS = 1000
FAST_MODE = os.getenv("CSM_FAST_MODE", "0") == "1"

# Feature-selection knobs.
SELECTOR_CONFIG = {
    "filter_top_n": int(os.getenv("CSM_FILTER_TOP_N", "40" if FAST_MODE else "55")),
    "embedded_target_n": int(os.getenv("CSM_EMBEDDED_TARGET_N", "10" if FAST_MODE else "15")),
    "feature_cost": 200,
    "max_offers": MAX_OFFERS,
    "random_state": RANDOM_STATE,
}

# Profit search knobs.
TUNING_CONFIG = {
    "param_search_iter": int(os.getenv("CSM_PARAM_SEARCH_ITER", "8" if FAST_MODE else "24")),
    "threshold_min": float(os.getenv("CSM_THRESHOLD_MIN", "0.15")),
    "threshold_max": float(os.getenv("CSM_THRESHOLD_MAX", "0.50")),
    "threshold_points": int(os.getenv("CSM_THRESHOLD_GRID_SIZE", "17" if FAST_MODE else "36")),
}

# HistGradientBoosting search space.
HGB_PARAM_SPACE = {
    "learning_rate": np.linspace(0.01, 0.18, 12),
    "max_leaf_nodes": [7, 15, 31, 63, 127],
    "min_samples_leaf": [5, 10, 20, 40, 80, 120],
    "max_iter": [120, 180, 240, 320, 420],
    "l2_regularization": [0.0, 0.05, 0.1, 0.5, 1.0, 2.0],
    "max_depth": [None, 3, 5, 7],
}

# Support models for the final ensemble. Easy to edit in one place.
SUPPORT_MODEL_CONFIG = {
    "ensemble_weights": {
        "hgb": float(os.getenv("CSM_WEIGHT_HGB", "0.55")),
        "rf": float(os.getenv("CSM_WEIGHT_RF", "0.30")),
        "lr": float(os.getenv("CSM_WEIGHT_LR", "0.15")),
    },
    "rf": {
        "n_estimators": int(os.getenv("CSM_SUPPORT_RF_ESTIMATORS", "250" if FAST_MODE else "400")),
        "max_depth": None if os.getenv("CSM_SUPPORT_RF_MAX_DEPTH", "6") == "None" else int(os.getenv("CSM_SUPPORT_RF_MAX_DEPTH", "6")),
        "min_samples_split": int(os.getenv("CSM_SUPPORT_RF_MIN_SPLIT", "10")),
        "min_samples_leaf": int(os.getenv("CSM_SUPPORT_RF_MIN_LEAF", "8")),
        "max_features": float(os.getenv("CSM_SUPPORT_RF_MAX_FEATURES", "0.5")),
        "class_weight": None if os.getenv("CSM_SUPPORT_RF_CLASS_WEIGHT", "None") == "None" else os.getenv("CSM_SUPPORT_RF_CLASS_WEIGHT", "balanced_subsample"),
    },
    "lr": {
        "C": float(os.getenv("CSM_SUPPORT_LR_C", "0.2")),
        "class_weight": None if os.getenv("CSM_SUPPORT_LR_CLASS_WEIGHT", "balanced") == "None" else os.getenv("CSM_SUPPORT_LR_CLASS_WEIGHT", "balanced"),
    },
}


def get_threshold_grid():
    """
    Returns the grid of campaign thresholds explored during profit optimization.
    """
    return np.linspace(
        TUNING_CONFIG["threshold_min"],
        TUNING_CONFIG["threshold_max"],
        TUNING_CONFIG["threshold_points"],
    )


def get_cv_splits(y, upper_bound=5):
    """
    Returns a valid number of stratified CV folds based on the minority class size.
    """
    min_class_count = int(y.value_counts().min())
    n_splits = min(upper_bound, min_class_count)
    if n_splits < 2:
        raise ValueError("At least two observations per class are required for CV-based tuning.")
    return n_splits


def build_calibrated_model(base_model, cv_splits):
    """
    Wraps an estimator with sigmoid calibration while remaining compatible across sklearn versions.
    """
    try:
        return CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=cv_splits)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_model, method="sigmoid", cv=cv_splits)


def build_logistic_pipeline():
    """
    Creates the regularized logistic regression support model used in the ensemble.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=SUPPORT_MODEL_CONFIG["lr"]["C"],
                    penalty="l2",
                    max_iter=3000,
                    class_weight=SUPPORT_MODEL_CONFIG["lr"]["class_weight"],
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def find_best_threshold(y_true, y_pred_proba, num_vars):
    """
    Finds the decision threshold that maximizes campaign profit.
    """
    best_threshold = get_threshold_grid()[0]
    best_profit = -np.inf

    for threshold in get_threshold_grid():
        profit = calculate_profit(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            threshold=threshold,
            num_vars=num_vars,
            max_offers=MAX_OFFERS,
        )
        if profit > best_profit:
            best_profit = profit
            best_threshold = threshold

    return best_threshold, best_profit


def generate_oof_predictions(model_factory, X, y):
    """
    Generates out-of-fold probabilities for a model factory using stratified CV.
    """
    cv = StratifiedKFold(n_splits=get_cv_splits(y), shuffle=True, random_state=RANDOM_STATE)
    oof_pred_proba = np.zeros(len(y), dtype=float)

    for train_idx, valid_idx in cv.split(X, y):
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_pred_proba[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]

    return oof_pred_proba


def tune_hgb_for_profit(X, y, num_vars):
    """
    Tunes HistGradientBoosting directly against campaign profit using randomized search.
    """
    sampled_params = list(
        ParameterSampler(
            HGB_PARAM_SPACE,
            n_iter=TUNING_CONFIG["param_search_iter"],
            random_state=RANDOM_STATE,
        )
    )

    best_params = None
    best_threshold = None
    best_profit = -np.inf

    for iteration, params in enumerate(sampled_params, start=1):
        oof_pred_proba = generate_oof_predictions(
            model_factory=lambda current_params=params: HistGradientBoostingClassifier(
                random_state=RANDOM_STATE,
                **current_params,
            ),
            X=X,
            y=y,
        )
        threshold, profit = find_best_threshold(y, oof_pred_proba, num_vars=num_vars)
        logger.info(
            "[HGB tuning %02d/%02d] CV profit=%0.0f | threshold=%0.3f | params=%s",
            iteration,
            len(sampled_params),
            profit,
            threshold,
            params,
        )

        if profit > best_profit:
            best_profit = profit
            best_threshold = threshold
            best_params = params

    logger.info(
        "Best HGB configuration: CV profit=%0.0f | threshold=%0.3f | params=%s",
        best_profit,
        best_threshold,
        best_params,
    )
    return best_params, best_threshold, best_profit


def select_features(X_train, y):
    """
    Runs the three-stage selector and stores the resulting profit curve plot.
    """
    selector = ProfitDrivenFeatureSelector(**SELECTOR_CONFIG)
    selector.fit(X_train, y)

    plot_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "docs",
        "profit_curve.png",
    )
    plot_learning_curve(selector, output_path=plot_path)
    return selector


def build_support_models(best_hgb_params):
    """
    Creates the tuned HGB model and the auxiliary ensemble members.
    """
    hgb_model = HistGradientBoostingClassifier(random_state=RANDOM_STATE, **best_hgb_params)
    rf_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **SUPPORT_MODEL_CONFIG["rf"],
    )
    lr_model = build_logistic_pipeline()
    return hgb_model, rf_model, lr_model


def evaluate_final_strategies(X, y, best_hgb_params, num_vars):
    """
    Compares tuned HGB against a weighted ensemble using out-of-fold profit.
    """
    hgb_oof = generate_oof_predictions(
        lambda: HistGradientBoostingClassifier(random_state=RANDOM_STATE, **best_hgb_params),
        X,
        y,
    )
    hgb_threshold, hgb_profit = find_best_threshold(y, hgb_oof, num_vars=num_vars)

    rf_oof = generate_oof_predictions(
        lambda: RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **SUPPORT_MODEL_CONFIG["rf"],
        ),
        X,
        y,
    )
    lr_oof = generate_oof_predictions(build_logistic_pipeline, X, y)

    weights = SUPPORT_MODEL_CONFIG["ensemble_weights"]
    ensemble_oof = (
        weights["hgb"] * hgb_oof
        + weights["rf"] * rf_oof
        + weights["lr"] * lr_oof
    )
    ensemble_threshold, ensemble_profit = find_best_threshold(y, ensemble_oof, num_vars=num_vars)

    logger.info("Strategy HGB-only    | CV profit=%0.0f | threshold=%0.3f", hgb_profit, hgb_threshold)
    logger.info("Strategy Ensemble    | CV profit=%0.0f | threshold=%0.3f", ensemble_profit, ensemble_threshold)

    if ensemble_profit > hgb_profit:
        return "ensemble", ensemble_threshold, ensemble_profit
    return "hgb", hgb_threshold, hgb_profit


def fit_and_predict_final_probabilities(X_train_final, y, X_test_final, best_hgb_params, strategy_name):
    """
    Fits the selected final strategy on the full training data and returns test probabilities.
    """
    hgb_model, rf_model, lr_model = build_support_models(best_hgb_params)

    if strategy_name == "ensemble":
        hgb_model.fit(X_train_final, y)
        rf_model.fit(X_train_final, y)
        lr_model.fit(X_train_final, y)

        weights = SUPPORT_MODEL_CONFIG["ensemble_weights"]
        return (
            weights["hgb"] * hgb_model.predict_proba(X_test_final)[:, 1]
            + weights["rf"] * rf_model.predict_proba(X_test_final)[:, 1]
            + weights["lr"] * lr_model.predict_proba(X_test_final)[:, 1]
        )

    calibration_cv = get_cv_splits(y)
    final_model = build_calibrated_model(hgb_model, cv_splits=calibration_cv)
    final_model.fit(X_train_final, y)
    return final_model.predict_proba(X_test_final)[:, 1]


def write_submission(submission_indices, final_features):
    """
    Writes the observation and variable submission files to the project root.
    """
    raw_var_indices = [
        int(feature.replace("V", "")) for feature in final_features
    ] if all(feature.startswith("V") for feature in final_features) else final_features

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    obs_file = os.path.join(project_root, f"{TEAM_NAME}_obs.txt")
    vars_file = os.path.join(project_root, f"{TEAM_NAME}_vars.txt")

    pd.Series(submission_indices).to_csv(obs_file, index=False, header=False)
    pd.DataFrame(raw_var_indices).to_csv(vars_file, index=False, header=False)

    logger.info("Successfully generated submissions:")
    logger.info("Observations: %s", obs_file)
    logger.info("Variables:    %s", vars_file)


def main():
    """
    Runs the main optimization pipeline with configurable model search and final strategy selection.
    """
    logger.info("Initializing Cost-Sensitive Marketing Pipeline...")
    logger.info(
        "Runtime config | fast_mode=%s | selector=%s | tuning=%s",
        FAST_MODE,
        SELECTOR_CONFIG,
        TUNING_CONFIG,
    )

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    X_train, y_train, X_test = load_project_data(data_dir=data_dir)
    y = y_train.iloc[:, 0]

    selector = select_features(X_train, y)
    final_features = selector.selected_features_
    if not final_features:
        logger.warning("Pipeline dictated no profitable variables. Shutting down generation.")
        return

    logger.info("Training final models using features: %s", final_features)
    X_train_final = X_train[final_features]
    X_test_final = X_test[final_features]

    best_hgb_params, _, best_hgb_profit = tune_hgb_for_profit(
        X_train_final,
        y,
        num_vars=len(final_features),
    )
    logger.info("Best tuned HGB profit before final strategy selection: %0.0f EUR", best_hgb_profit)

    strategy_name, final_threshold, strategy_profit = evaluate_final_strategies(
        X_train_final,
        y,
        best_hgb_params,
        num_vars=len(final_features),
    )
    logger.info(
        "Selected final strategy: %s | expected CV profit=%0.0f | threshold=%0.3f",
        strategy_name,
        strategy_profit,
        final_threshold,
    )

    y_test_pred_proba = fit_and_predict_final_probabilities(
        X_train_final,
        y,
        X_test_final,
        best_hgb_params,
        strategy_name=strategy_name,
    )

    top_indices = select_offer_indices(
        y_pred_proba=y_test_pred_proba,
        threshold=final_threshold,
        max_offers=MAX_OFFERS,
    )
    submission_indices = top_indices + 1
    logger.info("Selected %d profitable customers (max allowed: %d).", len(submission_indices), MAX_OFFERS)

    write_submission(submission_indices, final_features)


if __name__ == "__main__":
    main()
