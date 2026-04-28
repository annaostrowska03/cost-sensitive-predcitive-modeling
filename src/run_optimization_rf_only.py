import logging
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler, StratifiedKFold

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_project_data
from src.evaluation import calculate_profit, select_offer_indices
from src.feature_selection import ProfitDrivenFeatureSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MAX_OFFERS = 1000
TEAM_NAME = os.getenv("TEAM_NAME", "ids")
FAST_MODE = os.getenv("CSM_FAST_MODE", "0") == "1"
PARAM_SEARCH_ITER = int(os.getenv("CSM_PARAM_SEARCH_ITER", "8" if FAST_MODE else "16"))
FILTER_TOP_N = int(os.getenv("CSM_FILTER_TOP_N", "35" if FAST_MODE else "45"))
EMBEDDED_TARGET_N = int(os.getenv("CSM_EMBEDDED_TARGET_N", "10" if FAST_MODE else "12"))
THRESHOLD_MIN = float(os.getenv("CSM_THRESHOLD_MIN", "0.05"))
THRESHOLD_MAX = float(os.getenv("CSM_THRESHOLD_MAX", "0.45"))
THRESHOLD_GRID_SIZE = int(os.getenv("CSM_THRESHOLD_GRID_SIZE", "21" if FAST_MODE else "41"))
THRESHOLD_GRID = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_GRID_SIZE)


def get_cv_splits(y, upper_bound=5):
    """
    Returns a valid number of stratified CV folds based on the minority class size.
    """
    min_class_count = int(y.value_counts().min())
    n_splits = min(upper_bound, min_class_count)
    if n_splits < 2:
        raise ValueError("Not enough class samples for cross-validation.")
    return n_splits


def best_threshold_for_profit(y_true, y_pred_proba, num_vars):
    """
    Finds the decision threshold that maximizes campaign profit.
    """
    best_threshold = THRESHOLD_GRID[0]
    best_profit = -np.inf

    for threshold in THRESHOLD_GRID:
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


def evaluate_rf_candidate(X, y, params, num_vars, random_state=42):
    """
    Evaluates one Random Forest configuration with out-of-fold profit scoring.
    """
    cv = StratifiedKFold(n_splits=get_cv_splits(y), shuffle=True, random_state=random_state)
    oof_pred_proba = np.zeros(len(y), dtype=float)

    for train_idx, valid_idx in cv.split(X, y):
        model = RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            **params,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_pred_proba[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]

    return best_threshold_for_profit(y, oof_pred_proba, num_vars=num_vars)


def main():
    """
    Runs the Random Forest-only pipeline with hyperparameter and threshold search for profit.
    """
    logger.info("Running Random Forest-only optimization pipeline...")
    logger.info(
        "Runtime config | fast_mode=%s | param_iter=%d | threshold_range=[%0.3f, %0.3f] | threshold_grid=%d",
        FAST_MODE,
        PARAM_SEARCH_ITER,
        THRESHOLD_MIN,
        THRESHOLD_MAX,
        THRESHOLD_GRID_SIZE,
    )

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    X_train, y_train, X_test = load_project_data(data_dir=data_dir)
    y = y_train.iloc[:, 0]

    selector = ProfitDrivenFeatureSelector(
        filter_top_n=FILTER_TOP_N,
        embedded_target_n=EMBEDDED_TARGET_N,
        feature_cost=200,
        max_offers=MAX_OFFERS,
    )
    selector.fit(X_train, y)

    selected_features = selector.selected_features_
    if not selected_features:
        logger.warning("No profitable feature subset selected. Exiting.")
        return

    X_train_final = X_train[selected_features]
    X_test_final = X_test[selected_features]

    param_space = {
        "n_estimators": [200, 300, 500, 700],
        "max_depth": [None, 4, 6, 8, 12],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8, 12],
        "max_features": ["sqrt", None, 0.5, 0.75],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }
    sampled_params = list(ParameterSampler(param_space, n_iter=PARAM_SEARCH_ITER, random_state=42))

    best_params = None
    best_threshold = None
    best_profit = -np.inf

    for i, params in enumerate(sampled_params, start=1):
        threshold, profit = evaluate_rf_candidate(
            X_train_final,
            y,
            params=params,
            num_vars=len(selected_features),
            random_state=42,
        )
        logger.info(
            "[Tuning %02d/%02d] profit=%0.0f | threshold=%0.3f | params=%s",
            i,
            len(sampled_params),
            profit,
            threshold,
            params,
        )
        if profit > best_profit:
            best_profit = profit
            best_threshold = threshold
            best_params = params

    logger.info("Best CV configuration: profit=%0.0f | threshold=%0.3f", best_profit, best_threshold)
    logger.info("Best params: %s", best_params)

    final_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **best_params,
    )
    final_model.fit(X_train_final, y)

    y_test_pred_proba = final_model.predict_proba(X_test_final)[:, 1]
    top_indices = select_offer_indices(
        y_pred_proba=y_test_pred_proba,
        threshold=best_threshold,
        max_offers=MAX_OFFERS,
    )
    submission_indices = top_indices + 1

    raw_var_indices = [
        int(feature.replace("V", "")) for feature in selected_features
    ] if all(feature.startswith("V") for feature in selected_features) else selected_features

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    obs_file = os.path.join(project_root, f"{TEAM_NAME}_rf_only_obs.txt")
    vars_file = os.path.join(project_root, f"{TEAM_NAME}_rf_only_vars.txt")

    pd.Series(submission_indices).to_csv(obs_file, index=False, header=False)
    pd.DataFrame(raw_var_indices).to_csv(vars_file, index=False, header=False)

    logger.info("Submission files written:")
    logger.info("Observations: %s", obs_file)
    logger.info("Variables:    %s", vars_file)
    logger.info("Selected customers: %d", len(submission_indices))


if __name__ == "__main__":
    main()
