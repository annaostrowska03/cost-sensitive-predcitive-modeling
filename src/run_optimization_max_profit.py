import logging
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_project_data
from src.evaluation import calculate_profit, select_offer_indices
from src.feature_selection import ProfitDrivenFeatureSelector
from src.free_feature_engineering import build_free_engineered_features

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
FILTER_TOP_N = int(os.getenv("CSM_FILTER_TOP_N", "40" if FAST_MODE else "45"))
EMBEDDED_TARGET_N = int(os.getenv("CSM_EMBEDDED_TARGET_N", "10" if FAST_MODE else "12"))
PCA_COMPONENTS = int(os.getenv("CSM_PCA_COMPONENTS", "2"))
CLUSTER_COUNT = int(os.getenv("CSM_CLUSTER_COUNT", "4"))
RANKING_THRESHOLD = float(os.getenv("CSM_RANKING_THRESHOLD", "-0.000001"))
BLEND_STEP = float(os.getenv("CSM_BLEND_STEP", "0.1" if FAST_MODE else "0.05"))


def get_cv_splits(y):
    """
    Returns a valid number of stratified CV folds based on the minority class size.
    """
    min_class_count = int(y.value_counts().min())
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        raise ValueError("Not enough class samples for CV.")
    return n_splits


def score_profit(y_true, y_pred_proba, num_vars):
    """
    Scores a ranking by always allowing the model to sort all customers and keep the best 1000.
    """
    return calculate_profit(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        threshold=RANKING_THRESHOLD,
        num_vars=num_vars,
        max_offers=MAX_OFFERS,
    )


def build_logistic_pipeline(C_value, class_weight):
    """
    Creates a regularized logistic regression pipeline for engineered features.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=C_value,
                    max_iter=4000,
                    class_weight=class_weight,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def candidate_models():
    """
    Defines the model zoo searched for the highest empirical profit.
    """
    models = {
        "logreg_c_0.05_balanced": build_logistic_pipeline(0.05, "balanced"),
        "logreg_c_0.10_balanced": build_logistic_pipeline(0.10, "balanced"),
        "logreg_c_0.20_balanced": build_logistic_pipeline(0.20, "balanced"),
        "logreg_c_0.50_balanced": build_logistic_pipeline(0.50, "balanced"),
        "logreg_c_1.00_balanced": build_logistic_pipeline(1.00, "balanced"),
        "logreg_c_0.20_none": build_logistic_pipeline(0.20, None),
        "logreg_c_0.50_none": build_logistic_pipeline(0.50, None),
        "rf_depth_6_leaf_8": RandomForestClassifier(
            n_estimators=300 if FAST_MODE else 500,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=8,
            max_features=0.5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "rf_depth_8_leaf_4": RandomForestClassifier(
            n_estimators=300 if FAST_MODE else 500,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features=0.5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "rf_depth_6_leaf_12": RandomForestClassifier(
            n_estimators=200 if FAST_MODE else 300,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=12,
            max_features=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hgb_cautious": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            max_iter=320,
            learning_rate=0.03,
            min_samples_leaf=40,
            max_leaf_nodes=63,
        ),
        "hgb_expressive": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            max_iter=280,
            learning_rate=0.05,
            min_samples_leaf=12,
            max_leaf_nodes=127,
        ),
    }
    return models


def generate_oof_predictions(model, X, y):
    """
    Generates out-of-fold probabilities for one model.
    """
    cv = StratifiedKFold(n_splits=get_cv_splits(y), shuffle=True, random_state=RANDOM_STATE)
    oof_pred_proba = np.zeros(len(y), dtype=float)

    for train_idx, valid_idx in cv.split(X, y):
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_pred_proba[valid_idx] = fold_model.predict_proba(X.iloc[valid_idx])[:, 1]

    return oof_pred_proba


def evaluate_individual_models(models, X, y, num_vars):
    """
    Evaluates each candidate model independently and returns their OOF predictions and profits.
    """
    model_results = []
    oof_predictions = {}

    for model_name, model in models.items():
        logger.info("Evaluating model: %s", model_name)
        oof_pred_proba = generate_oof_predictions(model, X, y)
        profit = score_profit(y, oof_pred_proba, num_vars=num_vars)
        oof_predictions[model_name] = oof_pred_proba
        model_results.append((model_name, profit))
        logger.info("Model %s | CV-profit=%0.0f", model_name, profit)

    model_results.sort(key=lambda item: item[1], reverse=True)
    return model_results, oof_predictions


def search_two_model_blends(model_results, oof_predictions, y, num_vars):
    """
    Searches weighted blends for the strongest pairs of models.
    """
    best_name = None
    best_profit = -np.inf
    best_scores = None

    top_model_names = [name for name, _ in model_results[:5]]
    weight_grid = np.arange(0.0, 1.0 + BLEND_STEP, BLEND_STEP)

    for left_name, right_name in combinations(top_model_names, 2):
        left_pred = oof_predictions[left_name]
        right_pred = oof_predictions[right_name]
        for left_weight in weight_grid:
            right_weight = 1.0 - left_weight
            blended_scores = left_weight * left_pred + right_weight * right_pred
            profit = score_profit(y, blended_scores, num_vars=num_vars)
            if profit > best_profit:
                best_profit = profit
                best_name = f"blend::{left_name}::{right_name}::{left_weight:0.2f}"
                best_scores = blended_scores

    return best_name, best_profit, best_scores


def fit_full_model(model, X, y):
    """
    Fits one model on the full training set.
    """
    fitted_model = clone(model)
    fitted_model.fit(X, y)
    return fitted_model


def parse_blend_name(blend_name):
    """
    Parses the encoded blend descriptor into model names and weights.
    """
    _, left_name, right_name, left_weight = blend_name.split("::")
    left_weight = float(left_weight)
    right_weight = 1.0 - left_weight
    return left_name, right_name, left_weight, right_weight


def write_submission(submission_indices, selected_features):
    """
    Writes the final observation and variable submission files.
    """
    raw_var_indices = [
        int(feature.replace("V", "")) for feature in selected_features
    ] if all(feature.startswith("V") for feature in selected_features) else selected_features

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    obs_file = os.path.join(project_root, f"{TEAM_NAME}_max_profit_obs.txt")
    vars_file = os.path.join(project_root, f"{TEAM_NAME}_max_profit_vars.txt")

    pd.Series(submission_indices).to_csv(obs_file, index=False, header=False)
    pd.DataFrame(raw_var_indices).to_csv(vars_file, index=False, header=False)

    logger.info("Submission files written:")
    logger.info("Observations: %s", obs_file)
    logger.info("Variables:    %s", vars_file)


def main():
    """
    Runs an aggressive search for the highest empirical profit using engineered features and blending.
    """
    logger.info("Running max-profit search pipeline...")
    logger.info(
        "Runtime config | fast_mode=%s | ranking_threshold=%0.6f | blend_step=%0.3f | pca_components=%d | clusters=%d",
        FAST_MODE,
        RANKING_THRESHOLD,
        BLEND_STEP,
        PCA_COMPONENTS,
        CLUSTER_COUNT,
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

    X_train_base = X_train[selected_features]
    X_test_base = X_test[selected_features]
    X_train_engineered, X_test_engineered = build_free_engineered_features(
        X_train_base=X_train_base,
        X_test_base=X_test_base,
        selected_features=selected_features,
        n_pca_components=PCA_COMPONENTS,
        n_clusters=CLUSTER_COUNT,
    )

    logger.info(
        "Base features: %d | engineered feature space: %d",
        X_train_base.shape[1],
        X_train_engineered.shape[1],
    )

    models = candidate_models()
    model_results, oof_predictions = evaluate_individual_models(
        models=models,
        X=X_train_engineered,
        y=y,
        num_vars=len(selected_features),
    )

    best_model_name, best_model_profit = model_results[0]
    logger.info("Best single model: %s | CV-profit=%0.0f", best_model_name, best_model_profit)

    best_blend_name, best_blend_profit, _ = search_two_model_blends(
        model_results=model_results,
        oof_predictions=oof_predictions,
        y=y,
        num_vars=len(selected_features),
    )
    logger.info("Best pair blend: %s | CV-profit=%0.0f", best_blend_name, best_blend_profit)

    if best_blend_profit > best_model_profit:
        left_name, right_name, left_weight, right_weight = parse_blend_name(best_blend_name)
        left_model = fit_full_model(models[left_name], X_train_engineered, y)
        right_model = fit_full_model(models[right_name], X_train_engineered, y)
        y_test_pred_proba = (
            left_weight * left_model.predict_proba(X_test_engineered)[:, 1]
            + right_weight * right_model.predict_proba(X_test_engineered)[:, 1]
        )
        final_name = best_blend_name
        final_profit = best_blend_profit
    else:
        final_model = fit_full_model(models[best_model_name], X_train_engineered, y)
        y_test_pred_proba = final_model.predict_proba(X_test_engineered)[:, 1]
        final_name = best_model_name
        final_profit = best_model_profit

    top_indices = select_offer_indices(
        y_pred_proba=y_test_pred_proba,
        threshold=RANKING_THRESHOLD,
        max_offers=MAX_OFFERS,
    )
    submission_indices = top_indices + 1

    logger.info("Selected final strategy: %s | expected CV-profit=%0.0f", final_name, final_profit)
    logger.info("Selected customers: %d", len(submission_indices))

    write_submission(submission_indices, selected_features)


if __name__ == "__main__":
    main()
