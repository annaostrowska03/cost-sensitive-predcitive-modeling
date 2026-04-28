import logging
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
THRESHOLD_GRID = np.linspace(0.20, 0.70, 51)


def get_cv_splits(y):
    min_class_count = int(y.value_counts().min())
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        raise ValueError("Not enough class samples for CV.")
    return n_splits


def model_factories(random_state=42):
    return {
        "hgb": lambda: HistGradientBoostingClassifier(
            random_state=random_state,
            max_iter=260,
            learning_rate=0.05,
            min_samples_leaf=20,
            max_leaf_nodes=63,
        ),
        "rf": lambda: RandomForestClassifier(
            n_estimators=800,
            min_samples_leaf=8,
            random_state=random_state,
            n_jobs=-1,
        ),
        "logreg": lambda: Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.2,
                        penalty="l2",
                        max_iter=2500,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def oof_predict_proba(factory, X, y, cv):
    oof_proba = np.zeros(len(y), dtype=float)
    for train_idx, valid_idx in cv.split(X, y):
        model = factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_proba[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
    return oof_proba


def find_best_threshold(y, y_pred_proba, num_vars):
    best_threshold = THRESHOLD_GRID[0]
    best_profit = -np.inf
    for threshold in THRESHOLD_GRID:
        profit = calculate_profit(
            y_true=y,
            y_pred_proba=y_pred_proba,
            threshold=threshold,
            num_vars=num_vars,
            max_offers=MAX_OFFERS,
        )
        if profit > best_profit:
            best_profit = profit
            best_threshold = threshold
    return best_threshold, best_profit


def main():
    logger.info("Running weighted-ensemble approach...")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    X_train, y_train, X_test = load_project_data(data_dir=data_dir)
    y = y_train.iloc[:, 0]

    selector = ProfitDrivenFeatureSelector(
        filter_top_n=50,
        embedded_target_n=15,
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

    cv = StratifiedKFold(n_splits=get_cv_splits(y), shuffle=True, random_state=42)
    factories = model_factories(random_state=42)

    oof_predictions = {}
    model_weights = {}

    for model_name, factory in factories.items():
        model_oof = oof_predict_proba(factory, X_train_final, y, cv)
        oof_predictions[model_name] = model_oof

        _, model_profit = find_best_threshold(y, model_oof, num_vars=0)
        model_weights[model_name] = max(model_profit, 1.0)
        logger.info("Model %s | standalone customer-profit=%0.0f", model_name, model_profit)

    weight_sum = sum(model_weights.values())
    normalized_weights = {name: weight / weight_sum for name, weight in model_weights.items()}
    logger.info("Ensemble weights: %s", normalized_weights)

    ensemble_oof = np.zeros(len(y), dtype=float)
    for model_name, model_oof in oof_predictions.items():
        ensemble_oof += normalized_weights[model_name] * model_oof

    best_threshold, best_profit = find_best_threshold(
        y=y,
        y_pred_proba=ensemble_oof,
        num_vars=len(selected_features),
    )
    logger.info("Best ensemble CV-profit=%0.0f | threshold=%0.3f", best_profit, best_threshold)

    fitted_models = {}
    for model_name, factory in factories.items():
        model = factory()
        model.fit(X_train_final, y)
        fitted_models[model_name] = model

    y_test_pred_proba = np.zeros(len(X_test_final), dtype=float)
    for model_name, model in fitted_models.items():
        y_test_pred_proba += normalized_weights[model_name] * model.predict_proba(X_test_final)[:, 1]

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
    obs_file = os.path.join(project_root, f"{TEAM_NAME}_ensemble_obs.txt")
    vars_file = os.path.join(project_root, f"{TEAM_NAME}_ensemble_vars.txt")

    pd.Series(submission_indices).to_csv(obs_file, index=False, header=False)
    pd.DataFrame(raw_var_indices).to_csv(vars_file, index=False, header=False)

    logger.info("Submission files written:")
    logger.info("Observations: %s", obs_file)
    logger.info("Variables:    %s", vars_file)
    logger.info("Selected customers: %d", len(submission_indices))


if __name__ == "__main__":
    main()
