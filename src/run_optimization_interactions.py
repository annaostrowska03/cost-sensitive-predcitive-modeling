import logging
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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
FILTER_TOP_N = int(os.getenv("CSM_FILTER_TOP_N", "35" if FAST_MODE else "45"))
EMBEDDED_TARGET_N = int(os.getenv("CSM_EMBEDDED_TARGET_N", "10" if FAST_MODE else "12"))
THRESHOLD_GRID_SIZE = int(os.getenv("CSM_THRESHOLD_GRID_SIZE", "15" if FAST_MODE else "31"))
THRESHOLD_GRID = np.linspace(0.20, 0.70, THRESHOLD_GRID_SIZE)


def get_cv_splits(y):
    """
    Returns a valid number of stratified CV folds based on the minority class size.
    """
    min_class_count = int(y.value_counts().min())
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        raise ValueError("Not enough class samples for CV.")
    return n_splits


def build_interaction_features(X_train_base, X_test_base, selected_features):
    """
    Expands the selected feature set with pairwise interaction terms.
    """
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly.fit(X_train_base)

    interaction_feature_names = poly.get_feature_names_out(selected_features)
    X_train_interactions = pd.DataFrame(
        poly.transform(X_train_base),
        columns=interaction_feature_names,
        index=X_train_base.index,
    )
    X_test_interactions = pd.DataFrame(
        poly.transform(X_test_base),
        columns=interaction_feature_names,
        index=X_test_base.index,
    )

    return X_train_interactions, X_test_interactions


def candidate_models(random_state=42):
    """
    Creates the model candidates compared in the interaction-based experiment.
    """
    return {
        "hgb_base": lambda: HistGradientBoostingClassifier(
            random_state=random_state,
            max_iter=240,
            learning_rate=0.05,
            min_samples_leaf=20,
            max_leaf_nodes=63,
        ),
        "hgb_interactions": lambda: HistGradientBoostingClassifier(
            random_state=random_state,
            max_iter=240,
            learning_rate=0.05,
            min_samples_leaf=20,
            max_leaf_nodes=63,
        ),
        "logreg_interactions": lambda: Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        penalty="l2",
                        max_iter=3000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def find_best_threshold(y, y_pred_proba, num_vars):
    """
    Finds the threshold that maximizes campaign profit for given probabilities.
    """
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


def evaluate_model(factory, X, y, num_vars):
    """
    Evaluates a model with out-of-fold probabilities and profit-based threshold search.
    """
    cv = StratifiedKFold(n_splits=get_cv_splits(y), shuffle=True, random_state=42)
    oof_pred_proba = np.zeros(len(y), dtype=float)

    for train_idx, valid_idx in cv.split(X, y):
        model = factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_pred_proba[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]

    return find_best_threshold(y, oof_pred_proba, num_vars=num_vars)


def main():
    """
    Runs the interaction-feature experiment and exports the best submission variant.
    """
    logger.info("Running interaction-feature approach...")
    logger.info(
        "Runtime config | fast_mode=%s | threshold_grid=%d",
        FAST_MODE,
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

    X_train_base = X_train[selected_features]
    X_test_base = X_test[selected_features]
    X_train_interactions, X_test_interactions = build_interaction_features(
        X_train_base=X_train_base,
        X_test_base=X_test_base,
        selected_features=selected_features,
    )

    logger.info(
        "Base features: %d | interaction-expanded features: %d",
        X_train_base.shape[1],
        X_train_interactions.shape[1],
    )

    models = candidate_models(random_state=42)
    model_inputs = {
        "hgb_base": (X_train_base, X_test_base),
        "hgb_interactions": (X_train_interactions, X_test_interactions),
        "logreg_interactions": (X_train_interactions, X_test_interactions),
    }

    if FAST_MODE:
        models = {name: factory for name, factory in models.items() if name != "logreg_interactions"}

    best_model_name = None
    best_threshold = None
    best_profit = -np.inf

    for model_name, factory in models.items():
        X_train_current, _ = model_inputs[model_name]
        threshold, profit = evaluate_model(
            factory=factory,
            X=X_train_current,
            y=y,
            num_vars=len(selected_features),
        )
        logger.info("Model %s | CV-profit=%0.0f | threshold=%0.3f", model_name, profit, threshold)

        if profit > best_profit:
            best_profit = profit
            best_threshold = threshold
            best_model_name = model_name

    logger.info(
        "Best model: %s | CV-profit=%0.0f | threshold=%0.3f",
        best_model_name,
        best_profit,
        best_threshold,
    )

    final_model = models[best_model_name]()
    X_train_best, X_test_best = model_inputs[best_model_name]
    final_model.fit(X_train_best, y)
    y_test_pred_proba = final_model.predict_proba(X_test_best)[:, 1]

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
    obs_file = os.path.join(project_root, f"{TEAM_NAME}_interactions_obs.txt")
    vars_file = os.path.join(project_root, f"{TEAM_NAME}_interactions_vars.txt")

    pd.Series(submission_indices).to_csv(obs_file, index=False, header=False)
    pd.DataFrame(raw_var_indices).to_csv(vars_file, index=False, header=False)

    logger.info("Submission files written:")
    logger.info("Observations: %s", obs_file)
    logger.info("Variables:    %s", vars_file)
    logger.info("Selected customers: %d", len(submission_indices))


if __name__ == "__main__":
    main()
