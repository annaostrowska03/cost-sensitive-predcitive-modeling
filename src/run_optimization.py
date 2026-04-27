import logging
import os
import sys
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from src.visualization import plot_learning_curve
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_loader import load_project_data
from src.feature_selection import ProfitDrivenFeatureSelector
from src.evaluation import calculate_profit, select_offer_indices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

PROFIT_THRESHOLD = 1.0 / 3.0
MAX_OFFERS = 1000

def tune_hgb_for_profit(X, y, threshold=PROFIT_THRESHOLD, max_offers=MAX_OFFERS, random_state=42, n_iter=20):
    """
    Lightweight randomized tuning of HistGradientBoosting hyperparameters
    directly against campaign profit.
    """
    param_space = {
        "learning_rate": np.linspace(0.02, 0.2, 10),
        "max_leaf_nodes": [15, 31, 63, 127],
        "min_samples_leaf": [5, 10, 20, 40, 80],
        "max_iter": [100, 150, 200, 300],
        "l2_regularization": [0.0, 0.1, 0.5, 1.0],
    }

    min_class_count = int(y.value_counts().min())
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        raise ValueError("At least two observations per class are required for CV-based tuning.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    sampled_params = list(ParameterSampler(param_space, n_iter=n_iter, random_state=random_state))

    best_params = None
    best_profit = -np.inf

    for i, params in enumerate(sampled_params, start=1):
        oof_pred_proba = np.zeros(len(y), dtype=float)

        for train_idx, valid_idx in cv.split(X, y):
            model = HistGradientBoostingClassifier(random_state=random_state, **params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof_pred_proba[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]

        fold_profit = calculate_profit(
            y_true=y,
            y_pred_proba=oof_pred_proba,
            threshold=threshold,
            num_vars=0,
            max_offers=max_offers,
        )
        logger.info(f"[Tuning {i:02d}/{len(sampled_params)}] CV profit={fold_profit:,.0f} | params={params}")

        if fold_profit > best_profit:
            best_profit = fold_profit
            best_params = params

    logger.info(f"Best tuning result: CV profit={best_profit:,.0f} | params={best_params}")
    return best_params, best_profit

def build_calibrated_model(base_model, cv_splits=5):
    """
    Compatibility helper for sklearn versions using either `estimator` or `base_estimator`.
    """
    try:
        return CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=cv_splits)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_model, method="sigmoid", cv=cv_splits)

def main():
    """
    Main execution pipeline for Cost-Sensitive Predictive Modeling.
    1. Loads dataset.
    2. Runs feature selector (Filter -> Embedded -> Wrapper).
    3. Trains final ML model.
    4. Evaluates test set and generates submission files compliant with requirements.
    """
    logger.info("Initializing Cost-Sensitive Marketing Pipeline...")

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    X_train, y_train, X_test = load_project_data(data_dir=data_dir)
    y = y_train.iloc[:, 0]
    selector = ProfitDrivenFeatureSelector(
        filter_top_n=50, 
        embedded_target_n=15, 
        feature_cost=200, 
        max_offers=MAX_OFFERS
    )
    
    selector.fit(X_train, y)
    final_features = selector.selected_features_
    expected_profit = selector.expected_profit_
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "profit_curve.png")
    plot_learning_curve(selector, output_path=plot_path)

    if not final_features:
        logger.warning("Pipeline dictated no profitable variables. Shutting down generation.")
        return
        
    logger.info(f"Training final Model utilizing: {final_features}")

    X_train_final = X_train[final_features]
    X_test_final = X_test[final_features]

    best_params, tuned_cv_profit = tune_hgb_for_profit(
        X_train_final,
        y,
        threshold=PROFIT_THRESHOLD,
        max_offers=MAX_OFFERS,
        random_state=42,
        n_iter=20,
    )

    logger.info(f"Profit-tuned CV estimate (before calibration): {tuned_cv_profit:,.0f} EUR")
    base_model = HistGradientBoostingClassifier(random_state=42, **best_params)
    calibration_cv = min(5, int(y.value_counts().min()))
    if calibration_cv >= 2:
        final_model = build_calibrated_model(base_model, cv_splits=calibration_cv)
        final_model.fit(X_train_final, y)
    else:
        logger.warning("Not enough positive/negative examples for calibration CV. Falling back to uncalibrated model.")
        final_model = base_model.fit(X_train_final, y)
    
    y_test_pred_proba = final_model.predict_proba(X_test_final)[:, 1]

    top_indices = select_offer_indices(
        y_pred_proba=y_test_pred_proba,
        threshold=PROFIT_THRESHOLD,
        max_offers=MAX_OFFERS,
    )

    submission_indices = top_indices + 1
    logger.info(f"Selected {len(submission_indices)} profitable customers (max allowed: {MAX_OFFERS}).")

    raw_var_indices = [int(f.replace('V', '')) for f in final_features] if all(f.startswith('V') for f in final_features) else final_features

    team_name = "ids"  # TODO: Replace with student ids!
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    obs_file = os.path.join(project_root, f"{team_name}_obs.txt")
    vars_file = os.path.join(project_root, f"{team_name}_vars.txt")
    
    try:
        pd.Series(submission_indices).to_csv(obs_file, index=False, header=False)
        pd.DataFrame(raw_var_indices).to_csv(vars_file, index=False, header=False)
        logger.info("Successfully generated submissions:")
        logger.info(f"Observations: {obs_file}")
        logger.info(f"Variables:    {vars_file}")
    except Exception as e:
        logger.error(f"Failed writing submission files: {e}")

if __name__ == "__main__":
    main()
