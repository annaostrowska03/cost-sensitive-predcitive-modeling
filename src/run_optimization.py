import logging
import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from src.visualization import plot_learning_curve
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_loader import load_project_data
from src.feature_selection import ProfitDrivenFeatureSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

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
        max_offers=1000
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

    final_model = HistGradientBoostingClassifier(random_state=42, max_iter=200)
    final_model.fit(X_train[final_features], y)
    
    y_test_pred_proba = final_model.predict_proba(X_test[final_features])[:, 1]

    top_indices = np.argsort(y_test_pred_proba)[::-1][:1000]

    submission_indices = top_indices + 1

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
