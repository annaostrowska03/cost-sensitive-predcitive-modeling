import logging
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_loader import load_project_data
from src.feature_selection import ProfitDrivenFeatureSelector
from src.run_optimization import tune_hgb_for_profit, build_calibrated_model
from src.evaluation import select_offer_indices
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

log_file = os.path.join(os.path.dirname(__file__), '..', 'experiments.log')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("experiments")

PROFIT_THRESHOLD = 1.0 / 3.0
MAX_OFFERS = 1000
FEATURE_COST = 200

def run_experiment(X_train, y_train, filter_n, embedded_n, corr_thresh, exp_name):
    logger.info(f"Starting Experiment: {exp_name}")
    
    selector = ProfitDrivenFeatureSelector(
        filter_top_n=filter_n, 
        embedded_target_n=embedded_n, 
        feature_cost=FEATURE_COST, 
        max_offers=MAX_OFFERS
    )
    old_method = selector._step0_remove_correlated
    selector._step0_remove_correlated = lambda X, threshold=corr_thresh: old_method(X, threshold=corr_thresh)
    
    selector.fit(X_train, y_train)
    features = selector.selected_features_
    sfs_profit = selector.expected_profit_
    
    logger.info(f"[{exp_name}] Selected {len(features)} features: {features} | SFS Profit: {sfs_profit:,.0f} EUR")
    
    if len(features) == 0:
        return {"name": exp_name, "profit": 0, "features": []}

    X_train_sub = X_train[features]

    best_params, tuned_cv_profit = tune_hgb_for_profit(
        X_train_sub,
        y_train,
        threshold=PROFIT_THRESHOLD,
        max_offers=MAX_OFFERS,
        random_state=42,
        n_iter=25,
    )
    
    logger.info(f"[{exp_name}] Tuned CV Profit: {tuned_cv_profit:,.0f} EUR")
    return {
        "name": exp_name,
        "profit": tuned_cv_profit,
        "features": features,
        "params": best_params
    }


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    X_train, y_train, X_test = load_project_data(data_dir=data_dir)
    y = y_train.iloc[:, 0]

    experiments = [
        {"name": "Exp 1: Standard (Filter 50, Embedded 15, Corr 0.85)", "filter_n": 50, "embedded_n": 15, "corr_thresh": 0.85},
        {"name": "Exp 2: Wide Net (Filter 100, Embedded 30, Corr 0.90)", "filter_n": 100, "embedded_n": 30, "corr_thresh": 0.90},
        {"name": "Exp 3: Strict Net (Filter 30, Embedded 10, Corr 0.80)", "filter_n": 30, "embedded_n": 10, "corr_thresh": 0.80},
    ]

    results = []
    for exp in experiments:
        res = run_experiment(X_train, y, exp["filter_n"], exp["embedded_n"], exp["corr_thresh"], exp["name"])
        results.append(res)
    
    logger.info("EXPERIMENT SUMMARY")
    best_exp = max(results, key=lambda x: x["profit"])
    
    # Save results to a CSV file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(project_root, "experiment_results.csv"), index=False)
    logger.info(f"Results saved to {os.path.join(project_root, 'experiment_results.csv')}")

    for r in results:
        logger.info(f"{r['name']}: Features: {len(r['features'])} | Tuned Profit: {r['profit']:,.0f} EUR")
        
    logger.info(f"WINNER: {best_exp['name']} with {best_exp['profit']:,.0f} EUR!")
    logger.info(f"Winner Features: {best_exp['features']}")
    
    # Train the ultimate model using winner's config
    logger.info("Training Ultimate Ensemble on winning features...")
    final_features = best_exp['features']
    X_tr_final = X_train[final_features]
    X_te_final = X_test[final_features]

    # Soft Voting Ensemble
    hgb_model = HistGradientBoostingClassifier(random_state=42, **best_exp['params'])
    rf_model = RandomForestClassifier(random_state=42, n_estimators=300, max_depth=8, class_weight='balanced')
    lr_model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000, C=0.01))
    mlp_model = make_pipeline(StandardScaler(), MLPClassifier(random_state=42, hidden_layer_sizes=(64, 32), max_iter=500))
    svm_model = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42, class_weight='balanced'))

    ensemble = VotingClassifier(
        estimators=[('hgb', hgb_model), ('rf', rf_model), ('lr', lr_model), ('mlp', mlp_model), ('svm', svm_model)],
        voting='soft'
    )

    calibration_cv = min(5, int(y.value_counts().min()))
    if calibration_cv >= 2:
        final_model = build_calibrated_model(ensemble, cv_splits=calibration_cv)
    else:
        final_model = ensemble

    final_model.fit(X_tr_final, y)
    y_test_pred = final_model.predict_proba(X_te_final)[:, 1]

    top_indices = select_offer_indices(y_test_pred, threshold=PROFIT_THRESHOLD, max_offers=MAX_OFFERS)
    submission_indices = top_indices + 1
    raw_var_indices = [int(f.replace('V', '')) for f in final_features] if all(f.startswith('V') for f in final_features) else final_features

    team_name = "team_ids"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pd.Series(submission_indices).to_csv(os.path.join(project_root, f"{team_name}_obs.txt"), index=False, header=False)
    pd.DataFrame(raw_var_indices).to_csv(os.path.join(project_root, f"{team_name}_vars.txt"), index=False, header=False)
    
    logger.info("Files generated successfully!")

if __name__ == "__main__":
    main()
