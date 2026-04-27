import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from .evaluation import calculate_profit

logger = logging.getLogger(__name__)

class ProfitDrivenFeatureSelector:
    """
    A 3-stage funnel strategy for cost-sensitive feature selection.
    
    This class implements:
    1. Filter Stage: Rapid dimensionality reduction using Random Forest.
    2. Embedded Stage: Strict penalization using L1-regularized Logistic Regression (Lasso).
    3. Wrapper Stage: Sequential Forward Selection (SFS) guided directly by the financial profit function.
    """
    
    def __init__(self, filter_top_n=50, embedded_target_n=15, feature_cost=200, max_offers=1000, random_state=42):
        self.filter_top_n = filter_top_n
        self.embedded_target_n = embedded_target_n
        self.feature_cost = feature_cost
        self.max_offers = max_offers
        self.random_state = random_state
        self.selected_features_ = []
        self.expected_profit_ = 0.0
        self.profit_history_ = []  # Stores tuples of (num_features, expected_profit)

    def fit(self, X, y):
        """
        Executes the 3-step feature selection pipeline and fits the best subset.
        """
        logger.info("Initializing 3-stage profit-driven feature selection pipeline...")
        
        # Step 1: Filter
        candidate_features_1 = self._step1_filter(X, y)
        
        # Step 2: Embedded
        candidate_features_2 = self._step2_embedded(X, y, candidate_features_1)
        
        # Step 3: Wrapper
        self.selected_features_, self.expected_profit_ = self._step3_wrapper(X, y, candidate_features_2)
        
        if not self.selected_features_:
            logger.warning("No features selected! The pipeline determined that using any variable incurs more cost than profit.")
            
        logger.info(f"Pipeline completed successfully. Selected features: {self.selected_features_}")
        logger.info(f"Expected test profit: {self.expected_profit_:.2f} EUR")
        
        return self

    def _step1_filter(self, X, y):
        logger.info(f"Stage 1: Filter (Reducing from {X.shape[1]} to {self.filter_top_n} features)")
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        top_indices = np.argsort(importances)[::-1][:self.filter_top_n]
        best_features = X.columns[top_indices].tolist()
        
        logger.debug(f"Stage 1 features selected: {best_features[:5]}... (Total: {len(best_features)})")
        return best_features

    def _step2_embedded(self, X, y, candidate_features):
        logger.info(f"Stage 2: Embedded (Reducing from {len(candidate_features)} to max {self.embedded_target_n} features)")
        X_sub = X[candidate_features]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)
        
        lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.01, random_state=self.random_state)
        lasso.fit(X_scaled, y)
        
        importances = np.abs(lasso.coef_[0])
        top_indices = np.argsort(importances)[::-1][:self.embedded_target_n]
        best_features = [candidate_features[i] for i in top_indices if importances[i] > 0]
        
        # Fallback if Lasso zeroes out too many coefficients
        if len(best_features) < self.embedded_target_n and len(best_features) < len(candidate_features):
            best_features = [candidate_features[i] for i in top_indices][:self.embedded_target_n]
            
        logger.debug(f"Stage 2 features selected: {best_features} (Total: {len(best_features)})")
        return best_features

    def _step3_wrapper(self, X, y, candidate_features):
        logger.info("Stage 3: Wrapper (Financial Profit Optimization)")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        best_profit = -np.inf
        current_subset = []
        available_features = list(candidate_features)
        
        while available_features:
            profits = []
            for feature in available_features:
                test_subset = current_subset + [feature]
                model = HistGradientBoostingClassifier(random_state=self.random_state, max_iter=100)
                
                fold_profits = []
                for train_idx, val_idx in cv.split(X, y):
                    X_tr, y_tr = X.iloc[train_idx][test_subset], y.iloc[train_idx]
                    X_va, y_val = X.iloc[val_idx][test_subset], y.iloc[val_idx]
                    
                    model.fit(X_tr, y_tr)
                    preds_proba = model.predict_proba(X_va)[:, 1]

                    profit_from_customers = calculate_profit(
                        y_true=y_val, 
                        y_pred_proba=preds_proba, 
                        threshold=0.333, 
                        num_vars=0, # Penalty calculated collectively later
                        max_offers=self.max_offers // cv.n_splits
                    )

                    fold_profits.append(profit_from_customers * cv.n_splits)
                    
                avg_customers_profit = np.mean(fold_profits)
                total_profit = avg_customers_profit - (len(test_subset) * self.feature_cost)
                profits.append((total_profit, feature))
                
            best_avg_profit, top_feature = max(profits, key=lambda x: x[0])

            self.profit_history_.append((len(current_subset) + 1, best_avg_profit))
            
            if best_avg_profit > best_profit:
                best_profit = best_avg_profit
                current_subset.append(top_feature)
                available_features.remove(top_feature)
                logger.info(f"[+] Added feature '{top_feature}' | Expected Test Profit: {best_profit:,.0f} EUR")
            else:
                logger.info(f"[-] Optimum reached! Feature '{top_feature}' reduces profit to {best_avg_profit:,.0f} EUR.")
                break
                
        return current_subset, best_profit
