from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from .evaluation import calculate_profit

logger = logging.getLogger(__name__)


class ProfitDrivenFeatureSelector:
    """4-stage funnel for cost-sensitive feature selection.

    Progressively narrows the feature space to maximise net profit under:
    ``score = (TP * 10) - (FP * 5) - (num_vars * feature_cost)``

    Stages
    ------
    0. Correlation filter — drops one variable from each pair whose
       |Spearman r| exceeds *corr_threshold*.
    1. Filter — keeps the top-n features by Random Forest importance.
    2. Embedded — re-ranks survivors with a deeper RF and retains at most
       *embedded_target_n* features.
    3. Wrapper — Sequential Forward Selection driven by 5-fold CV profit,
       stopping when adding any further variable reduces expected score.
    """

    def __init__(
        self,
        filter_top_n: int = 50,
        embedded_target_n: int = 15,
        feature_cost: int = 200,
        max_offers: int = 1000,
        random_state: int = 42,
        corr_threshold: float = 0.85,
    ) -> None:
        self.filter_top_n = filter_top_n
        self.embedded_target_n = embedded_target_n
        self.feature_cost = feature_cost
        self.max_offers = max_offers
        self.random_state = random_state
        self.corr_threshold = corr_threshold
        self.selected_features_: list[str] = []
        self.expected_profit_: float = 0.0
        self.profit_history_: list[tuple[int, float]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ProfitDrivenFeatureSelector:
        """Execute the 4-stage pipeline and store the best feature subset."""
        logger.info("Initialising 4-stage profit-driven feature selection pipeline…")

        stage0 = self._step0_remove_correlated(X)
        stage1 = self._step1_filter(X[stage0], y)
        stage2 = self._step2_embedded(X, y, stage1)
        self.selected_features_, self.expected_profit_ = self._step3_wrapper(X, y, stage2)

        if not self.selected_features_:
            logger.warning(
                "No features selected — using any variable costs more than it earns."
            )
        logger.info("Pipeline done. Features: %s", self.selected_features_)
        logger.info("Expected CV profit: %.2f EUR", self.expected_profit_)
        return self

    def _step0_remove_correlated(self, X: pd.DataFrame) -> list[str]:
        """Drop one variable from each highly-correlated pair (Spearman |r| > threshold)."""
        logger.info(
            "Stage 0: correlation filter (threshold=%.2f)", self.corr_threshold
        )
        corr = X.corr(method="spearman").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
        retained = [c for c in X.columns if c not in to_drop]
        logger.debug("Stage 0: dropped %d, retained %d", len(to_drop), len(retained))
        return retained

    def _step1_filter(self, X: pd.DataFrame, y: pd.Series) -> list[str]:
        """Rank features by RF importance and keep the top *filter_top_n*."""
        logger.info("Stage 1: RF filter (%d → %d)", X.shape[1], self.filter_top_n)
        rf = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state, n_jobs=-1
        )
        rf.fit(X, y)
        top_idx = np.argsort(rf.feature_importances_)[::-1][: self.filter_top_n]
        best = X.columns[top_idx].tolist()
        logger.debug("Stage 1 top features (first 5): %s", best[:5])
        return best

    def _step2_embedded(
        self, X: pd.DataFrame, y: pd.Series, candidates: list[str]
    ) -> list[str]:
        """Re-rank with a deeper RF and retain at most *embedded_target_n* features."""
        logger.info(
            "Stage 2: embedded selection (%d → max %d)", len(candidates), self.embedded_target_n
        )
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=10, max_features="log2",
            random_state=self.random_state, n_jobs=-1,
        )
        rf.fit(X[candidates], y)
        imp = rf.feature_importances_
        top_idx = np.argsort(imp)[::-1][: self.embedded_target_n]
        best = [candidates[i] for i in top_idx if imp[i] > 0]

        # Fallback: degenerate forest — take the top slice regardless of zero importances.
        if len(best) < self.embedded_target_n and len(best) < len(candidates):
            best = [candidates[i] for i in top_idx][: self.embedded_target_n]

        logger.debug("Stage 2 selected: %s", best)
        return best

    def _step3_wrapper(
        self, X: pd.DataFrame, y: pd.Series, candidates: list[str]
    ) -> tuple[list[str], float]:
        """Sequential Forward Selection driven by 5-fold CV profit."""
        logger.info("Stage 3: SFS wrapper (profit-optimised)")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        best_profit = -np.inf
        current: list[str] = []
        available = list(candidates)

        while available:
            profits = [
                (self._cv_profit(X, y, cv, current + [feat]), feat)
                for feat in available
            ]
            best_candidate_profit, top_feat = max(profits, key=lambda x: x[0])
            self.profit_history_.append((len(current) + 1, best_candidate_profit))

            if best_candidate_profit > best_profit:
                best_profit = best_candidate_profit
                current.append(top_feat)
                available.remove(top_feat)
                logger.info(
                    "Added '%s' | CV profit: %,.0f EUR", top_feat, best_profit
                )
            else:
                logger.info(
                    "Optimum reached — '%s' would reduce profit to %,.0f EUR.",
                    top_feat, best_candidate_profit,
                )
                break

        return current, best_profit

    def _cv_profit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: StratifiedKFold,
        subset: list[str],
    ) -> float:
        """Average OOF profit for subset, penalised by variable cost."""
        model = HistGradientBoostingClassifier(
            random_state=self.random_state, max_iter=100
        )
        fold_profits: list[float] = []
        for train_idx, val_idx in cv.split(X, y):
            model.fit(X.iloc[train_idx][subset], y.iloc[train_idx])
            preds = model.predict_proba(X.iloc[val_idx][subset])[:, 1]
            fold_profits.append(
                calculate_profit(y.iloc[val_idx], preds, num_vars=0, max_offers=self.max_offers)
            )
        return float(np.mean(fold_profits)) - len(subset) * self.feature_cost
