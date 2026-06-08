"""Regenerate submission with the best feature combo: V160+V191+V215+V32."""
from __future__ import annotations

import os

from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import Config
from src.run_optimization import tune_hgb_for_profit
from src.utils import load_project_data, select_offer_indices, write_submission

cfg = Config()
FEATURES = ["V160", "V191", "V215", "V32"]
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    """Fit the best known combo and write submission files."""
    X_train, y_train, X_test = load_project_data()
    y = y_train.iloc[:, 0]

    best_params, best_threshold, best_profit = tune_hgb_for_profit(
        X_train[FEATURES], y, num_vars=len(FEATURES),
    )
    print(f"Best OOF profit: {best_profit:.0f} EUR | threshold: {best_threshold:.3f}")

    model = HistGradientBoostingClassifier(random_state=cfg.random_state, **best_params)
    model.fit(X_train[FEATURES], y)
    test_proba = model.predict_proba(X_test[FEATURES])[:, 1]

    top_indices = select_offer_indices(test_proba, threshold=best_threshold, max_offers=cfg.max_offers)
    submission_indices = top_indices + 1

    write_submission(submission_indices, FEATURES, PROJECT_ROOT, cfg.team_name)
    print(f"Done: {len(submission_indices)} customers | features={FEATURES}")


if __name__ == "__main__":
    main()
