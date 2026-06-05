from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def select_offer_indices(
    y_pred_proba: npt.ArrayLike,
    threshold: float = 0.333,
    max_offers: int = 1000,
) -> np.ndarray:
    """Select campaign recipients using business-aware ranking.

    Keeps only probabilities above threshold, ranks them descending,
    and returns at most max_offers indices.
    """
    scores = np.array(y_pred_proba).ravel()

    if max_offers <= 0:
        return np.array([], dtype=int)

    eligible = np.where(scores > threshold)[0]
    if len(eligible) == 0:
        return np.array([], dtype=int)

    ranked = eligible[np.argsort(scores[eligible])[::-1]]
    return ranked[:max_offers]


def calculate_profit(
    y_true: npt.ArrayLike,
    y_pred_proba: npt.ArrayLike,
    threshold: float = 0.333,
    num_vars: int = 0,
    max_offers: int = 1000,
    feature_cost: int = 200,
    debug: bool = False,
) -> float:
    """Calculate campaign profit (EUR) under the scoring formula.

    ``Profit = (TP * 10) - (FP * 5) - (num_vars * feature_cost)``

    Parameters
    ----------
    y_true:
        True binary labels.
    y_pred_proba:
        Model-predicted class-1 probabilities.
    threshold:
        Minimum probability to be considered for an offer.
    num_vars:
        Number of purchased variables (each costs *feature_cost* EUR).
    max_offers:
        Hard cap on offers sent.
    feature_cost:
        EUR penalty per purchased variable (default 200).
    debug:
        When ``True``, logs a breakdown of the profit components.
    """
    labels = np.array(y_true).ravel()
    scores = np.array(y_pred_proba).ravel()

    selected = select_offer_indices(scores, threshold=threshold, max_offers=max_offers)

    tp = int(np.sum(labels[selected] == 1)) if len(selected) else 0
    fp = int(np.sum(labels[selected] == 0)) if len(selected) else 0

    customer_profit = tp * 10 - fp * 5
    variable_cost = num_vars * feature_cost
    total = customer_profit - variable_cost

    if debug:
        logger.info("Selected customers: %d", len(selected))
        logger.info("Hits (TP): %d  (+%d EUR)", tp, tp * 10)
        logger.info("Misses (FP): %d  (-%d EUR)", fp, fp * 5)
        logger.info("Variable cost (%d vars): -%d EUR", num_vars, variable_cost)
        logger.info("FINAL SCORE: %d EUR", total)

    return float(total)
