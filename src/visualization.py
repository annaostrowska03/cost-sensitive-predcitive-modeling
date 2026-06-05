from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_profit_curve(selector: object, output_path: str = "img/profit_curve.png") -> None:
    """Plot the wrapper-stage profit history and save to output_path.

    Shows expected total profit vs. number of variables selected during
    Sequential Forward Selection.  No-ops with a warning when the selector
    has not been fitted yet.
    """
    if not getattr(selector, "profit_history_", None):
        logger.warning("No profit history found - call selector.fit() before plotting.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    history = selector.profit_history_
    x_vals = [h[0] for h in history]
    y_vals = [h[1] for h in history]

    best_idx = int(np.argmax(y_vals))
    best_x, best_y = x_vals[best_idx], y_vals[best_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_vals, y_vals, marker="o", linestyle="-", color="#1f77b4", linewidth=2, markersize=8)
    ax.axvline(x=best_x, color="red", linestyle="--", alpha=0.7)
    ax.scatter(
        [best_x], [best_y], color="red", zorder=5, s=150, edgecolor="black",
        label=f"Business Optimum\n({best_x} vars, {best_y:,.0f} €)",
    )
    ax.annotate(
        "Observation:\nAdding variables beyond the optimum reduces value due to the 200 EUR acquisition penalty.",
        xy=(0.5, -0.15), xycoords="axes fraction", ha="center", va="center",
        fontsize=10, style="italic", color="dimgrey",
    )
    ax.set_title("Cost-Sensitive Profit Curve (Sequential Forward Selection)", fontsize=14, pad=15)
    ax.set_xlabel("Number of Variables in Model", fontsize=12)
    ax.set_ylabel("Expected Profit (EUR)", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    logger.info("Profit curve saved to: %s", output_path)
