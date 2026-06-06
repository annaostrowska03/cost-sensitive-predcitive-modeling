from __future__ import annotations

import logging
import os
from collections import Counter

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


def plot_nested_cv_summary(
    fold_profits: list[float],
    fold_features: list[list[str]],
    stable_features: list[str],
    output_path: str = "docs/nested_cv_summary.png",
) -> None:
    """Two-panel summary of nested CV results.

    Left panel — unbiased profit per outer fold (bar chart) with mean ± std.
    Right panel — feature stability: how many folds each feature appeared in.
    Stable features (used in final model) are highlighted in dark blue.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    n_folds = len(fold_profits)
    mean_p = float(np.mean(fold_profits))
    std_p = float(np.std(fold_profits))

    counts = Counter(f for fs in fold_features for f in fs)
    all_features = sorted(counts, key=lambda f: (-counts[f], f))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))

    # --- left: fold profits ---
    colors = ["#1f77b4" if p >= mean_p else "#aec7e8" for p in fold_profits]
    bars = ax_left.bar(
        [f"Fold {i+1}" for i in range(n_folds)], fold_profits,
        color=colors, edgecolor="black", linewidth=0.8,
    )
    ax_left.axhline(mean_p, color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean = {mean_p:,.0f} EUR")
    ax_left.fill_between(
        [-0.5, n_folds - 0.5],
        mean_p - std_p, mean_p + std_p,
        color="red", alpha=0.1, label=f"±1 std = {std_p:,.0f} EUR",
    )
    for bar, val in zip(bars, fold_profits, strict=True):
        ax_left.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{val:,.0f}", ha="center", va="bottom", fontsize=10,
        )
    ax_left.set_title("Unbiased Profit per Outer Fold", fontsize=12)
    ax_left.set_ylabel("Profit (EUR)", fontsize=11)
    ax_left.set_ylim(0, max(fold_profits) * 1.25)
    ax_left.legend(fontsize=10)
    ax_left.grid(axis="y", linestyle="--", alpha=0.5)

    # --- right: feature stability ---
    feat_counts = [counts[f] for f in all_features]
    bar_colors = ["#1f4e79" if f in stable_features else "#aec7e8" for f in all_features]
    ax_right.barh(all_features, feat_counts, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax_right.axvline(2, color="red", linestyle="--", linewidth=1.5,
                     label=f"Stability threshold (≥2/{n_folds} folds)")
    ax_right.set_title("Feature Stability Across Folds", fontsize=12)
    ax_right.set_xlabel("Number of folds feature was selected", fontsize=11)
    ax_right.set_xlim(0, n_folds + 0.5)
    ax_right.set_xticks(range(n_folds + 1))
    ax_right.legend(fontsize=10)
    ax_right.grid(axis="x", linestyle="--", alpha=0.5)

    # Legend patch for stable vs unstable
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f4e79", edgecolor="black", label="Stable (used in model)"),
        Patch(facecolor="#aec7e8", edgecolor="black", label="Unstable (discarded)"),
    ]
    ax_right.legend(handles=legend_elements, fontsize=10, loc="lower right")

    fig.suptitle(
        f"Nested CV Results — Unbiased profit: {mean_p:,.0f} ± {std_p:,.0f} EUR",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Nested CV summary saved to: %s", output_path)
