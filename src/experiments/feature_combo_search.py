"""Quick OOF profit comparison for hand-picked feature combinations."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import Config
from src.data_loader import load_project_data as load_data
from src.run_optimization import tune_hgb_for_profit

cfg = Config()

COMBOS: list[tuple[str, list[str]]] = [
    ("V160+V191+V215 (original 3)",      ["V160", "V191", "V215"]),
    ("V215+V191+V160+V32+V380 (5-fold)", ["V215", "V191", "V160", "V32", "V380"]),
    ("V215+V191+V32",                    ["V215", "V191", "V32"]),
    ("V215+V191+V380",                   ["V215", "V191", "V380"]),
    ("V215+V160+V32",                    ["V215", "V160", "V32"]),
    ("V215+V191 (2 features)",           ["V215", "V191"]),
    ("V215+V160 (2 features)",           ["V215", "V160"]),
    ("V160+V191+V215+V32 (4)",           ["V160", "V191", "V215", "V32"]),
]

def main() -> None:
    X_train, y_train, _ = load_data()
    y = y_train.iloc[:, 0]

    results: list[tuple[str, float]] = []
    for label, features in COMBOS:
        _, profit, _ = tune_hgb_for_profit(X_train[features], y, num_vars=len(features))
        results.append((label, profit))
        print(f"{profit:6.0f} EUR  {label}")

    best_label, best_profit = max(results, key=lambda x: x[1])
    print(f"\nBest: {best_profit:.0f} EUR  →  {best_label}")

if __name__ == "__main__":
    main()
