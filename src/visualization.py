import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def plot_learning_curve(selector, output_path="img/profit_curve.png"):
    """
    Plots the expected total profit against the number of variables selected 
    during the Wrapper stage (Sequential Forward Selection).
    """
    if not selector.profit_history_:
        print("No history found. Ensure you fit the model first.")
        return
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    history = selector.profit_history_
    x_vals = [h[0] for h in history]
    y_vals = [h[1] for h in history]
    
    plt.figure(figsize=(9, 6))

    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)

    best_idx = np.argmax(y_vals)
    best_x = x_vals[best_idx]
    best_y = y_vals[best_idx]
    
    plt.axvline(x=best_x, color='red', linestyle='--', alpha=0.7)
    plt.scatter([best_x], [best_y], color='red', zorder=5, s=150, edgecolor='black', label=f'Business Optimum\n({best_x} vars, {best_y:,.0f} €)')

    plt.title('Cost-Sensitive Profit Curve (Sequential Forward Selection)', fontsize=14, pad=15)
    plt.xlabel('Number of Variables in Model', fontsize=12)
    plt.ylabel('Expected Profit (EUR)', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)

    plt.annotate(
        "Observation:\nAdding variables beyond the optimum curve reduces overall value due to the strict 200 EUR acquisition penalty.",
        xy=(0.5, -0.15), xycoords='axes fraction', ha='center', va='center',
        fontsize=10, style='italic', color='dimgrey'
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Plot successfully saved to: {output_path}")

