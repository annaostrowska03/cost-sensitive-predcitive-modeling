# Cost-Sensitive Predictive Modeling for Targeted Marketing

## Project Overview

The objective of this project is to develop a high-performance, cost-effective predictive
model to identify customers most likely to convert on a marketing offer.

Sending unsolicited offers to uninterested clients creates *Marketing Fatigue* and erodes
brand value. This project treats customer attention as a finite resource and operates under
two hard constraints:

1. **Data investment cost** — every feature used in the model incurs a 200 EUR acquisition
   cost, forcing aggressive feature selection.
2. **Targeting budget** — the campaign is capped at 1,000 customers; only those ranked
   highest by the model receive the offer.

### Scoring formula

```
Score = (TP × 10) − (FP × 5) − (num_variables × 200)
```

### Results

| | |
|---|---|
| **Selected features** | V160, V191, V215 (3 variables — stability-selected) |
| **Unbiased CV profit** (nested CV, 3 folds) | **2 510 ± 636 EUR** |
| **OOF CV profit** (HGB on stable features) | 5 125 EUR *(biased — do not report)* |
| **Strategy** | HGB-only (ensemble did not improve) |
| **Decision threshold** | 0.170 |
| **Submission** | 1 000 customers |

The unbiased estimate comes from nested CV where feature selection (all 4 stages)
runs independently inside each outer fold.  Features that appeared in ≥ 2/3 folds
(V160, V191, V215) are used for the final model.  See `docs/nested_cv_summary.png`.

## Team

- [Anna Ostrowska](https://github.com/annaostrowska03)
- [Gabriela Majstrak](https://github.com/GabrielaMajstrak)
- [Igor Lechoszest](https://github.com/IgorLechoszest)

## Project structure

```
.
├── data/                     # raw data (gitignored)
├── docs/                     # plots and visualisations
│   ├── profit_curve.png      # SFS profit vs number of features
│   ├── experiments_comparison.png
│   └── test_predictions.png
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Experiments_Summary.ipynb
├── presentation/
├── report/
├── results/                  # old submission variants and experiment CSVs
├── src/
│   ├── config.py             # all env-var config in one place
│   ├── data_loader.py
│   ├── evaluation.py         # calculate_profit, select_offer_indices
│   ├── feature_selection.py  # 4-stage ProfitDrivenFeatureSelector
│   ├── free_feature_engineering.py
│   ├── run_optimization.py   # main pipeline entry point
│   ├── run_experiments.py    # multi-config sweep
│   ├── utils.py              # shared helpers
│   ├── visualization.py
│   └── experiments/          # alternative pipeline variants
│       ├── run_optimization_ensemble.py
│       ├── run_optimization_interactions.py
│       ├── run_optimization_max_profit.py
│       ├── run_optimization_rf_only.py
│       └── run_optimization_threshold_search.py
├── ids_obs.txt               # current submission — customer indices
├── ids_vars.txt              # current submission — variable indices
├── pyproject.toml
└── requirements.txt
```

## Setup

**Recommended — [uv](https://docs.astral.sh/uv/):**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # install uv (once)
uv sync
```

**Fallback — pip:**

```powershell
pip install -r requirements.txt
```

## Running the main pipeline

```powershell
uv run python -m src.run_optimization
```

Runs the full pipeline: 4-stage feature selection → HGB hyperparameter tuning →
ensemble strategy selection → submission files written to project root.

```powershell
$env:TEAM_NAME = "123456_98765_98764"
uv run python -m src.run_optimization
```

For notebooks:

```powershell
uv sync --extra notebooks
uv run jupyter notebook
```

## Alternative pipelines

All variants live in `src/experiments/` and share the same interface.

| Module | Approach |
|--------|----------|
| `src.experiments.run_optimization_threshold_search` | Calibrated HGB + grid threshold search |
| `src.experiments.run_optimization_ensemble` | Weighted soft-voting (HGB + RF + LR) |
| `src.experiments.run_optimization_interactions` | Free engineered features (products, ratios, PCA, KMeans) |
| `src.experiments.run_optimization_rf_only` | Random Forest + threshold search |
| `src.experiments.run_optimization_max_profit` | Model zoo + pairwise blend search |
| `src.run_experiments` | Multi-config feature-selection sweep |

```powershell
uv run python -m src.experiments.run_optimization_ensemble
```

## Runtime knobs

| Environment variable | Default (slow/fast) | Effect |
|----------------------|---------------------|--------|
| `CSM_FAST_MODE=1` | `0` | Reduces grid sizes for quick benchmarks |
| `CSM_FILTER_TOP_N` | `55` / `40` | RF-filter stage width |
| `CSM_EMBEDDED_TARGET_N` | `15` / `10` | Embedded-stage target feature count |
| `CSM_PARAM_SEARCH_ITER` | `24` / `8` | HGB random-search iterations |
| `CSM_THRESHOLD_MIN` | `0.15` | Lower bound of threshold grid |
| `CSM_THRESHOLD_MAX` | `0.50` | Upper bound of threshold grid |
| `CSM_THRESHOLD_GRID_SIZE` | `36` / `17` | Number of threshold grid points |
| `CSM_THRESHOLD_MARGIN` | `0.02` | Conservative threshold shift to reduce FP on test |
| `CSM_SFS_REPEATS` | `3` / `1` | Repeated-CV runs in SFS wrapper |
| `CSM_SFS_FOLDS` | `5` | CV folds inside SFS wrapper |
| `CSM_FILTER_RF_N` | `100` | Trees in Stage-1 RF filter |
| `CSM_EMBEDDED_RF_N` | `300` | Trees in Stage-2 embedded RF |

---

*Advanced Machine Learning — Data Science @ Warsaw University of Technology*
