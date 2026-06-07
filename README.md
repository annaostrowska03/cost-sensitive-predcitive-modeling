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
| **Selected features** | V160, V191, V215, V32 (4 variables) |
| **Unbiased CV profit** (nested CV, 5 folds) | **315 ± 1 162 EUR** |
| **OOF CV profit** (HGB on stable features + combo search) | **5 165 EUR** *(biased — do not report as expected profit)* |
| **Strategy** | HGB-only (ensemble did not improve) |
| **Decision threshold** | 0.170 |
| **Submission** | 1 000 customers |

The unbiased estimate comes from nested CV where feature selection (all 4 stages)
runs independently inside each outer fold.  Features that appeared in ≥ 3/5 folds
are used for the final model.  See `docs/nested_cv_summary.png`.

## Methodology

### How features are selected

Feature selection uses a **4-stage funnel** that progressively narrows 500 variables down to a small profitable subset:

| Stage | Method | Input → Output |
|-------|--------|----------------|
| 0 | Spearman correlation filter — drop one from each pair with \|r\| > 0.85 | 500 → ~490 |
| 1 | Random Forest importance filter — keep top 55 by mean decrease in impurity | ~490 → 55 |
| 2 | Embedded RF — deeper RF (300 trees), keep top 15 | 55 → ≤15 |
| 3 | Sequential Forward Selection (SFS) — add one variable at a time while CV profit improves | ≤15 → final set |

SFS uses **ranking mode** (threshold = −∞): instead of optimising a decision threshold during selection, it always picks the top-K customers by score.  This avoids inflating the feature score with threshold-fitting noise.  Each candidate is evaluated by averaging CV profit over 3 random seeds × 5 folds = 15 CV runs.

### How overfitting is controlled

Stages 1–2 are supervised (they see training labels), which would normally cause leakage if CV was run afterwards on the same data.  The pipeline addresses this with **nested cross-validation**:

```
Outer fold 1  ┌─ train (4000 obs) ─ run full 4-stage selection ─ fit model ─┐
              └─ val  (1000 obs)  ─────────────────────────────── evaluate  ─┘
Outer fold 2  ┌─ train (4000 obs) ─ run full 4-stage selection ─ fit model ─┐
              └─ val  (1000 obs)  ─────────────────────────────── evaluate  ─┘
Outer folds 3–5  (same)
```

Stages 1–2 never see the validation fold labels → the reported profit is **unbiased**.

### How the final feature set is chosen (stability selection)

Each outer fold independently selects a different feature subset.  Only features that appear in **≥ 3 out of 5 folds** (majority) are kept for the final model.  This discards features that looked useful on one particular data split but are likely noise.

Current stable features: **V215, V191, V160, V32, V380**

### How the model and threshold are chosen

1. **HGB hyperparameter tuning** — randomised search over 24 configurations of `HistGradientBoostingClassifier`, each scored by out-of-fold (OOF) CV profit on the stable features.  The configuration with the highest OOF profit is selected.

2. **Strategy selection** — OOF profit of HGB-only is compared against a soft ensemble (HGB + Random Forest + Logistic Regression with profit-proportional weights).  The better strategy is used.

3. **Decision threshold** — grid search over [0.15 … 0.50] on OOF predictions, plus a **conservative margin of +0.02** to reduce false positives on unseen data.

### What profit to report

| Estimate | Value | Meaning |
|----------|-------|---------|
| Nested CV — 5 folds (unbiased) | **315 ± 1 162 EUR** | Expected profit on truly unseen data |
| OOF CV on stable features | 5 085 EUR | Biased — do not report as expected performance |

The gap reflects leakage from Stages 1–2 being supervised on the full dataset.  The nested CV estimate is the correct figure to report.  The wide interval (±1 162 EUR) indicates high variance across folds — two of the five folds produced negative profit, driven by folds where feature selection chose a costly set (6 features × 200 EUR = 1 200 EUR fixed cost).

---

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
| `CSM_NESTED_CV_FOLDS` | `3` | Outer folds in nested CV (more = less bias, slower) |
| `CSM_FILTER_RF_N` | `100` | Trees in Stage-1 RF filter |
| `CSM_EMBEDDED_RF_N` | `300` | Trees in Stage-2 embedded RF |

---

*Advanced Machine Learning — Data Science @ Warsaw University of Technology*
