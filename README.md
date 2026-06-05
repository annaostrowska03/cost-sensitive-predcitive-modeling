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

## Team

- [Anna Ostrowska](https://github.com/annaostrowska03)
- [Gabriela Majstrak](https://github.com/GabrielaMajstrak)
- [Igor Lechoszest](https://github.com/IgorLechoszest)

## Setup

**Recommended — [uv](https://docs.astral.sh/uv/):**

```powershell
# Install uv (once)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create venv and install dependencies
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

This runs the full 4-stage profit-driven feature selector followed by HGB hyperparameter
tuning and ensemble strategy selection. Submission files are written to the project root.

Set your team prefix before running:

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

Each script below follows the same interface (`TEAM_NAME`, `CSM_FAST_MODE`, etc.) and
writes its own pair of submission files.

| Script | Approach | Output prefix |
|--------|----------|---------------|
| `src.run_optimization_threshold_search` | Calibrated HGB + grid threshold search | `*_threshold_` |
| `src.run_optimization_ensemble` | Weighted soft-voting (HGB + RF + LR) | `*_ensemble_` |
| `src.run_optimization_interactions` | Free engineered features (products, ratios, PCA, KMeans) | `*_interactions_` |
| `src.run_optimization_rf_only` | Random Forest + threshold search | `*_rf_only_` |
| `src.run_optimization_max_profit` | Model zoo + pairwise blend search on engineered features | `*_max_profit_` |
| `src.run_experiments` | Multi-config feature-selection sweep + final ensemble | `*_obs / *_vars` |

Run any variant with:

```powershell
uv run python -m src.run_optimization_ensemble
```

## Runtime knobs

| Environment variable | Default | Effect |
|----------------------|---------|--------|
| `CSM_FAST_MODE=1` | `0` | Reduces grid sizes and estimator counts for quick benchmarks |
| `CSM_FILTER_TOP_N` | `55` | RF-filter stage width |
| `CSM_EMBEDDED_TARGET_N` | `15` | Embedded-stage target feature count |
| `CSM_PARAM_SEARCH_ITER` | `24` | HGB random-search iterations |
| `CSM_THRESHOLD_MIN` | `0.15` | Lower bound of threshold grid |
| `CSM_THRESHOLD_MAX` | `0.50` | Upper bound of threshold grid |
| `CSM_THRESHOLD_GRID_SIZE` | `36` | Number of threshold grid points |

---

*Advanced Machine Learning — Data Science @ Warsaw University of Technology*
