"""Runtime configuration. All environment-variable reading lives here.

Every pipeline script instantiates ``Config()`` once and reads attributes
instead of calling ``os.getenv`` directly.  This keeps env-var names and
defaults in a single, auditable location.

Recognised environment variables
---------------------------------
CSM_FAST_MODE            0 / 1              reduce grids for quick benchmarks
TEAM_NAME                ids                output file prefix
CSM_FEATURE_COST         200                EUR penalty per purchased variable
CSM_FILTER_TOP_N         55 (40 fast)       Stage-1 RF filter width
CSM_EMBEDDED_TARGET_N    15 (10 fast)       Stage-2 embedded target count
CSM_PARAM_SEARCH_ITER    24 (8 fast)        HGB random-search iterations
CSM_THRESHOLD_MIN        0.15               lower bound of threshold grid
CSM_THRESHOLD_MAX        0.50               upper bound of threshold grid
CSM_THRESHOLD_GRID_SIZE  36 (17 fast)       number of threshold grid points
CSM_THRESHOLD_MARGIN     0.02               shift CV-optimal threshold up to cut FP
CSM_RF_ESTIMATORS        400 (250 fast)     trees in the RF support model
CSM_BLEND_STEP           0.05 (0.10 fast)   grid step for pair-blend search
CSM_PCA_COMPONENTS       2                  PCA components in free engineering
CSM_CLUSTER_COUNT        4                  K-means clusters in free engineering
CSM_SFS_REPEATS          3 (1 fast)         repeated-CV runs in SFS wrapper
CSM_SFS_FOLDS            5                  CV folds inside SFS wrapper
CSM_NESTED_CV_FOLDS      5                  outer folds in nested CV
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _bool(name: str) -> bool:
    return os.getenv(name, "0") == "1"


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration object.

    All fields are resolved from environment variables during construction
    and cannot be modified afterwards (``frozen=True``).  Use
    ``object.__setattr__`` inside ``__post_init__`` to set derived fields.
    """

    team_name: str = field(default_factory=lambda: os.getenv("TEAM_NAME", "ids"))
    fast_mode: bool = field(default_factory=lambda: _bool("CSM_FAST_MODE"))
    random_state: int = 42
    max_offers: int = 1000
    feature_cost: int = field(default_factory=lambda: _int("CSM_FEATURE_COST", 200))

    # feature selection
    filter_top_n: int | None = field(default=None)
    embedded_target_n: int | None = field(default=None)

    # HGB tuning
    param_search_iter: int | None = field(default=None)

    # threshold grid
    threshold_min: float = field(default_factory=lambda: _float("CSM_THRESHOLD_MIN", 0.15))
    threshold_max: float = field(default_factory=lambda: _float("CSM_THRESHOLD_MAX", 0.50))
    threshold_points: int | None = field(default=None)
    # Shift CV-optimal threshold upward to reduce FP rate on unseen data.
    threshold_margin: float = field(default_factory=lambda: _float("CSM_THRESHOLD_MARGIN", 0.02))

    # RF support model
    rf_estimators: int | None = field(default=None)

    # free feature engineering
    pca_components: int = field(default_factory=lambda: _int("CSM_PCA_COMPONENTS", 2))
    cluster_count: int = field(default_factory=lambda: _int("CSM_CLUSTER_COUNT", 4))

    # blend search
    blend_step: float | None = field(default=None)

    # SFS wrapper robustness
    sfs_n_repeats: int | None = field(default=None)
    sfs_cv_folds: int = field(default_factory=lambda: _int("CSM_SFS_FOLDS", 5))

    # Nested CV outer folds (more = less bias, slower)
    nested_cv_folds: int = field(default_factory=lambda: _int("CSM_NESTED_CV_FOLDS", 5))

    # RF trees in feature selection stages
    filter_n_estimators: int = field(default_factory=lambda: _int("CSM_FILTER_RF_N", 100))
    embedded_n_estimators: int = field(default_factory=lambda: _int("CSM_EMBEDDED_RF_N", 300))

    def __post_init__(self) -> None:
        # frozen=True prevents normal assignment; use object.__setattr__ for
        # fields whose defaults depend on fast_mode (resolved at call time).
        fm = self.fast_mode
        if self.filter_top_n is None:
            object.__setattr__(self, "filter_top_n", _int("CSM_FILTER_TOP_N", 40 if fm else 55))
        if self.embedded_target_n is None:
            object.__setattr__(self, "embedded_target_n", _int("CSM_EMBEDDED_TARGET_N", 10 if fm else 15))
        if self.param_search_iter is None:
            object.__setattr__(self, "param_search_iter", _int("CSM_PARAM_SEARCH_ITER", 8 if fm else 24))
        if self.threshold_points is None:
            object.__setattr__(self, "threshold_points", _int("CSM_THRESHOLD_GRID_SIZE", 17 if fm else 36))
        if self.rf_estimators is None:
            object.__setattr__(self, "rf_estimators", _int("CSM_RF_ESTIMATORS", 250 if fm else 400))
        if self.blend_step is None:
            object.__setattr__(self, "blend_step", _float("CSM_BLEND_STEP", 0.10 if fm else 0.05))
        if self.sfs_n_repeats is None:
            object.__setattr__(self, "sfs_n_repeats", _int("CSM_SFS_REPEATS", 1 if fm else 3))

    @property
    def threshold_grid(self) -> np.ndarray:
        """Evenly-spaced decision-threshold search grid."""
        return np.linspace(self.threshold_min, self.threshold_max, self.threshold_points)

    @property
    def selector_kwargs(self) -> dict[str, int | float]:
        """Keyword arguments for ProfitDrivenFeatureSelector."""
        return {
            "filter_top_n": self.filter_top_n,
            "embedded_target_n": self.embedded_target_n,
            "feature_cost": self.feature_cost,
            "max_offers": self.max_offers,
            "random_state": self.random_state,
            "sfs_n_repeats": self.sfs_n_repeats,
            "sfs_cv_folds": self.sfs_cv_folds,
            "filter_n_estimators": self.filter_n_estimators,
            "embedded_n_estimators": self.embedded_n_estimators,
        }

    @property
    def ensemble_weights(self) -> dict[str, float]:
        """Soft-voting ensemble weights (sum to 1.0). Used as fallback only."""
        return {"hgb": 0.55, "rf": 0.30, "lr": 0.15}
