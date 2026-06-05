"""Runtime configuration — all environment-variable reading lives here.

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
CSM_WEIGHT_HGB           0.55               ensemble weight for HGB
CSM_WEIGHT_RF            0.30               ensemble weight for RF
CSM_WEIGHT_LR            0.15               ensemble weight for LR
CSM_RF_ESTIMATORS        400 (250 fast)     trees in the RF support model
CSM_BLEND_STEP           0.05 (0.10 fast)   grid step for pair-blend search
CSM_PCA_COMPONENTS       2                  PCA components in free engineering
CSM_CLUSTER_COUNT        4                  K-means clusters in free engineering
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


@dataclass
class Config:
    """Immutable-after-init configuration object."""

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

    # ensemble weights
    weight_hgb: float = field(default_factory=lambda: _float("CSM_WEIGHT_HGB", 0.55))
    weight_rf: float = field(default_factory=lambda: _float("CSM_WEIGHT_RF", 0.30))
    weight_lr: float = field(default_factory=lambda: _float("CSM_WEIGHT_LR", 0.15))

    # RF support model
    rf_estimators: int | None = field(default=None)

    # free feature engineering
    pca_components: int = field(default_factory=lambda: _int("CSM_PCA_COMPONENTS", 2))
    cluster_count: int = field(default_factory=lambda: _int("CSM_CLUSTER_COUNT", 4))

    # blend search
    blend_step: float | None = field(default=None)

    def __post_init__(self) -> None:
        fm = self.fast_mode
        if self.filter_top_n is None:
            self.filter_top_n = _int("CSM_FILTER_TOP_N", 40 if fm else 55)
        if self.embedded_target_n is None:
            self.embedded_target_n = _int("CSM_EMBEDDED_TARGET_N", 10 if fm else 15)
        if self.param_search_iter is None:
            self.param_search_iter = _int("CSM_PARAM_SEARCH_ITER", 8 if fm else 24)
        if self.threshold_points is None:
            self.threshold_points = _int("CSM_THRESHOLD_GRID_SIZE", 17 if fm else 36)
        if self.rf_estimators is None:
            self.rf_estimators = _int("CSM_RF_ESTIMATORS", 250 if fm else 400)
        if self.blend_step is None:
            self.blend_step = _float("CSM_BLEND_STEP", 0.10 if fm else 0.05)

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
        }

    @property
    def ensemble_weights(self) -> dict[str, float]:
        """Soft-voting ensemble weights (sum to 1.0)."""
        return {"hgb": self.weight_hgb, "rf": self.weight_rf, "lr": self.weight_lr}
