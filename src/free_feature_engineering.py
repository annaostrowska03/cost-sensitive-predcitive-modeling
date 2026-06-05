from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Denominators smaller than this are clamped to avoid near-zero division.
_RATIO_EPSILON = 1e-3


def signed_log1p(values: npt.ArrayLike) -> np.ndarray:
    """Sign-preserving log transform defined for negative values."""
    v = np.asarray(values)
    return np.sign(v) * np.log1p(np.abs(v))


def safe_ratio(
    numerator: npt.ArrayLike,
    denominator: npt.ArrayLike,
    epsilon: float = _RATIO_EPSILON,
) -> np.ndarray:
    """Numerically stable ratio — small denominators are clamped to *epsilon*."""
    d = np.asarray(denominator)
    d = np.where(np.abs(d) < epsilon, epsilon, d)
    return np.asarray(numerator) / d


def _append_features(
    train_parts: list[pd.DataFrame],
    test_parts: list[pd.DataFrame],
    train_dict: dict[str, npt.ArrayLike],
    test_dict: dict[str, npt.ArrayLike],
    train_index: pd.Index,
    test_index: pd.Index,
) -> None:
    """Append train_dict / test_dict as DataFrames if non-empty."""
    if train_dict:
        train_parts.append(pd.DataFrame(train_dict, index=train_index))
        test_parts.append(pd.DataFrame(test_dict, index=test_index))


def build_free_engineered_features(
    X_train_base: pd.DataFrame,
    X_test_base: pd.DataFrame,
    selected_features: list[str],
    n_pca_components: int = 2,
    n_clusters: int = 4,
    include_products: bool = True,
    include_ratios: bool = True,
    include_logs: bool = True,
    include_row_stats: bool = True,
    include_pca: bool = True,
    include_cluster_features: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build derived features from an already-purchased variable subset.

    All transformations fitted on training data are applied identically to
    test data to prevent leakage.  Returns (X_train_engineered, X_test_engineered).
    """
    X_tr = X_train_base.copy()
    X_te = X_test_base.copy()

    train_parts: list[pd.DataFrame] = [X_tr]
    test_parts: list[pd.DataFrame] = [X_te]

    tr_idx, te_idx = X_tr.index, X_te.index

    if include_products:
        tr_p, te_p = {}, {}
        for l, r in combinations(selected_features, 2):
            name = f"{l}__mul__{r}"
            tr_p[name] = X_tr[l] * X_tr[r]
            te_p[name] = X_te[l] * X_te[r]
        _append_features(train_parts, test_parts, tr_p, te_p, tr_idx, te_idx)

    if include_ratios:
        tr_r, te_r = {}, {}
        for num, den in permutations(selected_features, 2):
            name = f"{num}__div__{den}"
            tr_r[name] = safe_ratio(X_tr[num].to_numpy(), X_tr[den].to_numpy())
            te_r[name] = safe_ratio(X_te[num].to_numpy(), X_te[den].to_numpy())
        _append_features(train_parts, test_parts, tr_r, te_r, tr_idx, te_idx)

    if include_logs:
        tr_l, te_l = {}, {}
        for feat in selected_features:
            name = f"{feat}__log1p_signed"
            tr_l[name] = signed_log1p(X_tr[feat].to_numpy())
            te_l[name] = signed_log1p(X_te[feat].to_numpy())
        _append_features(train_parts, test_parts, tr_l, te_l, tr_idx, te_idx)

    if include_row_stats:
        def _row_stats(df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({
                "row_mean":  df.mean(axis=1),
                "row_std":   df.std(axis=1),
                "row_min":   df.min(axis=1),
                "row_max":   df.max(axis=1),
                "row_range": df.max(axis=1) - df.min(axis=1),
            }, index=df.index)
        train_parts.append(_row_stats(X_tr))
        test_parts.append(_row_stats(X_te))

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    if include_pca:
        n_comp = min(n_pca_components, len(selected_features))
        if n_comp >= 1:
            pca = PCA(n_components=n_comp, random_state=42)
            cols = [f"pca_component_{i + 1}" for i in range(n_comp)]
            train_parts.append(
                pd.DataFrame(pca.fit_transform(X_tr_scaled), columns=cols, index=tr_idx)
            )
            test_parts.append(
                pd.DataFrame(pca.transform(X_te_scaled), columns=cols, index=te_idx)
            )

    if include_cluster_features:
        k = min(n_clusters, max(2, len(selected_features)))
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
        kmeans.fit(X_tr_scaled)

        dist_cols = [f"cluster_distance_{i}" for i in range(k)]
        train_parts.append(
            pd.DataFrame({"cluster_id": kmeans.predict(X_tr_scaled)}, index=tr_idx)
        )
        train_parts.append(
            pd.DataFrame(kmeans.transform(X_tr_scaled), columns=dist_cols, index=tr_idx)
        )
        test_parts.append(
            pd.DataFrame({"cluster_id": kmeans.predict(X_te_scaled)}, index=te_idx)
        )
        test_parts.append(
            pd.DataFrame(kmeans.transform(X_te_scaled), columns=dist_cols, index=te_idx)
        )

    X_tr_eng = pd.concat(train_parts, axis=1)
    X_te_eng = pd.concat(test_parts, axis=1)

    # Deduplicate columns that may arise from name collisions between feature groups.
    X_tr_eng = X_tr_eng.loc[:, ~X_tr_eng.columns.duplicated()]
    X_te_eng = X_te_eng.loc[:, ~X_te_eng.columns.duplicated()]

    return X_tr_eng, X_te_eng
