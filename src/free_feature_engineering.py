import numpy as np
import pandas as pd
from itertools import combinations, permutations
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

EPSILON = 1e-3


def signed_log1p(values):
    """
    Applies a sign-preserving log transform that remains defined for negative values.
    """
    return np.sign(values) * np.log1p(np.abs(values))


def safe_ratio(numerator, denominator, epsilon=EPSILON):
    """
    Computes a numerically stable ratio by clipping very small denominators.
    """
    denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
    return numerator / denominator


def build_free_engineered_features(
    X_train_base,
    X_test_base,
    selected_features,
    n_pca_components=2,
    n_clusters=4,
    include_products=True,
    include_ratios=True,
    include_logs=True,
    include_row_stats=True,
    include_pca=True,
    include_cluster_features=True,
):
    """
    Builds "free" derived features from an already purchased subset of original variables.
    """
    X_train_base = X_train_base.copy()
    X_test_base = X_test_base.copy()

    train_parts = [X_train_base]
    test_parts = [X_test_base]

    if include_products:
        train_products = {}
        test_products = {}
        for left_feature, right_feature in combinations(selected_features, 2):
            feature_name = f"{left_feature}__mul__{right_feature}"
            train_products[feature_name] = X_train_base[left_feature] * X_train_base[right_feature]
            test_products[feature_name] = X_test_base[left_feature] * X_test_base[right_feature]
        if train_products:
            train_parts.append(pd.DataFrame(train_products, index=X_train_base.index))
            test_parts.append(pd.DataFrame(test_products, index=X_test_base.index))

    if include_ratios:
        train_ratios = {}
        test_ratios = {}
        for numerator_feature, denominator_feature in permutations(selected_features, 2):
            feature_name = f"{numerator_feature}__div__{denominator_feature}"
            train_ratios[feature_name] = safe_ratio(
                X_train_base[numerator_feature].to_numpy(),
                X_train_base[denominator_feature].to_numpy(),
            )
            test_ratios[feature_name] = safe_ratio(
                X_test_base[numerator_feature].to_numpy(),
                X_test_base[denominator_feature].to_numpy(),
            )
        if train_ratios:
            train_parts.append(pd.DataFrame(train_ratios, index=X_train_base.index))
            test_parts.append(pd.DataFrame(test_ratios, index=X_test_base.index))

    if include_logs:
        train_logs = {}
        test_logs = {}
        for feature in selected_features:
            feature_name = f"{feature}__log1p_signed"
            train_logs[feature_name] = signed_log1p(X_train_base[feature].to_numpy())
            test_logs[feature_name] = signed_log1p(X_test_base[feature].to_numpy())
        train_parts.append(pd.DataFrame(train_logs, index=X_train_base.index))
        test_parts.append(pd.DataFrame(test_logs, index=X_test_base.index))

    if include_row_stats:
        train_stats = pd.DataFrame(
            {
                "row_mean": X_train_base.mean(axis=1),
                "row_std": X_train_base.std(axis=1),
                "row_min": X_train_base.min(axis=1),
                "row_max": X_train_base.max(axis=1),
                "row_range": X_train_base.max(axis=1) - X_train_base.min(axis=1),
            },
            index=X_train_base.index,
        )
        test_stats = pd.DataFrame(
            {
                "row_mean": X_test_base.mean(axis=1),
                "row_std": X_test_base.std(axis=1),
                "row_min": X_test_base.min(axis=1),
                "row_max": X_test_base.max(axis=1),
                "row_range": X_test_base.max(axis=1) - X_test_base.min(axis=1),
            },
            index=X_test_base.index,
        )
        train_parts.append(train_stats)
        test_parts.append(test_stats)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_base)
    X_test_scaled = scaler.transform(X_test_base)

    if include_pca:
        n_components = min(n_pca_components, len(selected_features))
        if n_components >= 1:
            pca = PCA(n_components=n_components, random_state=42)
            train_pca = pca.fit_transform(X_train_scaled)
            test_pca = pca.transform(X_test_scaled)
            pca_columns = [f"pca_component_{idx + 1}" for idx in range(n_components)]
            train_parts.append(pd.DataFrame(train_pca, columns=pca_columns, index=X_train_base.index))
            test_parts.append(pd.DataFrame(test_pca, columns=pca_columns, index=X_test_base.index))

    if include_cluster_features:
        n_clusters = min(n_clusters, max(2, len(selected_features)))
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        train_cluster_labels = kmeans.fit_predict(X_train_scaled)
        test_cluster_labels = kmeans.predict(X_test_scaled)
        train_cluster_distances = kmeans.transform(X_train_scaled)
        test_cluster_distances = kmeans.transform(X_test_scaled)

        cluster_label_column = pd.DataFrame(
            {"cluster_id": train_cluster_labels},
            index=X_train_base.index,
        )
        test_cluster_label_column = pd.DataFrame(
            {"cluster_id": test_cluster_labels},
            index=X_test_base.index,
        )
        train_parts.append(cluster_label_column)
        test_parts.append(test_cluster_label_column)

        cluster_distance_columns = [f"cluster_distance_{idx}" for idx in range(n_clusters)]
        train_parts.append(
            pd.DataFrame(train_cluster_distances, columns=cluster_distance_columns, index=X_train_base.index)
        )
        test_parts.append(
            pd.DataFrame(test_cluster_distances, columns=cluster_distance_columns, index=X_test_base.index)
        )

    X_train_engineered = pd.concat(train_parts, axis=1)
    X_test_engineered = pd.concat(test_parts, axis=1)

    X_train_engineered = X_train_engineered.loc[:, ~X_train_engineered.columns.duplicated()]
    X_test_engineered = X_test_engineered.loc[:, ~X_test_engineered.columns.duplicated()]

    return X_train_engineered, X_test_engineered
