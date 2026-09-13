
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


CLUSTER_FEATURES = [
    "average_daily_demand",
    "coefficient_variation",
    "peak_intensity",
    "demand_growth_rate",
    "fulfillment_rate",
    "lost_demand_rate",
    "stockout_product_day_rate",
    "performance_score",
    "opportunity_index",
    "risk_index",
]


def _validate_features(df: pd.DataFrame) -> None:
    missing = [
        column
        for column in CLUSTER_FEATURES
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing clustering features: {missing}"
        )


def build_clustering_matrix(
    performance: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    _validate_features(performance)

    if "outlet_id" not in performance.columns:
        raise ValueError(
            "Missing required column: outlet_id"
        )

    features = performance[
        CLUSTER_FEATURES
    ].copy()

    features = (
        features
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    scaler = StandardScaler()

    matrix = scaler.fit_transform(features)

    matrix_df = pd.DataFrame(
        matrix,
        columns=CLUSTER_FEATURES,
        index=performance.index,
    )

    return (
        matrix_df,
        features,
        scaler,
    )


def evaluate_kmeans(
    matrix: pd.DataFrame,
    min_clusters: int = 2,
    max_clusters: int = 8,
) -> pd.DataFrame:
    n_samples = len(matrix)

    if n_samples < 3:
        raise ValueError(
            "At least 3 outlets are required for clustering."
        )

    max_allowed = min(
        max_clusters,
        n_samples - 1,
    )

    if min_clusters > max_allowed:
        raise ValueError(
            "Invalid cluster range."
        )

    rows = []

    values = matrix.to_numpy()

    for k in range(
        min_clusters,
        max_allowed + 1,
    ):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
        )

        labels = model.fit_predict(values)

        score = silhouette_score(
            values,
            labels,
        )

        cluster_sizes = (
            pd.Series(labels)
            .value_counts()
            .sort_index()
        )

        rows.append(
            {
                "n_clusters": k,
                "silhouette_score": float(score),
                "min_cluster_size": int(
                    cluster_sizes.min()
                ),
                "max_cluster_size": int(
                    cluster_sizes.max()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "silhouette_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def fit_outlet_clusters(
    performance: pd.DataFrame,
    min_clusters: int = 2,
    max_clusters: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    matrix, raw_features, scaler = (
        build_clustering_matrix(
            performance
        )
    )

    evaluation = evaluate_kmeans(
        matrix,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
    )

    best_k = int(
        evaluation.iloc[0]["n_clusters"]
    )

    model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=20,
    )

    labels = model.fit_predict(
        matrix.to_numpy()
    )

    result = performance[
        ["outlet_id"]
    ].copy()

    result["cluster_id"] = labels

    result = result.merge(
        performance[
            [
                "outlet_id",
                *CLUSTER_FEATURES,
            ]
        ],
        on="outlet_id",
        how="left",
        validate="one_to_one",
    )

    result["cluster_size"] = (
        result["cluster_id"]
        .map(
            result["cluster_id"]
            .value_counts()
        )
        .astype(int)
    )

    metadata = {
        "best_k": best_k,
        "silhouette_score": float(
            evaluation.iloc[0]["silhouette_score"]
        ),
        "feature_columns": CLUSTER_FEATURES,
        "scaler": scaler,
        "model": model,
    }

    return (
        result,
        evaluation,
        metadata,
    )


def build_cluster_profiles(
    clustered: pd.DataFrame,
) -> pd.DataFrame:
    _validate_features(clustered)

    grouped = (
        clustered
        .groupby("cluster_id")
        [CLUSTER_FEATURES]
        .mean()
    )

    counts = (
        clustered["cluster_id"]
        .value_counts()
        .sort_index()
        .rename("outlet_count")
    )

    profiles = grouped.join(
        counts
    ).reset_index()

    # Relative characteristics.
    profiles["demand_scale"] = (
        profiles["average_daily_demand"]
        / profiles["average_daily_demand"].median()
    )

    profiles["growth_class"] = np.select(
        [
            profiles["demand_growth_rate"] >= 0.10,
            profiles["demand_growth_rate"] <= -0.10,
        ],
        [
            "growing",
            "declining",
        ],
        default="stable",
    )

    profiles["volatility_class"] = np.select(
        [
            profiles["coefficient_variation"] >= 0.30,
            profiles["coefficient_variation"] >= 0.15,
        ],
        [
            "high",
            "moderate",
        ],
        default="low",
    )

    profiles["fulfillment_class"] = np.select(
        [
            profiles["fulfillment_rate"] >= 0.97,
            profiles["fulfillment_rate"] >= 0.90,
        ],
        [
            "strong",
            "moderate",
        ],
        default="weak",
    )

    profiles["opportunity_class"] = np.select(
        [
            profiles["opportunity_index"] >= 75,
            profiles["opportunity_index"] >= 45,
        ],
        [
            "high",
            "moderate",
        ],
        default="low",
    )

    profiles["risk_class"] = np.select(
        [
            profiles["risk_index"] >= 75,
            profiles["risk_index"] >= 45,
        ],
        [
            "high",
            "moderate",
        ],
        default="low",
    )

    profiles["segment_name"] = profiles.apply(
        _segment_name,
        axis=1,
    )

    return profiles


def _segment_name(row: pd.Series) -> str:
    if (
        row["risk_class"] == "high"
        and row["opportunity_class"] == "high"
    ):
        return "high_opportunity_high_risk"

    if (
        row["fulfillment_class"] == "strong"
        and row["risk_class"] == "low"
    ):
        return "high_performing_stable"

    if (
        row["growth_class"] == "growing"
        and row["opportunity_class"] in {
            "moderate",
            "high",
        }
    ):
        return "growing_opportunity"

    if row["volatility_class"] == "high":
        return "volatile_demand"

    if row["risk_class"] == "high":
        return "operational_risk"

    if row["opportunity_class"] == "high":
        return "high_opportunity"

    if row["growth_class"] == "declining":
        return "declining_demand"

    return "balanced"


def build_outlet_segments(
    performance: pd.DataFrame,
    min_clusters: int = 2,
    max_clusters: int = 8,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict,
]:
    clustered, evaluation, metadata = (
        fit_outlet_clusters(
            performance,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )
    )

    profiles = build_cluster_profiles(
        clustered
    )

    segment_map = profiles[
        [
            "cluster_id",
            "segment_name",
        ]
    ]

    clustered = clustered.merge(
        segment_map,
        on="cluster_id",
        how="left",
        validate="many_to_one",
    )

    return (
        clustered,
        profiles,
        evaluation,
        metadata,
    )

