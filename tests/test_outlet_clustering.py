
import pandas as pd
import pytest

from app.intelligence.outlet.clustering import (
    CLUSTER_FEATURES,
    build_clustering_matrix,
    build_cluster_profiles,
    build_outlet_segments,
    evaluate_kmeans,
)


def sample_performance():
    rows = []

    for i in range(12):
        rows.append(
            {
                "outlet_id": f"O{i:03d}",
                "average_daily_demand": 100 + i * 20,
                "coefficient_variation": (
                    0.05
                    if i < 6
                    else 0.30
                ),
                "peak_intensity": (
                    1.2
                    if i < 6
                    else 1.8
                ),
                "demand_growth_rate": (
                    0.15
                    if i % 3 == 0
                    else 0.02
                ),
                "fulfillment_rate": (
                    0.98
                    if i < 6
                    else 0.85
                ),
                "lost_demand_rate": (
                    0.02
                    if i < 6
                    else 0.15
                ),
                "stockout_product_day_rate": (
                    0.05
                    if i < 6
                    else 0.30
                ),
                "performance_score": (
                    90.0
                    if i < 6
                    else 60.0
                ),
                "opportunity_index": (
                    20.0
                    if i < 6
                    else 85.0
                ),
                "risk_index": (
                    15.0
                    if i < 6
                    else 90.0
                ),
            }
        )

    return pd.DataFrame(rows)


def test_required_features_exist():
    df = sample_performance()

    matrix, raw, scaler = (
        build_clustering_matrix(df)
    )

    assert list(matrix.columns) == CLUSTER_FEATURES
    assert raw.shape == matrix.shape
    assert scaler is not None


def test_matrix_is_standardized():
    df = sample_performance()

    matrix, _, _ = (
        build_clustering_matrix(df)
    )

    assert matrix.mean().abs().max() < 1e-8
    assert (
        (matrix.std(ddof=0) - 1.0)
        .abs()
        .max()
        < 1e-8
    )


def test_evaluate_kmeans():
    df = sample_performance()

    matrix, _, _ = (
        build_clustering_matrix(df)
    )

    evaluation = evaluate_kmeans(
        matrix,
        min_clusters=2,
        max_clusters=4,
    )

    assert len(evaluation) == 3
    assert "silhouette_score" in evaluation
    assert evaluation["n_clusters"].tolist() == [
        2,
        3,
        4,
    ] or set(
        evaluation["n_clusters"]
    ) == {2, 3, 4}


def test_segments_are_generated():
    df = sample_performance()

    clustered, profiles, evaluation, metadata = (
        build_outlet_segments(
            df,
            min_clusters=2,
            max_clusters=4,
        )
    )

    assert len(clustered) == len(df)
    assert not profiles.empty
    assert not evaluation.empty
    assert metadata["best_k"] >= 2


def test_every_outlet_has_one_cluster():
    df = sample_performance()

    clustered, _, _, _ = (
        build_outlet_segments(
            df,
            min_clusters=2,
            max_clusters=4,
        )
    )

    assert (
        clustered["outlet_id"].nunique()
        == len(df)
    )

    assert (
        clustered["cluster_id"].notna().all()
    )


def test_cluster_sizes_are_correct():
    df = sample_performance()

    clustered, _, _, _ = (
        build_outlet_segments(
            df,
            min_clusters=2,
            max_clusters=4,
        )
    )

    actual_sizes = (
        clustered["cluster_id"]
        .value_counts()
    )

    for _, row in clustered.iterrows():
        assert row["cluster_size"] == (
            actual_sizes[
                row["cluster_id"]
            ]
        )


def test_cluster_profiles_have_segment_names():
    df = sample_performance()

    clustered, profiles, _, _ = (
        build_outlet_segments(
            df,
            min_clusters=2,
            max_clusters=4,
        )
    )

    assert "segment_name" in profiles
    assert profiles[
        "segment_name"
    ].notna().all()

    assert clustered[
        "segment_name"
    ].notna().all()


def test_missing_features_raise():
    df = sample_performance().drop(
        columns=["risk_index"]
    )

    with pytest.raises(ValueError):
        build_clustering_matrix(df)

