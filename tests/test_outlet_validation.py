
import pandas as pd
import pytest

from app.intelligence.outlet.validation import (
    validate_cross_layer,
    validate_outlet_intelligence,
    validate_performance,
)


def performance_data():
    return pd.DataFrame(
        {
            "outlet_id": [
                "O001",
                "O002",
                "O003",
            ],
            "performance_days": [
                365,
                365,
                365,
            ],
            "total_observed_demand": [
                900.0,
                950.0,
                800.0,
            ],
            "total_deconstrained_demand": [
                1000.0,
                1000.0,
                1000.0,
            ],
            "total_estimated_lost_demand": [
                100.0,
                50.0,
                200.0,
            ],
            "fulfillment_rate": [
                0.90,
                0.95,
                0.80,
            ],
            "lost_demand_rate": [
                0.10,
                0.05,
                0.20,
            ],
            "stockout_product_day_rate": [
                0.10,
                0.05,
                0.20,
            ],
            "performance_score": [
                80.0,
                85.0,
                70.0,
            ],
            "opportunity_index": [
                40.0,
                30.0,
                90.0,
            ],
            "risk_index": [
                45.0,
                25.0,
                95.0,
            ],
            "performance_rank": [
                2,
                1,
                3,
            ],
            "opportunity_rank": [
                2,
                3,
                1,
            ],
            "risk_rank": [
                2,
                3,
                1,
            ],
        }
    )


def common_layers():
    performance = performance_data()

    profiles = pd.DataFrame(
        {
            "outlet_id": [
                "O001",
                "O002",
                "O003",
            ]
        }
    )

    behavior = pd.DataFrame(
        {
            "outlet_id": [
                "O001",
                "O002",
                "O003",
            ],
            "behavior_days": [
                365,
                365,
                365,
            ],
            "behavior_average_daily_demand": [
                100.0,
                100.0,
                100.0,
            ],
            "behavior_coefficient_variation": [
                0.1,
                0.2,
                0.3,
            ],
            "behavior_peak_intensity": [
                1.2,
                1.3,
                1.4,
            ],
        }
    )

    segments = pd.DataFrame(
        {
            "outlet_id": [
                "O001",
                "O002",
                "O003",
            ],
            "cluster_id": [
                0,
                0,
                1,
            ],
            "cluster_size": [
                2,
                2,
                1,
            ],
            "segment_name": [
                "balanced",
                "balanced",
                "high_opportunity",
            ],
        }
    )

    return (
        profiles,
        behavior,
        performance,
        segments,
    )


def test_performance_validation_passes():
    result = validate_performance(
        performance_data()
    )

    assert result["passed"] is True
    assert not result["errors"]


def test_observed_demand_cannot_exceed_demand():
    data = performance_data()

    data.loc[
        0,
        "total_observed_demand",
    ] = 1100.0

    result = validate_performance(
        data
    )

    assert result["passed"] is False
    assert any(
        "exceeds deconstrained demand"
        in error
        for error in result["errors"]
    )


def test_fulfillment_must_reconcile():
    data = performance_data()

    data.loc[
        0,
        "fulfillment_rate",
    ] = 0.50

    result = validate_performance(
        data
    )

    assert result["passed"] is False
    assert any(
        "fulfillment rate"
        in error
        for error in result["errors"]
    )


def test_cross_layer_requires_same_outlets():
    profiles, behavior, performance, segments = (
        common_layers()
    )

    behavior.loc[
        0,
        "outlet_id",
    ] = "O999"

    result = validate_cross_layer(
        profiles,
        behavior,
        performance,
        segments,
    )

    assert result["passed"] is False


def test_complete_validation_passes():
    layers = common_layers()

    result = validate_outlet_intelligence(
        *layers
    )

    assert result["passed"] is True
    assert result["outlet_count"] == 3


def test_duplicate_outlets_fail():
    data = performance_data()

    duplicate = pd.concat(
        [data, data.iloc[[0]]],
        ignore_index=True,
    )

    result = validate_performance(
        duplicate
    )

    assert result["passed"] is False


def test_out_of_range_index_fails():
    data = performance_data()

    data.loc[
        0,
        "risk_index",
    ] = 101.0

    result = validate_performance(
        data
    )

    assert result["passed"] is False


def test_missing_required_column_fails():
    data = performance_data().drop(
        columns=["opportunity_index"]
    )

    result = validate_performance(
        data
    )

    assert result["passed"] is False


def test_cluster_size_mismatch_fails():
    profiles, behavior, performance, segments = (
        common_layers()
    )

    segments.loc[
        0,
        "cluster_size",
    ] = 99

    result = validate_outlet_intelligence(
        profiles,
        behavior,
        performance,
        segments,
    )

    assert result["passed"] is False

