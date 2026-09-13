from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features.advanced_feature_pipeline import (
    AdvancedFeaturePipeline,
)


def build_data() -> pd.DataFrame:

    dates = pd.date_range(
        "2025-01-01",
        "2025-01-30",
        freq="D",
    )

    rows = []

    for outlet_id in [
        "O001",
        "O002",
    ]:

        for product_id in [
            "P001",
            "P002",
        ]:

            for date in dates:

                rows.append(
                    {
                        "outlet_id": outlet_id,
                        "product_id": product_id,
                        "date": date,
                        "quantity_sold": float(
                            20
                            + date.day % 10
                        ),
                        "revenue": 100.0,
                        "temperature": 25.0,
                        "temperature_squared": 625.0,
                        "is_rainy": 0.0,
                        "holiday_active": 0.0,
                        "promotion_active": 0.0,
                        "event_active": 0.0,
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# BASIC PIPELINE CONTRACT
# ============================================================

def test_pipeline_generates_features():

    df = build_data()

    pipeline = AdvancedFeaturePipeline()

    result = pipeline.transform(df)

    assert not result.empty

    assert len(result) == len(df)

    expected = {
        "advanced_dow_sin_3",
        "advanced_weekend_month",
        "advanced_demand_lag_1",
        "advanced_mean_7",
        "advanced_momentum_3_7",
        "advanced_acceleration",
        "advanced_persistence",
        "advanced_outlet_mean_demand",
        "advanced_product_mean_demand",
        "advanced_outlet_product_affinity",
        "advanced_temperature_rain",
        "advanced_holiday_rain",
        "advanced_context_intensity",
    }

    missing = (
        expected
        -
        set(result.columns)
    )

    assert not missing, (
        "Missing expected features: "
        + ", ".join(sorted(missing))
    )


# ============================================================
# ROW COUNT
# ============================================================

def test_pipeline_preserves_row_count():

    df = build_data()

    pipeline = AdvancedFeaturePipeline()

    result = pipeline.transform(df)

    assert len(result) == len(df)


# ============================================================
# REQUIRED IDENTIFIERS
# ============================================================

def test_pipeline_preserves_identifiers():

    df = build_data()

    pipeline = AdvancedFeaturePipeline()

    result = pipeline.transform(df)

    assert "outlet_id" in result.columns
    assert "product_id" in result.columns
    assert "date" in result.columns


# ============================================================
# TARGET IS RETAINED
# ============================================================

def test_target_is_retained():

    df = build_data()

    pipeline = AdvancedFeaturePipeline()

    result = pipeline.transform(df)

    assert "quantity_sold" in result.columns


# ============================================================
# FORBIDDEN RAW COLUMNS ARE REMOVED
# ============================================================

def test_forbidden_columns_are_removed():

    df = build_data()

    pipeline = AdvancedFeaturePipeline()

    result = pipeline.transform(df)

    forbidden = {
        "revenue",
        "actual_demand",
        "true_demand",
        "lost_demand",
        "future_demand",
        "closing_stock",
        "demand_truth",
        "lost_demand_truth",
        "prediction",
        "target",
    }

    assert not (
        forbidden
        &
        set(result.columns)
    )


# ============================================================
# FEATURE MATRIX IS NUMERIC
# ============================================================

def test_feature_matrix_is_numeric():

    df = build_data()

    pipeline = AdvancedFeaturePipeline()

    features = pipeline.build_feature_matrix(
        df
    )

    assert not features.empty

    for column in features.columns:

        assert pd.api.types.is_numeric_dtype(
            features[column]
        )


# ============================================================
# TRUE CAUSALITY TEST
# ============================================================

def test_changing_current_target_does_not_change_past_features():

    df1 = build_data()

    df2 = df1.copy()

    changed_date = pd.Timestamp(
        "2025-01-20"
    )

    # Change demand only on the selected date.
    df2.loc[
        df2["date"] == changed_date,
        "quantity_sold",
    ] += 1000.0

    pipeline = AdvancedFeaturePipeline()

    result1 = pipeline.transform(
        df1
    )

    result2 = pipeline.transform(
        df2
    )

    feature_columns = [
        "advanced_demand_lag_1",
        "advanced_demand_lag_7",
        "advanced_mean_7",
        "advanced_mean_14",
        "advanced_mean_28",
    ]

    # --------------------------------------------------------
    # Causal contract:
    #
    # Changing target at T must NOT change any feature
    # generated for dates before T.
    #
    # Features after T are allowed to change because T
    # legitimately becomes historical information.
    # --------------------------------------------------------

    mask = (
        result1["date"]
        <
        changed_date
    )

    for column in feature_columns:

        a = result1.loc[
            mask,
            column,
        ].to_numpy()

        b = result2.loc[
            mask,
            column,
        ].to_numpy()

        assert np.allclose(
            a,
            b,
            equal_nan=True,
        )


# ============================================================
# VALIDATION CONTRACT
# ============================================================

def test_pipeline_validation_passes():

    df = build_data()

    pipeline = AdvancedFeaturePipeline()

    result = pipeline.transform(df)

    validation = pipeline.validate(
        result
    )

    assert validation["passed"] is True

    assert validation["errors"] == []

    assert validation["rows"] == len(df)

    assert validation["feature_count"] > 0