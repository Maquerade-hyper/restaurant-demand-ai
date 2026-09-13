from __future__ import annotations

import numpy as np
import pandas as pd

from app.intelligence.multimodal.benchmark import (
    build_series_splits,
    build_temporal_features,
    build_visual_proxy_features,
    discover_context_columns,
    evaluate,
    evaluate_ablation,
    fit_predict,
    validate,
)


def make_frame(days: int = 120):
    dates = pd.date_range(
        "2025-01-01",
        periods=days,
        freq="D",
    )

    rows = []

    for outlet in ["O001", "O002"]:
        for product in ["P001", "P002"]:
            base = (
                20
                if outlet == "O001"
                else 35
            )

            product_factor = (
                1
                if product == "P001"
                else 1.5
            )

            for i, date in enumerate(dates):
                demand = (
                    base
                    * product_factor
                    + 2 * np.sin(i / 7)
                    + (5 if date.dayofweek >= 5 else 0)
                )

                rows.append(
                    {
                        "date": date,
                        "outlet_id": outlet,
                        "product_id": product,
                        "deconstrained_demand": max(
                            0,
                            demand,
                        ),
                        "stockout": 0,
                        "holiday_active": int(
                            date.day == 15
                        ),
                        "promotion_active": int(
                            date.dayofweek == 4
                        ),
                        "outlet_type": (
                            "restaurant"
                            if outlet == "O001"
                            else "bar"
                        ),
                        "location_type": (
                            "business"
                            if outlet == "O001"
                            else "tourist"
                        ),
                        "country": "India",
                        "product_category": (
                            "food"
                            if product == "P001"
                            else "beverage"
                        ),
                    }
                )

    return pd.DataFrame(rows)


def test_temporal_features_do_not_use_current_target():
    df = make_frame()

    result = build_temporal_features(
        df,
        "deconstrained_demand",
    )

    assert "history_lag_1" in result.columns
    assert "history_mean_7" in result.columns

    first = result.sort_values(
        ["outlet_id", "product_id", "date"]
    ).iloc[0]

    assert pd.isna(first["history_lag_1"])


def test_temporal_features_are_finite_after_warmup():
    df = make_frame()

    result = build_temporal_features(
        df,
        "deconstrained_demand",
    )

    numeric = result.select_dtypes(
        include=np.number
    )

    values = numeric.to_numpy()

    assert np.all(
        np.isfinite(
            np.nan_to_num(values)
        )
    )


def test_series_split_is_chronological():
    df = make_frame()

    series = df[
        (df["outlet_id"] == "O001")
        &
        (df["product_id"] == "P001")
    ].copy()

    train, validation, test = build_series_splits(
        series,
        validation_days=20,
        test_days=20,
    )

    assert train["date"].max() < validation["date"].min()
    assert validation["date"].max() < test["date"].min()


def test_series_split_has_disjoint_dates():
    df = make_frame()

    series = df[
        (df["outlet_id"] == "O001")
        &
        (df["product_id"] == "P001")
    ].copy()

    train, validation, test = build_series_splits(
        series,
        validation_days=20,
        test_days=20,
    )

    train_dates = set(train["date"])
    val_dates = set(validation["date"])
    test_dates = set(test["date"])

    assert train_dates.isdisjoint(val_dates)
    assert train_dates.isdisjoint(test_dates)
    assert val_dates.isdisjoint(test_dates)


def test_context_discovery_excludes_target():
    df = make_frame()

    numeric, text = discover_context_columns(
        df
    )

    assert "deconstrained_demand" not in numeric
    assert "deconstrained_demand" not in text


def test_context_discovery_finds_real_context():
    df = make_frame()

    numeric, text = discover_context_columns(
        df
    )

    assert "stockout" in numeric
    assert "outlet_type" in text
    assert "location_type" in text


def test_visual_proxy_shape():
    df = make_frame(
        days=10
    )

    features = build_visual_proxy_features(
        df
    )

    assert len(features) == len(df)
    assert features.shape[1] >= 2

    assert np.all(
        np.isfinite(
            features.to_numpy(
                dtype=float
            )
        )
    )


def test_evaluate_returns_required_metrics():
    actual = np.array(
        [10.0, 12.0, 14.0]
    )

    prediction = np.array(
        [11.0, 11.0, 15.0]
    )

    metrics = evaluate(
        actual,
        prediction,
    )

    required = {
        "mae",
        "rmse",
        "smape",
        "bias",
        "high_demand_mae",
        "spike_recall",
    }

    assert required.issubset(
        metrics.keys()
    )


def test_evaluate_ablation_rejects_shape_mismatch():
    actual = np.array(
        [1.0, 2.0]
    )

    prediction = np.array(
        [1.0]
    )

    try:
        evaluate_ablation(
            actual,
            prediction,
        )
    except ValueError:
        return

    raise AssertionError(
        "Shape mismatch was not rejected"
    )


def test_evaluate_ablation_rejects_nonfinite_predictions():
    actual = np.array(
        [1.0, 2.0]
    )

    prediction = np.array(
        [1.0, np.nan]
    )

    try:
        evaluate_ablation(
            actual,
            prediction,
        )
    except ValueError:
        return

    raise AssertionError(
        "Non-finite predictions were not rejected"
    )


def test_validation_contract():
    results = pd.DataFrame(
        {
            "mode": [
                "numeric",
                "numeric_text",
                "numeric_visual",
                "multimodal",
            ],
            "mae": [
                1.0,
                0.9,
                0.95,
                0.85,
            ],
            "rmse": [
                1.2,
                1.1,
                1.15,
                1.0,
            ],
            "smape": [
                10,
                9,
                9.5,
                8,
            ],
            "bias": [
                0,
                0,
                0,
                0,
            ],
            "high_demand_mae": [
                2,
                1.8,
                1.9,
                1.7,
            ],
            "spike_recall": [
                0.4,
                0.5,
                0.45,
                0.6,
            ],
        }
    )

    result = validate(
        results
    )

    assert result["passed"] is True
    assert result["errors"] == []


def test_validation_rejects_missing_mode():
    results = pd.DataFrame(
        {
            "mae": [1.0],
            "rmse": [1.0],
            "smape": [1.0],
            "bias": [0.0],
            "high_demand_mae": [1.0],
            "spike_recall": [0.5],
        }
    )

    result = validate(
        results
    )

    assert result["passed"] is False


def test_prediction_is_nonnegative():
    df = make_frame(
        days=120
    )

    df = build_temporal_features(
        df,
        "deconstrained_demand",
    )

    series = df[
        (df["outlet_id"] == "O001")
        &
        (df["product_id"] == "P001")
    ].copy()

    train, validation, test = build_series_splits(
        series,
        validation_days=15,
        test_days=15,
    )

    numeric = ["stockout"]

    text = [
        "outlet_type",
        "location_type",
        "country",
        "product_category",
        "holiday_active",
        "promotion_active",
    ]

    _, predictions = fit_predict(
        train=train,
        validation=validation,
        test=test,
        target="deconstrained_demand",
        mode="numeric",
        numeric_context_columns=numeric,
        text_columns=text,
    )

    assert np.all(
        predictions >= 0
    )


def test_prediction_is_finite():
    df = make_frame(
        days=120
    )

    df = build_temporal_features(
        df,
        "deconstrained_demand",
    )

    series = df[
        (df["outlet_id"] == "O001")
        &
        (df["product_id"] == "P001")
    ].copy()

    train, validation, test = build_series_splits(
        series,
        validation_days=15,
        test_days=15,
    )

    _, predictions = fit_predict(
        train=train,
        validation=validation,
        test=test,
        target="deconstrained_demand",
        mode="multimodal",
        numeric_context_columns=[
            "stockout"
        ],
        text_columns=[
            "outlet_type",
            "location_type",
            "country",
            "product_category",
            "holiday_active",
            "promotion_active",
        ],
    )

    assert np.all(
        np.isfinite(predictions)
    )


def test_all_ablation_modes_execute():
    df = make_frame(
        days=120
    )

    df = build_temporal_features(
        df,
        "deconstrained_demand",
    )

    series = df[
        (df["outlet_id"] == "O001")
        &
        (df["product_id"] == "P001")
    ].copy()

    train, validation, test = build_series_splits(
        series,
        validation_days=15,
        test_days=15,
    )

    for mode in [
        "numeric",
        "numeric_text",
        "numeric_visual",
        "multimodal",
    ]:
        validation_pred, test_pred = fit_predict(
            train=train,
            validation=validation,
            test=test,
            target="deconstrained_demand",
            mode=mode,
            numeric_context_columns=[
                "stockout"
            ],
            text_columns=[
                "outlet_type",
                "location_type",
                "country",
                "product_category",
                "holiday_active",
                "promotion_active",
            ],
        )

        assert len(validation_pred) == len(validation)
        assert len(test_pred) == len(test)

        assert np.all(
            np.isfinite(validation_pred)
        )

        assert np.all(
            np.isfinite(test_pred)
        )


def test_forecast_time_firewall_rejects_part16_outputs():
    from app.intelligence.multimodal.benchmark import (
        discover_context_columns,
    )

    df = make_frame(
        days=20
    )

    df["estimated_lost_demand"] = 10.0
    df["recovery_ratio"] = 1.2
    df["clean_reference_demand"] = 20.0
    df["censoring_strength"] = 0.5

    numeric, text = (
        discover_context_columns(
            df
        )
    )

    forbidden = {
        "estimated_lost_demand",
        "recovery_ratio",
        "clean_reference_demand",
        "censoring_strength",
    }

    assert forbidden.isdisjoint(
        set(numeric)
    )

    assert forbidden.isdisjoint(
        set(text)
    )


def test_known_context_is_discovered():
    from app.intelligence.multimodal.benchmark import (
        discover_context_columns,
    )

    df = make_frame(
        days=20
    )

    numeric, text = (
        discover_context_columns(
            df
        )
    )

    assert (
        "holiday_active"
        in numeric
    )

    assert (
        "promotion_active"
        in numeric
    )

    assert (
        "outlet_type"
        in text
    )

    assert (
        "location_type"
        in text
    )


def test_context_interactions_are_finite():
    from app.intelligence.multimodal.benchmark import (
        build_context_interactions,
    )

    df = make_frame(
        days=20
    )

    features = (
        build_context_interactions(
            df
        )
    )

    assert len(features) == len(df)

    values = features.to_numpy(
        dtype=float
    )

    assert np.all(
        np.isfinite(values)
    )


def test_visual_proxy_is_deterministic():
    from app.intelligence.multimodal.benchmark import (
        build_visual_proxy_features,
    )

    df = make_frame(
        days=20
    )

    first = (
        build_visual_proxy_features(
            df
        )
    )

    second = (
        build_visual_proxy_features(
            df
        )
    )

    assert np.allclose(
        first.to_numpy(
            dtype=float
        ),
        second.to_numpy(
            dtype=float
        ),
    )


def test_v3_multimodal_predictions_are_finite():
    from app.intelligence.multimodal.benchmark import (
        build_temporal_features,
        build_series_splits,
        fit_predict,
    )

    df = make_frame(
        days=120
    )

    df = build_temporal_features(
        df,
        "deconstrained_demand",
    )

    series = df[
        (df["outlet_id"] == "O001")
        &
        (df["product_id"] == "P001")
    ].copy()

    train, validation, test = (
        build_series_splits(
            series,
            validation_days=15,
            test_days=15,
        )
    )

    validation_prediction, test_prediction = (
        fit_predict(
            train=train,
            validation=validation,
            test=test,
            target="deconstrained_demand",
            mode="multimodal",
            numeric_context_columns=[
                "stockout",
                "holiday_active",
                "promotion_active",
            ],
            text_columns=[
                "outlet_type",
                "location_type",
                "country",
                "product_category",
            ],
        )
    )

    assert np.all(
        np.isfinite(
            validation_prediction
        )
    )

    assert np.all(
        np.isfinite(
            test_prediction
        )
    )

    assert np.all(
        validation_prediction
        >= 0
    )

    assert np.all(
        test_prediction
        >= 0
    )