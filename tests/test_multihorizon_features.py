import pandas as pd
import numpy as np

from app.forecasting.multihorizon.forecast_features import (
    build_forecast_features,
    append_forecast_to_history,
)


def make_history():
    dates = pd.date_range(
        "2025-11-01",
        "2025-12-02",
        freq="D",
    )

    rows = []

    for date in dates:
        rows.append(
            {
                "outlet_id": "O001",
                "product_id": "P001",
                "date": date,
                "quantity_sold": 50.0,
                "unit": "kg",
                "revenue": 500.0,
            }
        )

    return pd.DataFrame(rows)


def test_d1_feature_builder_does_not_use_future_target():

    history = make_history()

    future = pd.DataFrame(
        [
            {
                "outlet_id": "O001",
                "product_id": "P001",
                "date": pd.Timestamp("2025-12-03"),
                "unit": "kg",
                "revenue": np.nan,
            }
        ]
    )

    features = build_forecast_features(
        history=history,
        future_row=future,
    )

    assert len(features) == 1

    assert pd.isna(
        features.iloc[0]["quantity_sold"]
    )

    assert features.iloc[0]["lag_1"] == 50.0
    assert features.iloc[0]["lag_7"] == 50.0
    assert features.iloc[0]["lag_28"] == 50.0


def test_d1_rolling_features_use_history_only():

    history = make_history()

    future = pd.DataFrame(
        [
            {
                "outlet_id": "O001",
                "product_id": "P001",
                "date": pd.Timestamp("2025-12-03"),
                "unit": "kg",
                "revenue": np.nan,
            }
        ]
    )

    features = build_forecast_features(
        history=history,
        future_row=future,
    )

    row = features.iloc[0]

    assert row["rolling_mean_3"] == 50.0
    assert row["rolling_mean_7"] == 50.0
    assert row["rolling_mean_14"] == 50.0
    assert row["rolling_mean_28"] == 50.0


def test_forecast_can_be_appended_as_recursive_history():

    history = make_history()

    future = pd.DataFrame(
        [
            {
                "outlet_id": "O001",
                "product_id": "P001",
                "date": pd.Timestamp("2025-12-03"),
                "unit": "kg",
                "revenue": np.nan,
            }
        ]
    )

    updated = append_forecast_to_history(
        history=history,
        forecast_row=future,
        prediction=65.0,
    )

    row = updated[
        updated["date"]
        == pd.Timestamp("2025-12-03")
    ].iloc[0]

    assert row["quantity_sold"] == 65.0

    assert len(updated) == len(history) + 1


def test_recursive_d2_uses_d1_prediction():

    history = make_history()

    d1 = pd.DataFrame(
        [
            {
                "outlet_id": "O001",
                "product_id": "P001",
                "date": pd.Timestamp("2025-12-03"),
                "unit": "kg",
                "revenue": np.nan,
            }
        ]
    )

    d1_features = build_forecast_features(
        history=history,
        future_row=d1,
    )

    assert d1_features.iloc[0]["lag_1"] == 50.0

    history_with_d1 = append_forecast_to_history(
        history=history,
        forecast_row=d1,
        prediction=65.0,
    )

    d2 = pd.DataFrame(
        [
            {
                "outlet_id": "O001",
                "product_id": "P001",
                "date": pd.Timestamp("2025-12-04"),
                "unit": "kg",
                "revenue": np.nan,
            }
        ]
    )

    d2_features = build_forecast_features(
        history=history_with_d1,
        future_row=d2,
    )

    assert d2_features.iloc[0]["lag_1"] == 65.0

    # D2 must not use its own unknown target.
    assert pd.isna(
        d2_features.iloc[0]["quantity_sold"]
    )