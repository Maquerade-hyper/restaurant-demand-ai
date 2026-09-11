import pandas as pd

from app.features.feature_pipeline import build_features


def test_feature_pipeline():

    dates = pd.date_range(
        "2025-01-01",
        periods=40,
    )

    df = pd.DataFrame(
        {
            "outlet_id": ["O001"] * 40,
            "product_id": ["P001"] * 40,
            "date": dates,
            "quantity_sold": range(1, 41),
        }
    )

    result = build_features(df)

    assert "year" in result.columns
    assert "day_of_week" in result.columns
    assert "lag_1" in result.columns
    assert "lag_7" in result.columns
    assert "rolling_mean_7" in result.columns
    assert "rolling_std_7" in result.columns

    # Leakage check:
    # today's lag_1 must equal yesterday's target.
    assert result.iloc[1]["lag_1"] == 1

    # First row cannot have historical lag.
    assert pd.isna(result.iloc[0]["lag_1"])