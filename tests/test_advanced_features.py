
import numpy as np
import pandas as pd

from app.features.trend_features import (
    create_trend_features,
)

from app.features.volatility_features import (
    create_volatility_features,
)

from app.features.order_velocity import (
    create_order_velocity_features,
)

from app.features.seasonality_features import (
    create_seasonality_features,
)


def make_data():

    dates = pd.date_range(
        "2025-01-01",
        periods=40,
        freq="D",
    )

    return pd.DataFrame(
        {
            "outlet_id": ["O001"] * 40,
            "product_id": ["P001"] * 40,
            "date": dates,
            "quantity_sold": np.arange(
                10,
                50,
            ),
            "day_of_week": dates.dayofweek,
            "month": dates.month,
            "quarter": dates.quarter,
            "is_weekend": (
                dates.dayofweek >= 5
            ).astype(int),
        }
    )


def test_trend_features():

    df = make_data()

    result = create_trend_features(
        df
    )

    assert "trend_1d" in result
    assert "trend_7d" in result
    assert "trend_14d" in result

    # Current day must not use
    # current target value.
    assert pd.isna(
        result.iloc[0]["trend_1d"]
    )


def test_volatility_features():

    df = make_data()

    result = create_volatility_features(
        df
    )

    assert (
        "volatility_7"
        in result.columns
    )

    assert (
        "coefficient_variation_14"
        in result.columns
    )


def test_order_velocity():

    df = make_data()

    result = create_order_velocity_features(
        df
    )

    assert (
        "order_velocity_3d"
        in result.columns
    )

    assert (
        "order_velocity_14d"
        in result.columns
    )


def test_seasonality():

    df = make_data()

    result = create_seasonality_features(
        df
    )

    assert (
        "dow_sin_2"
        in result.columns
    )

    assert (
        "month_cos_2"
        in result.columns
    )

    assert np.isfinite(
        result["dow_sin_2"]
    ).all()

