from __future__ import annotations

import numpy as np
import pandas as pd

from app.forecasting.multihorizon.forecast import (
    MultiHorizonForecastService,
)
from app.forecasting.multihorizon.optimized_service import (
    OptimizedMultiHorizonForecastService,
)


SALES_PATH = "data/synthetic/sales.csv"

CUTOFF_DATE = "2025-12-02"
FORECAST_START = "2025-12-03"


def test_optimized_matches_original_for_single_series():

    sales = pd.read_csv(SALES_PATH)

    sales["date"] = pd.to_datetime(
        sales["date"]
    )

    # Use one representative series first.
    pair = (
        sales[
            ["outlet_id", "product_id"]
        ]
        .drop_duplicates()
        .iloc[0]
    )

    outlet_id = pair["outlet_id"]
    product_id = pair["product_id"]

    history = sales[
        (sales["outlet_id"] == outlet_id)
        & (sales["product_id"] == product_id)
        & (
            sales["date"]
            < pd.Timestamp(CUTOFF_DATE)
        )
    ].copy()

    original = MultiHorizonForecastService()

    original.train(
        sales=sales,
        cutoff_date=CUTOFF_DATE,
    )

    original_prediction = (
        original.forecast_series(
            history=history,
            outlet_id=outlet_id,
            product_id=product_id,
            start_date=FORECAST_START,
            horizon=7,
        )
    )

    optimized = (
        OptimizedMultiHorizonForecastService()
    )

    optimized.train(
        sales=sales,
        cutoff_date=CUTOFF_DATE,
    )

    optimized_prediction = (
        optimized.forecast(
            history=history,
            start_date=FORECAST_START,
            horizon=7,
        )
    )

    original_values = (
        original_prediction[
            "prediction"
        ].to_numpy()
    )

    optimized_values = (
        optimized_prediction[
            "prediction"
        ].to_numpy()
    )

    assert len(original_values) == 7
    assert len(optimized_values) == 7

    np.testing.assert_allclose(
        optimized_values,
        original_values,
        rtol=1e-5,
        atol=1e-4,
    )