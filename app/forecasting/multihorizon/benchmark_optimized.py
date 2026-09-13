from __future__ import annotations

import time

import pandas as pd

from app.forecasting.multihorizon.optimized_service import (
    OptimizedMultiHorizonForecastService,
)


SALES_PATH = "data/synthetic/sales.csv"

CUTOFF_DATE = "2025-12-02"
FORECAST_START = "2025-12-03"


def main():

    print("=" * 72)
    print(" OPTIMIZED MULTI-HORIZON PERFORMANCE BENCHMARK")
    print("=" * 72)

    sales = pd.read_csv(SALES_PATH)

    sales["date"] = pd.to_datetime(
        sales["date"]
    )

    history = sales[
        sales["date"] < pd.Timestamp(CUTOFF_DATE)
    ].copy()

    pairs = (
        history[
            ["outlet_id", "product_id"]
        ]
        .drop_duplicates()
    )

    print(
        f"Sales rows       : {len(sales):,}"
    )

    print(
        f"History rows     : {len(history):,}"
    )

    print(
        f"Series count     : {len(pairs):,}"
    )

    print(
        f"Forecast horizon : D+7"
    )

    service = (
        OptimizedMultiHorizonForecastService()
    )

    print()
    print("Training XGBoost...")

    train_start = time.perf_counter()

    service.train(
        sales=sales,
        cutoff_date=CUTOFF_DATE,
    )

    train_seconds = (
        time.perf_counter()
        - train_start
    )

    print(
        f"Training seconds : {train_seconds:.2f}"
    )

    print()
    print("Generating full D+7 forecast...")

    forecast_start = time.perf_counter()

    forecast = service.forecast(
        history=history,
        start_date=FORECAST_START,
        horizon=7,
    )

    forecast_seconds = (
        time.perf_counter()
        - forecast_start
    )

    print()
    print("-" * 72)

    print(
        f"Forecast rows    : {len(forecast):,}"
    )

    print(
        f"Forecast seconds : {forecast_seconds:.2f}"
    )

    print(
        f"Forecast minutes : "
        f"{forecast_seconds / 60:.2f}"
    )

    print(
        f"Rows / second    : "
        f"{len(forecast) / forecast_seconds:,.2f}"
    )

    print(
        f"Series / second  : "
        f"{len(pairs) / forecast_seconds:,.2f}"
    )

    print("-" * 72)

    expected_rows = (
        len(pairs) * 7
    )

    print(
        f"Expected rows    : {expected_rows:,}"
    )

    if len(forecast) != expected_rows:
        raise RuntimeError(
            "Forecast row count mismatch."
        )

    if forecast["prediction"].isna().any():
        raise RuntimeError(
            "NaN predictions detected."
        )

    if (forecast["prediction"] < 0).any():
        raise RuntimeError(
            "Negative predictions detected."
        )

    print()
    print("PERFORMANCE BENCHMARK: PASSED")


if __name__ == "__main__":
    main()