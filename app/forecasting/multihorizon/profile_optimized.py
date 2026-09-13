from __future__ import annotations

import time

import numpy as np
import pandas as pd

from app.forecasting.multihorizon.optimized_forecast import (
    append_prediction,
    build_batch_feature_matrix,
)
from app.forecasting.xgboost_features import (
    prepare_xgboost_dataset,
)
from app.forecasting.xgboost_service import (
    XGBoostForecastService,
)


SALES_PATH = "data/synthetic/sales.csv"
CUTOFF_DATE = "2025-12-02"
FORECAST_START = "2025-12-03"


def main():

    print("=" * 72)
    print(" OPTIMIZED FORECAST PROFILE")
    print("=" * 72)

    sales = pd.read_csv(SALES_PATH)
    sales["date"] = pd.to_datetime(sales["date"])

    history = sales[
        sales["date"] < pd.Timestamp(CUTOFF_DATE)
    ].copy()

    pairs = (
        history[
            ["outlet_id", "product_id"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    print("Preparing training data...")

    train_start = time.perf_counter()

    train_dataset = prepare_xgboost_dataset(
        history
    )

    train_prepare_seconds = (
        time.perf_counter()
        - train_start
    )

    service = XGBoostForecastService()

    train_start = time.perf_counter()

    service.train(
        train_dataset
    )

    train_seconds = (
        time.perf_counter()
        - train_start
    )

    print(
        f"Training preparation : "
        f"{train_prepare_seconds:.2f}s"
    )

    print(
        f"Training             : "
        f"{train_seconds:.2f}s"
    )

    # ---------------------------------------------------------
    # STATES
    # ---------------------------------------------------------

    states = []
    metadata = []

    state_start = time.perf_counter()

    for _, pair in pairs.iterrows():

        series = history[
            (history["outlet_id"] == pair["outlet_id"])
            & (
                history["product_id"]
                == pair["product_id"]
            )
        ]

        values = (
            series["quantity_sold"]
            .astype(float)
            .to_numpy()
        )

        states.append(
            values[-28:]
        )

        metadata.append(
            (
                pair["outlet_id"],
                pair["product_id"],
            )
        )

    state_seconds = (
        time.perf_counter()
        - state_start
    )

    print(
        f"State preparation    : "
        f"{state_seconds:.2f}s"
    )

    # ---------------------------------------------------------
    # PROFILE 7 FORECAST DAYS
    # ---------------------------------------------------------

    total_feature_seconds = 0.0
    total_prediction_seconds = 0.0
    total_append_seconds = 0.0

    forecast_start = pd.Timestamp(
        FORECAST_START
    )

    for step in range(1, 8):

        forecast_date = (
            forecast_start
            + pd.Timedelta(
                days=step - 1
            )
        )

        feature_start = time.perf_counter()

        X = build_batch_feature_matrix(
            states=states,
            forecast_date=forecast_date,
        )

        feature_seconds = (
            time.perf_counter()
            - feature_start
        )

        total_feature_seconds += (
            feature_seconds
        )

        prediction_start = time.perf_counter()

        predictions = service.model.predict(
            X
        )

        predictions = np.maximum(
            predictions,
            0.0,
        )

        prediction_seconds = (
            time.perf_counter()
            - prediction_start
        )

        total_prediction_seconds += (
            prediction_seconds
        )

        append_start = time.perf_counter()

        for index, prediction in enumerate(
            predictions
        ):
            states[index] = append_prediction(
                states[index],
                prediction,
            )

        append_seconds = (
            time.perf_counter()
            - append_start
        )

        total_append_seconds += (
            append_seconds
        )

        print(
            f"DAY {step}: "
            f"features={feature_seconds:.3f}s "
            f"prediction={prediction_seconds:.3f}s "
            f"append={append_seconds:.3f}s"
        )

    print()
    print("-" * 72)

    print(
        f"Total feature time  : "
        f"{total_feature_seconds:.3f}s"
    )

    print(
        f"Total prediction time: "
        f"{total_prediction_seconds:.3f}s"
    )

    print(
        f"Total append time   : "
        f"{total_append_seconds:.3f}s"
    )

    measured = (
        total_feature_seconds
        + total_prediction_seconds
        + total_append_seconds
    )

    print(
        f"Measured forecast work: "
        f"{measured:.3f}s"
    )

    print("-" * 72)


if __name__ == "__main__":
    main()