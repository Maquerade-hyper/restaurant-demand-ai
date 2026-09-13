from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.forecasting.multihorizon.forecast import (
    MultiHorizonForecastService,
)
from app.forecasting.multihorizon.evaluation import (
    evaluate_horizons,
)


SALES_PATH = Path(
    "data/synthetic/sales.csv"
)


def main():

    print()
    print("=" * 72)
    print(" MULTI-HORIZON FORECAST EVALUATION")
    print("=" * 72)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    sales = pd.read_csv(
        SALES_PATH
    )

    sales["date"] = pd.to_datetime(
        sales["date"]
    )

    print(
        f"Sales rows       : {len(sales):,}"
    )

    # ---------------------------------------------------------
    # Train / evaluation split
    # ---------------------------------------------------------

    cutoff = pd.Timestamp(
        "2025-12-03"
    )

    history = sales[
        sales["date"] < cutoff
    ].copy()

    actual = sales[
        sales["date"] >= cutoff
    ].copy()

    print(
        f"Training end     : "
        f"{history['date'].max().date()}"
    )

    print(
        f"Evaluation start : "
        f"{actual['date'].min().date()}"
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    service = (
        MultiHorizonForecastService()
    )

    service.train(
        sales=history,
        cutoff_date=cutoff,
    )

    # ---------------------------------------------------------
    # Generate D+7 forecasts
    #
    # One D+7 run contains:
    # D+1 ... D+7
    # ---------------------------------------------------------

    print()
    print("Generating D+7 recursive forecasts...")

    predictions = service.forecast(
        history=history,
        start_date=cutoff,
        horizon=7,
    )

    print(
        f"Forecast rows    : "
        f"{len(predictions):,}"
    )

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    metrics = evaluate_horizons(
        actual=actual,
        predictions=predictions,
    )

    print()
    print("-" * 72)
    print(" HORIZON PERFORMANCE")
    print("-" * 72)

    print(
        metrics.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    # ---------------------------------------------------------
    # Basic acceptance
    # ---------------------------------------------------------

    required_horizons = {
        1,
        3,
        7,
    }

    available_horizons = set(
        metrics["horizon"].tolist()
    )

    if not required_horizons.issubset(
        available_horizons
    ):
        raise RuntimeError(
            "Missing required forecast horizons."
        )

    if (
        metrics["mae"] < 0
    ).any():
        raise RuntimeError(
            "Invalid negative MAE."
        )

    if (
        metrics["rmse"] < 0
    ).any():
        raise RuntimeError(
            "Invalid negative RMSE."
        )

    print()
    print(
        "HORIZON EVALUATION: PASSED"
    )


if __name__ == "__main__":
    main()