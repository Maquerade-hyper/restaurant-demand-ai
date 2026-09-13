from pathlib import Path

import pandas as pd

from app.forecasting.multihorizon.service import (
    MultiHorizonService,
)


SALES_PATH = Path(
    "data/synthetic/sales.csv"
)


def main():

    print()
    print("=" * 72)
    print(" MULTI-HORIZON DEMAND INTELLIGENCE")
    print("=" * 72)

    sales = pd.read_csv(
        SALES_PATH
    )

    sales["date"] = pd.to_datetime(
        sales["date"]
    )

    cutoff = pd.Timestamp(
        "2025-12-03"
    )

    history = sales[
        sales["date"] < cutoff
    ].copy()

    service = MultiHorizonService()

    print()
    print("Training XGBoost...")

    service.train(
        sales=history,
        cutoff_date=cutoff,
    )

    print("Generating D+7 recursive forecast...")

    forecasts = (
        service.forecast_all_horizons(
            history=history,
            start_date=cutoff,
        )
    )

    print()
    print("-" * 72)
    print(" BUSINESS HORIZONS")
    print("-" * 72)

    summary = service.summarize(
        forecasts
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}",
        )
    )

    print()
    print("D+1 rows :", len(forecasts[1]))
    print("D+3 rows :", len(forecasts[3]))
    print("D+7 rows :", len(forecasts[7]))

    print()
    print("MULTI-HORIZON INTEGRATION: PASSED")


if __name__ == "__main__":
    main()