from __future__ import annotations

import pandas as pd

from app.forecasting.multihorizon.forecast import (
    MultiHorizonForecastService,
)


class MultiHorizonService:

    SUPPORTED_HORIZONS = (1, 3, 7)

    def __init__(self):
        self.forecaster = (
            MultiHorizonForecastService()
        )

    def train(
        self,
        sales: pd.DataFrame,
        cutoff_date: pd.Timestamp | str,
    ) -> None:

        self.forecaster.train(
            sales=sales,
            cutoff_date=cutoff_date,
        )

    def forecast(
        self,
        history: pd.DataFrame,
        start_date: pd.Timestamp | str,
        horizon: int,
    ) -> pd.DataFrame:

        if horizon not in self.SUPPORTED_HORIZONS:
            raise ValueError(
                f"Unsupported horizon: {horizon}. "
                f"Supported: {self.SUPPORTED_HORIZONS}"
            )

        return self.forecaster.forecast(
            history=history,
            start_date=start_date,
            horizon=horizon,
        )

    def forecast_all_horizons(
        self,
        history: pd.DataFrame,
        start_date: pd.Timestamp | str,
    ) -> dict[int, pd.DataFrame]:

        results = {}

        # Generate D+7 once.
        # It already contains D+1 ... D+7.
        d7 = self.forecast(
            history=history,
            start_date=start_date,
            horizon=7,
        )

        for horizon in self.SUPPORTED_HORIZONS:

            results[horizon] = d7[
                d7["horizon"] <= horizon
            ].copy()

        return results

    @staticmethod
    def summarize(
        forecasts: dict[int, pd.DataFrame],
    ) -> pd.DataFrame:

        rows = []

        for horizon, dataframe in forecasts.items():

            rows.append(
                {
                    "horizon": horizon,
                    "rows": len(dataframe),
                    "total_forecast": float(
                        dataframe["prediction"].sum()
                    ),
                    "mean_daily_forecast": float(
                        dataframe["prediction"].mean()
                    ),
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values("horizon")
            .reset_index(drop=True)
        )