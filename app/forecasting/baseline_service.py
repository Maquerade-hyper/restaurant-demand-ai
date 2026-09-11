import pandas as pd

from app.forecasting.baseline import (
    baseline_forecast,
)


class BaselineForecastService:

    def predict(
        self,
        history: pd.Series,
        horizon: int = 7,
        method: str = "seasonal_naive",
    ) -> list[float]:

        forecast = baseline_forecast(
            history=history,
            horizon=horizon,
            method=method,
        )

        return forecast.tolist()