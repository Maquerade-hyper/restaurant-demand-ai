import pandas as pd

from app.forecasting.baseline import (
    baseline_forecast,
)
from app.forecasting.metrics import (
    evaluate_forecast,
)


def backtest_baseline(
    series: pd.Series,
    horizon: int = 7,
    method: str = "seasonal_naive",
) -> dict:

    if len(series) <= horizon:
        raise ValueError(
            "Not enough observations for backtesting"
        )

    train = series.iloc[:-horizon]
    actual = series.iloc[-horizon:]

    predicted = baseline_forecast(
        history=train,
        horizon=horizon,
        method=method,
    )

    metrics = evaluate_forecast(
        actual=actual.to_numpy(),
        predicted=predicted,
    )

    return {
        "method": method,
        "horizon": horizon,
        "metrics": metrics,
        "predicted": predicted.tolist(),
        "actual": actual.to_list(),
    }