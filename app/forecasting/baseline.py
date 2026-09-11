import pandas as pd
import numpy as np


def naive_forecast(
    history: pd.Series,
    horizon: int,
) -> np.ndarray:

    if len(history) == 0:
        raise ValueError("History cannot be empty")

    value = float(history.iloc[-1])

    return np.repeat(
        max(value, 0.0),
        horizon,
    )


def seasonal_naive_forecast(
    history: pd.Series,
    horizon: int,
    season_length: int = 7,
) -> np.ndarray:

    if len(history) < season_length:
        return naive_forecast(
            history,
            horizon,
        )

    values = history.iloc[-season_length:].to_numpy()

    repeats = int(
        np.ceil(horizon / season_length)
    )

    forecast = np.tile(
        values,
        repeats,
    )

    return np.maximum(
        forecast[:horizon],
        0.0,
    )


def moving_average_forecast(
    history: pd.Series,
    horizon: int,
    window: int = 7,
) -> np.ndarray:

    if len(history) == 0:
        raise ValueError("History cannot be empty")

    window = min(
        window,
        len(history),
    )

    value = float(
        history.iloc[-window:].mean()
    )

    return np.repeat(
        max(value, 0.0),
        horizon,
    )


def baseline_forecast(
    history: pd.Series,
    horizon: int,
    method: str = "seasonal_naive",
) -> np.ndarray:

    if method == "naive":
        return naive_forecast(
            history,
            horizon,
        )

    if method == "moving_average":
        return moving_average_forecast(
            history,
            horizon,
        )

    if method == "seasonal_naive":
        return seasonal_naive_forecast(
            history,
            horizon,
        )

    raise ValueError(
        f"Unknown baseline method: {method}"
    )