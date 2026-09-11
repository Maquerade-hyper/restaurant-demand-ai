import numpy as np


def mae(
    actual,
    predicted,
) -> float:

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return float(
        np.mean(
            np.abs(actual - predicted)
        )
    )


def rmse(
    actual,
    predicted,
) -> float:

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return float(
        np.sqrt(
            np.mean(
                (actual - predicted) ** 2
            )
        )
    )


def smape(
    actual,
    predicted,
) -> float:

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    denominator = (
        np.abs(actual)
        + np.abs(predicted)
    )

    mask = denominator != 0

    if not np.any(mask):
        return 0.0

    return float(
        np.mean(
            2
            * np.abs(
                actual[mask]
                - predicted[mask]
            )
            / denominator[mask]
        )
        * 100
    )


def bias(
    actual,
    predicted,
) -> float:

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return float(
        np.mean(predicted - actual)
    )


def evaluate_forecast(
    actual,
    predicted,
) -> dict:

    return {
        "mae": mae(
            actual,
            predicted,
        ),
        "rmse": rmse(
            actual,
            predicted,
        ),
        "smape": smape(
            actual,
            predicted,
        ),
        "bias": bias(
            actual,
            predicted,
        ),
    }