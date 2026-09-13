from __future__ import annotations

import numpy as np


def calculate_lead_time_demand(
    daily_forecast,
    lead_time_days: int,
) -> float:

    if lead_time_days < 0:
        raise ValueError(
            "lead_time_days cannot be negative."
        )

    forecast = np.asarray(
        daily_forecast,
        dtype=float,
    )

    if lead_time_days == 0:
        return 0.0

    if len(forecast) < lead_time_days:
        raise ValueError(
            "Not enough forecast values "
            "for the requested lead time."
        )

    return float(
        np.sum(
            forecast[:lead_time_days]
        )
    )