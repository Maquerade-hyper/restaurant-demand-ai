from __future__ import annotations

import math


SERVICE_LEVEL_Z = {
    0.80: 0.8416,
    0.85: 1.0364,
    0.90: 1.2816,
    0.95: 1.6449,
    0.975: 1.9600,
    0.99: 2.3263,
}


def calculate_safety_stock(
    demand_std: float,
    lead_time_days: int,
    service_level: float = 0.95,
) -> float:

    if demand_std < 0:
        raise ValueError(
            "demand_std cannot be negative."
        )

    if lead_time_days < 0:
        raise ValueError(
            "lead_time_days cannot be negative."
        )

    if service_level not in SERVICE_LEVEL_Z:
        raise ValueError(
            "Unsupported service level. "
            f"Choose from: {list(SERVICE_LEVEL_Z)}"
        )

    z = SERVICE_LEVEL_Z[
        service_level
    ]

    return float(
        z
        * demand_std
        * math.sqrt(
            lead_time_days
        )
    )