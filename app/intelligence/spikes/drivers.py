from __future__ import annotations

import numpy as np
import pandas as pd


CONTEXT_COLUMNS = [
    "holiday_active",
    "promotion_active",
    "event_active",
    "is_rainy",
]


def analyze_spike_drivers(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Part 23C - Spike Driver Intelligence.

    Identifies contextual signals associated with demand spikes.

    These are associations/signals only.
    They are NOT causal claims.
    """

    if df.empty:
        return df.copy()

    required = {
        "is_demand_spike",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    out = df.copy()

    # ------------------------------------------------------------
    # Normalize contextual columns
    # ------------------------------------------------------------

    for column in CONTEXT_COLUMNS:

        if column not in out.columns:
            out[column] = 0.0

        out[column] = (
            pd.to_numeric(
                out[column],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(
                lower=0.0,
                upper=1.0,
            )
        )

    # ------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------

    if "temperature" not in out.columns:
        out["temperature"] = np.nan

    out["temperature"] = pd.to_numeric(
        out["temperature"],
        errors="coerce",
    )

    spike = out["is_demand_spike"].astype(bool)

    # ------------------------------------------------------------
    # Context signals coinciding with spikes
    # ------------------------------------------------------------

    out["spike_holiday_signal"] = (
        (out["holiday_active"] > 0)
        & spike
    )

    out["spike_promotion_signal"] = (
        (out["promotion_active"] > 0)
        & spike
    )

    out["spike_event_signal"] = (
        (out["event_active"] > 0)
        & spike
    )

    out["spike_weather_signal"] = (
        (out["is_rainy"] > 0)
        & spike
    )

    # ------------------------------------------------------------
    # Number of simultaneous contexts
    # ------------------------------------------------------------

    signal_columns = [
        "spike_holiday_signal",
        "spike_promotion_signal",
        "spike_event_signal",
        "spike_weather_signal",
    ]

    out["spike_context_signal_count"] = (
        out[signal_columns]
        .astype(int)
        .sum(axis=1)
    )

    # ------------------------------------------------------------
    # Context intensity
    # ------------------------------------------------------------

    out["spike_context_intensity"] = (
        out[
            [
                "holiday_active",
                "promotion_active",
                "event_active",
                "is_rainy",
            ]
        ]
        .mean(axis=1)
    )

    # ------------------------------------------------------------
    # Primary context
    # ------------------------------------------------------------

    context_scores = pd.DataFrame(
        {
            "holiday": out["holiday_active"],
            "promotion": out["promotion_active"],
            "event": out["event_active"],
            "weather": out["is_rainy"],
        },
        index=out.index,
    )

    out["spike_primary_context"] = (
        context_scores.idxmax(axis=1)
    )

    no_context = (
        context_scores.max(axis=1) <= 0
    )

    out.loc[
        no_context,
        "spike_primary_context",
    ] = "none"

    # ------------------------------------------------------------
    # Multiple context signals
    # ------------------------------------------------------------

    out["spike_multi_context"] = (
        out["spike_context_signal_count"] >= 2
    )

    # ------------------------------------------------------------
    # Temperature availability
    # ------------------------------------------------------------

    out["spike_temperature_available"] = (
        out["temperature"].notna()
    )

    # Use global distribution only to identify unusual
    # temperature observations. This is descriptive intelligence,
    # not a forecasting feature.
    if out["temperature"].notna().any():

        low_temperature = (
            out["temperature"]
            .quantile(0.10)
        )

        high_temperature = (
            out["temperature"]
            .quantile(0.90)
        )

        out["spike_temperature_extreme"] = (
            out["temperature"].notna()
            &
            (
                (out["temperature"] <= low_temperature)
                |
                (out["temperature"] >= high_temperature)
            )
        )

    else:

        out["spike_temperature_extreme"] = False

    # ------------------------------------------------------------
    # Driver classification
    # ------------------------------------------------------------

    out["spike_driver_class"] = np.select(
        [
            ~spike,

            out["spike_multi_context"],

            out["spike_context_signal_count"] == 1,

            (
                out["spike_temperature_extreme"]
                & spike
            ),
        ],
        [
            "none",
            "multiple_contexts",
            "single_context",
            "weather_temperature",
        ],
        default="unexplained",
    )

    return out