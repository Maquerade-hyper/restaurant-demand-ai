from __future__ import annotations

import numpy as np
import pandas as pd


def classify_spike_severity(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Part 23B - Demand Spike Severity.

    Converts spike magnitude into interpretable severity classes.

    Severity is based on already-computed causal spike statistics.
    """

    required = {
        "spike_score",
        "spike_ratio",
        "spike_z_score",
        "is_demand_spike",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    out = df.copy()

    score = pd.to_numeric(
        out["spike_score"],
        errors="coerce",
    ).fillna(0.0)

    ratio = pd.to_numeric(
        out["spike_ratio"],
        errors="coerce",
    ).fillna(0.0)

    z_score = pd.to_numeric(
        out["spike_z_score"],
        errors="coerce",
    ).fillna(0.0)

    spike = (
        out["is_demand_spike"]
        .astype(bool)
    )

    # ============================================================
    # SEVERITY
    # ============================================================

    conditions = [
        (
            spike
            &
            (
                (score >= 2.0)
                |
                (ratio >= 3.0)
                |
                (z_score >= 5.0)
            )
        ),
        (
            spike
            &
            (
                (score >= 1.25)
                |
                (ratio >= 2.0)
                |
                (z_score >= 3.0)
            )
        ),
        spike,
    ]

    choices = [
        "extreme",
        "major",
        "moderate",
    ]

    out["spike_severity"] = np.select(
        conditions,
        choices,
        default="none",
    )

    # ============================================================
    # SEVERITY SCORE
    # ============================================================

    severity_map = {
        "none": 0,
        "moderate": 1,
        "major": 2,
        "extreme": 3,
    }

    out["spike_severity_score"] = (
        out["spike_severity"]
        .map(severity_map)
        .fillna(0)
        .astype(int)
    )

    # ============================================================
    # OPERATIONAL PRIORITY
    # ============================================================

    out["spike_priority"] = np.select(
        [
            out["spike_severity"]
            == "extreme",

            out["spike_severity"]
            == "major",

            out["spike_severity"]
            == "moderate",
        ],
        [
            "critical",
            "high",
            "watch",
        ],
        default="normal",
    )

    return out