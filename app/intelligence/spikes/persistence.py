from __future__ import annotations

import numpy as np
import pandas as pd


def analyze_spike_persistence(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Part 23D - Spike Persistence and Recovery.

    Identifies:

    - isolated spikes
    - consecutive spike episodes
    - episode length
    - days since previous spike
    - recovery after a spike

    All calculations are chronological within each
    outlet-product series.
    """

    required = {
        "outlet_id",
        "product_id",
        "date",
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

    if df.empty:
        return df.copy()

    out = df.copy()

    out["date"] = pd.to_datetime(
        out["date"]
    )

    out["_p23_original_order"] = np.arange(
        len(out)
    )

    out = (
        out.sort_values(
            [
                "outlet_id",
                "product_id",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    group_keys = [
        "outlet_id",
        "product_id",
    ]

    spike = (
        out["is_demand_spike"]
        .astype(bool)
    )

    out["_p23_spike_int"] = (
        spike.astype(int)
    )

    grouped = out.groupby(
        group_keys,
        sort=False,
    )

    # ============================================================
    # PREVIOUS SPIKE
    # ============================================================

    previous_spike_date = (
        out["date"]
        .where(spike)
        .groupby(
            [
                out["outlet_id"],
                out["product_id"],
            ],
            sort=False,
        )
        .ffill()
        .groupby(
            [
                out["outlet_id"],
                out["product_id"],
            ],
            sort=False,
        )
        .shift(1)
    )

    out["days_since_previous_spike"] = (
        (
            out["date"]
            -
            previous_spike_date
        )
        .dt.days
    )

    # ============================================================
    # CONSECUTIVE EPISODES
    # ============================================================

    previous_spike = (
        grouped["_p23_spike_int"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    new_episode = (
        spike
        &
        (
            previous_spike
            == 0
        )
    )

    out["_p23_new_episode"] = (
        new_episode.astype(int)
    )

    out["spike_episode_id"] = (
        grouped["_p23_new_episode"]
        .cumsum()
    )

    # Rows without a spike don't belong to an active episode.
    out.loc[
        ~spike,
        "spike_episode_id",
    ] = 0

    # ============================================================
    # EPISODE LENGTH
    # ============================================================

    episode_lengths = (
        out.loc[
            spike,
            [
                "outlet_id",
                "product_id",
                "spike_episode_id",
            ],
        ]
        .groupby(
            [
                "outlet_id",
                "product_id",
                "spike_episode_id",
            ]
        )
        .size()
        .rename(
            "spike_episode_length"
        )
        .reset_index()
    )

    out = out.merge(
        episode_lengths,
        on=[
            "outlet_id",
            "product_id",
            "spike_episode_id",
        ],
        how="left",
    )

    out["spike_episode_length"] = (
        out["spike_episode_length"]
        .fillna(0)
        .astype(int)
    )

    # ============================================================
    # SPIKE TYPE
    # ============================================================

    out["spike_pattern"] = np.select(
        [
            ~spike,

            spike
            &
            (
                out["spike_episode_length"]
                >= 3
            ),

            spike
            &
            (
                out["spike_episode_length"]
                == 2
            ),

            spike,
        ],
        [
            "normal",
            "persistent",
            "repeated",
            "isolated",
        ],
        default="normal",
    )

    # ============================================================
    # RECOVERY
    # ============================================================

    # A recovery row is the first non-spike day immediately
    # following a spike.
    next_spike = (
        grouped["_p23_spike_int"]
        .shift(-1)
        .fillna(0)
        .astype(int)
    )

    out["spike_recovery_day"] = (
        (~spike)
        &
        (
            previous_spike
            == 1
        )
    )

    out["spike_followed_by_spike"] = (
        spike
        &
        (
            next_spike
            == 1
        )
    )

    # ============================================================
    # POST-SPIKE RECOVERY PRESSURE
    # ============================================================

    if "spike_baseline" in out.columns:
        if "spike_ratio" in out.columns:
            out["post_spike_demand_ratio"] = (
                pd.to_numeric(
                    out["spike_ratio"],
                    errors="coerce",
                )
            )
        else:
            out["post_spike_demand_ratio"] = np.nan

    else:
        out["post_spike_demand_ratio"] = np.nan

    # ============================================================
    # CLEANUP
    # ============================================================

    numeric_columns = out.select_dtypes(
        include=[np.number]
    ).columns

    out[numeric_columns] = (
        out[numeric_columns]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    out = (
        out
        .sort_values(
            "_p23_original_order"
        )
        .reset_index(drop=True)
    )

    out.drop(
        columns=[
            "_p23_original_order",
            "_p23_spike_int",
            "_p23_new_episode",
        ],
        inplace=True,
        errors="ignore",
    )

    return out