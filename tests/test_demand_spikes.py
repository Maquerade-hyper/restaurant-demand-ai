from __future__ import annotations

import numpy as np
import pandas as pd

from app.intelligence.spikes.service import (
    DemandSpikeIntelligenceService,
)


def build_data() -> pd.DataFrame:

    dates = pd.date_range(
        "2025-01-01",
        "2025-03-31",
        freq="D",
    )

    rows = []

    for outlet_id in [
        "O001",
        "O002",
    ]:

        for product_id in [
            "P001",
            "P002",
        ]:

            for date in dates:

                demand = 20.0

                # Controlled artificial spike.
                if (
                    outlet_id == "O001"
                    and product_id == "P001"
                    and date
                    in pd.date_range(
                        "2025-03-15",
                        "2025-03-17",
                    )
                ):
                    demand = 100.0

                rows.append(
                    {
                        "outlet_id": outlet_id,
                        "product_id": product_id,
                        "date": date,
                        "quantity_sold": demand,
                        "temperature": 25.0,
                        "is_rainy": 0.0,
                        "holiday_active": 0.0,
                        "promotion_active": 0.0,
                        "event_active": 0.0,
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# BASIC PIPELINE
# ============================================================

def test_spike_pipeline_generates_output():

    df = build_data()

    service = DemandSpikeIntelligenceService()

    result = service.transform(
        df
    )

    assert not result.empty

    assert len(result) == len(df)


# ============================================================
# REQUIRED FEATURES
# ============================================================

def test_required_spike_features_exist():

    df = build_data()

    service = DemandSpikeIntelligenceService()

    result = service.transform(
        df
    )

    expected = {
        "spike_baseline",
        "spike_ratio",
        "spike_percentage_lift",
        "spike_z_score",
        "spike_score",
        "is_demand_spike",
        "spike_severity",
        "spike_severity_score",
        "spike_priority",
        "spike_pattern",
        "spike_episode_length",
        "days_since_previous_spike",
        "spike_recovery_day",
    }

    missing = (
        expected
        -
        set(result.columns)
    )

    assert not missing


# ============================================================
# SPIKE DETECTION
# ============================================================

def test_known_spike_is_detected():

    df = build_data()

    service = DemandSpikeIntelligenceService()

    result = service.transform(
        df
    )

    target = result[
        (
            result["outlet_id"]
            == "O001"
        )
        &
        (
            result["product_id"]
            == "P001"
        )
        &
        (
            result["date"]
            == pd.Timestamp(
                "2025-03-15"
            )
        )
    ]

    assert len(target) == 1

    assert bool(
        target.iloc[0]["is_demand_spike"]
    ) is True


# ============================================================
# HISTORY CAUSALITY
# ============================================================

def test_current_target_does_not_change_past_baseline():

    df1 = build_data()

    df2 = df1.copy()

    changed_date = pd.Timestamp(
        "2025-03-15"
    )

    df2.loc[
        df2["date"] == changed_date,
        "quantity_sold",
    ] += 500.0

    service = DemandSpikeIntelligenceService()

    result1 = service.transform(
        df1
    )

    result2 = service.transform(
        df2
    )

    # Only dates BEFORE the changed target are compared.
    mask = (
        result1["date"]
        <
        changed_date
    )

    for column in [
        "spike_baseline",
        "spike_baseline_std",
        "spike_baseline_median",
        "spike_history_count",
    ]:

        a = result1.loc[
            mask,
            column,
        ].to_numpy()

        b = result2.loc[
            mask,
            column,
        ].to_numpy()

        assert np.allclose(
            a,
            b,
            equal_nan=True,
        )


# ============================================================
# SEVERITY CONTRACT
# ============================================================

def test_non_spike_rows_have_none_severity():

    df = build_data()

    service = DemandSpikeIntelligenceService()

    result = service.transform(
        df
    )

    mask = ~result[
        "is_demand_spike"
    ]

    assert (
        result.loc[
            mask,
            "spike_severity",
        ]
        == "none"
    ).all()


# ============================================================
# DRIVER CONTRACT
# ============================================================

def test_promotion_context_is_reported():

    df = build_data()

    spike_date = pd.Timestamp(
        "2025-03-15"
    )

    df.loc[
        df["date"] == spike_date,
        "promotion_active",
    ] = 1.0

    service = DemandSpikeIntelligenceService()

    result = service.transform(
        df
    )

    target = result[
        (
            result["date"]
            == spike_date
        )
        &
        (
            result["outlet_id"]
            == "O001"
        )
        &
        (
            result["product_id"]
            == "P001"
        )
    ]

    assert len(target) == 1

    row = target.iloc[0]

    assert bool(
        row["spike_promotion_signal"]
    ) is True


# ============================================================
# VALIDATION
# ============================================================

def test_validation_passes():

    df = build_data()

    service = DemandSpikeIntelligenceService()

    result = service.transform(
        df
    )

    validation = service.validate(
        result
    )

    assert validation["passed"] is True

    assert validation["errors"] == []

    assert validation["rows"] == len(df)


# ============================================================
# SUMMARY
# ============================================================

def test_summary_is_consistent():

    df = build_data()

    service = DemandSpikeIntelligenceService()

    result = service.transform(
        df
    )

    summary = service.summary(
        result
    )

    assert (
        summary["rows"]
        ==
        len(df)
    )

    assert (
        summary["spikes"]
        ==
        int(
            result[
                "is_demand_spike"
            ].sum()
        )
    )

    assert (
        0.0
        <=
        summary["spike_rate"]
        <=
        1.0
    )