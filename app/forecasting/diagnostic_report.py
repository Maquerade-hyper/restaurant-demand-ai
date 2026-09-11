from __future__ import annotations

import pandas as pd

from app.forecasting.diagnostics import (
    diagnostic_metrics,
    bias_analysis,
    high_demand_analysis,
    spike_recall,
    group_diagnostics,
)


def create_diagnostic_report(
    df: pd.DataFrame,
) -> dict:

    actual = df[
        "quantity_sold"
    ].to_numpy()

    predicted = df[
        "prediction"
    ].to_numpy()

    overall = diagnostic_metrics(
        actual,
        predicted,
    )

    bias = bias_analysis(
        actual,
        predicted,
    )

    high_demand = high_demand_analysis(
        actual,
        predicted,
    )

    spikes = spike_recall(
        actual,
        predicted,
    )

    outlet = group_diagnostics(
        df,
        group_column="outlet_id",
    )

    product = group_diagnostics(
        df,
        group_column="product_id",
    )

    return {
        "overall": overall,
        "bias": bias,
        "high_demand": high_demand,
        "spike_recall": spikes,
        "outlet_diagnostics": outlet,
        "product_diagnostics": product,
    }