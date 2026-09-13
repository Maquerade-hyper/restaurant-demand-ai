from __future__ import annotations

import pandas as pd

from .censoring import (
    calculate_censoring_signals,
    classify_censoring,
)
from .events import (
    summarize_stockout_events,
)
from .recovery import (
    calculate_post_stockout_recovery,
    summarize_recovery,
)


class DemandCensoringService:
    """
    Public service for demand censoring and stockout intelligence.

    This service operates only on observed/deconstrained demand
    intelligence. It does not consume true-demand ground truth.
    """

    def __init__(
        self,
        pre_window: int = 7,
        recovery_window: int = 7,
    ) -> None:

        self.pre_window = pre_window
        self.recovery_window = recovery_window

    def analyze(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        result = calculate_censoring_signals(
            df,
            pre_window=self.pre_window,
        )

        result = classify_censoring(result)

        result = calculate_post_stockout_recovery(
            result,
            recovery_window=self.recovery_window,
        )

        return result

    def summarize_events(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        analyzed = self.analyze(df)

        events = summarize_stockout_events(
            analyzed
        )

        if events.empty:
            return events

        event_features = (
            analyzed[
                analyzed["stockout"]
            ]
            .groupby(
                [
                    "outlet_id",
                    "product_id",
                    "stockout_event_id",
                ],
                as_index=False,
            )
            .agg(
                average_censoring_strength=(
                    "censoring_strength",
                    "mean",
                ),
                max_censoring_strength=(
                    "censoring_strength",
                    "max",
                ),
                strong_censoring_days=(
                    "censoring_class",
                    lambda x: (x == "strong").sum(),
                ),
                moderate_censoring_days=(
                    "censoring_class",
                    lambda x: (x == "moderate").sum(),
                ),
            )
        )

        return events.merge(
            event_features,
            on=[
                "outlet_id",
                "product_id",
                "stockout_event_id",
            ],
            how="left",
        )

    def summarize_outlets(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        analyzed = self.analyze(df)

        return (
            analyzed.groupby(
                "outlet_id",
                as_index=False,
            )
            .agg(
                records=("date", "count"),
                stockout_days=("stockout", "sum"),
                censored_days=(
                    "censored_demand_flag",
                    "sum",
                ),
                average_censoring_strength=(
                    "censoring_strength",
                    "mean",
                ),
                strong_censoring_days=(
                    "censoring_class",
                    lambda x: (x == "strong").sum(),
                ),
                recovery_events=(
                    "recovery_event",
                    "sum",
                ),
                strong_recovery_events=(
                    "post_stockout_recovery_flag",
                    "sum",
                ),
            )
        )

    def summarize_products(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        analyzed = self.analyze(df)

        return (
            analyzed.groupby(
                "product_id",
                as_index=False,
            )
            .agg(
                records=("date", "count"),
                stockout_days=("stockout", "sum"),
                censored_days=(
                    "censored_demand_flag",
                    "sum",
                ),
                average_censoring_strength=(
                    "censoring_strength",
                    "mean",
                ),
                strong_censoring_days=(
                    "censoring_class",
                    lambda x: (x == "strong").sum(),
                ),
                recovery_events=(
                    "recovery_event",
                    "sum",
                ),
            )
        )

    def top_censored_opportunities(
        self,
        df: pd.DataFrame,
        limit: int = 20,
    ) -> pd.DataFrame:

        analyzed = self.analyze(df)

        opportunities = analyzed[
            analyzed["censored_demand_flag"]
        ].copy()

        if opportunities.empty:
            return opportunities

        opportunities["censoring_priority"] = (
            opportunities["estimated_lost_demand"]
            * (
                1.0
                + opportunities["censoring_strength"]
            )
            * (
                1.0
                + opportunities["stockout_event_day"]
                .fillna(0)
                / 7.0
            )
        )

        return (
            opportunities.sort_values(
                "censoring_priority",
                ascending=False,
            )
            .head(limit)
            .reset_index(drop=True)
        )