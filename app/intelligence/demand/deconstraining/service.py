from __future__ import annotations

import pandas as pd

from .estimator import (
    DeconstrainedDemandEstimator,
)


class DemandDeconstrainingService:
    """
    Public service interface for Part 16B.

    Returns availability-aware demand estimates.
    """

    def __init__(
        self,
        lookback_days: int = 56,
        minimum_clean_days: int = 7,
    ):

        self.estimator = (
            DeconstrainedDemandEstimator(
                lookback_days=lookback_days,
                minimum_clean_days=minimum_clean_days,
            )
        )

    def analyze(
        self,
        sales: pd.DataFrame,
        inventory: pd.DataFrame,
    ) -> pd.DataFrame:

        result = self.estimator.transform(
            sales=sales,
            inventory=inventory,
        )

        output_columns = [
            "outlet_id",
            "product_id",
            "date",
            "quantity_sold",
            "closing_stock",
            "stockout",
            "stockout_duration",
            "deconstrained_demand",
            "estimated_lost_demand",
            "estimated_fulfillment_rate",
            "demand_constrained_estimated",
        ]

        available_columns = [
            column
            for column in output_columns
            if column in result.columns
        ]

        return result[
            available_columns
        ].copy()