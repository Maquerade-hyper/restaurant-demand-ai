from __future__ import annotations

import pandas as pd

from app.intelligence.demand.estimator import (
    TrueDemandEstimator,
)


class DemandIntelligenceService:

    def __init__(
        self,
        lookback_days: int = 28,
        min_history: int = 7,
    ):
        self.estimator = TrueDemandEstimator(
            lookback_days=lookback_days,
            min_history=min_history,
        )

    def analyze(
        self,
        sales: pd.DataFrame,
        inventory: pd.DataFrame,
    ) -> pd.DataFrame:

        return self.estimator.estimate(
            sales=sales,
            inventory=inventory,
        )

    @staticmethod
    def summarize(
        demand_data: pd.DataFrame,
    ) -> dict:

        true_demand = demand_data[
            "true_demand"
        ].sum()

        observed_sales = demand_data[
            "quantity_sold"
        ].sum()

        lost_demand = demand_data[
            "lost_demand"
        ].sum()

        stockout_records = int(
            demand_data["stockout"].sum()
        )

        total_records = len(demand_data)

        return {
            "records": total_records,
            "true_demand": float(true_demand),
            "observed_sales": float(observed_sales),
            "lost_demand": float(lost_demand),
            "stockout_records": stockout_records,
            "stockout_rate": (
                stockout_records / total_records
                if total_records
                else 0.0
            ),
            "overall_fulfillment_rate": (
                observed_sales / true_demand
                if true_demand > 0
                else 1.0
            ),
        }