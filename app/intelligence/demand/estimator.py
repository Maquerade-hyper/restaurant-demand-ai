from __future__ import annotations

import numpy as np
import pandas as pd


class TrueDemandEstimator:
    """
    Estimate latent true demand from observed sales and inventory constraints.

    Non-stockout observations:
        true_demand = observed_sales

    Stockout observations:
        true_demand is estimated from prior demand behavior.

    This estimator does not claim that latent demand is directly observed.
    """

    def __init__(
        self,
        lookback_days: int = 28,
        min_history: int = 7,
    ):
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive.")

        if min_history <= 0:
            raise ValueError("min_history must be positive.")

        if min_history > lookback_days:
            raise ValueError(
                "min_history cannot be greater than lookback_days."
            )

        self.lookback_days = lookback_days
        self.min_history = min_history

    def estimate(
        self,
        sales: pd.DataFrame,
        inventory: pd.DataFrame,
    ) -> pd.DataFrame:

        required_sales = {
            "outlet_id",
            "product_id",
            "date",
            "quantity_sold",
        }

        required_inventory = {
            "outlet_id",
            "product_id",
            "date",
            "closing_stock",
            "stockout",
        }

        missing_sales = required_sales - set(sales.columns)

        if missing_sales:
            raise ValueError(
                f"Sales is missing required columns: "
                f"{sorted(missing_sales)}"
            )

        missing_inventory = (
            required_inventory - set(inventory.columns)
        )

        if missing_inventory:
            raise ValueError(
                f"Inventory is missing required columns: "
                f"{sorted(missing_inventory)}"
            )

        sales_df = sales.copy()
        inventory_df = inventory.copy()

        sales_df["date"] = pd.to_datetime(
            sales_df["date"]
        )

        inventory_df["date"] = pd.to_datetime(
            inventory_df["date"]
        )

        result = sales_df.merge(
            inventory_df[
                [
                    "outlet_id",
                    "product_id",
                    "date",
                    "closing_stock",
                    "stockout",
                ]
            ],
            on=[
                "outlet_id",
                "product_id",
                "date",
            ],
            how="left",
            validate="one_to_one",
        )

        result = result.sort_values(
            [
                "outlet_id",
                "product_id",
                "date",
            ]
        ).reset_index(drop=True)

        result["stockout"] = (
            result["stockout"]
            .fillna(False)
            .astype(bool)
        )

        # ---------------------------------------------------------
        # Historical demand reference
        #
        # Shift first so today's observed sales cannot influence
        # today's estimate.
        # ---------------------------------------------------------

        result["historical_demand_estimate"] = (
            result.groupby(
                ["outlet_id", "product_id"],
                sort=False,
            )["quantity_sold"]
            .transform(
                lambda series: (
                    series
                    .shift(1)
                    .rolling(
                        window=self.lookback_days,
                        min_periods=self.min_history,
                    )
                    .mean()
                )
            )
        )

        # If insufficient history exists, use the current observed
        # sales value only as a fallback. This does not create lost
        # demand by itself.
        result["historical_demand_estimate"] = (
            result["historical_demand_estimate"]
            .fillna(
                result["quantity_sold"].astype(float)
            )
        )

        # ---------------------------------------------------------
        # True demand
        # ---------------------------------------------------------

        result["true_demand"] = (
            result["quantity_sold"]
            .astype(float)
        )

        stockout_mask = result["stockout"]

        result.loc[
            stockout_mask,
            "true_demand",
        ] = np.maximum(
            result.loc[
                stockout_mask,
                "quantity_sold",
            ].astype(float),
            result.loc[
                stockout_mask,
                "historical_demand_estimate",
            ].astype(float),
        )

        # ---------------------------------------------------------
        # Lost demand
        # ---------------------------------------------------------

        result["lost_demand"] = (
            result["true_demand"]
            - result["quantity_sold"].astype(float)
        ).clip(lower=0.0)

        # ---------------------------------------------------------
        # Fulfillment
        # ---------------------------------------------------------

        result["fulfillment_rate"] = np.where(
            result["true_demand"] > 0,
            (
                result["quantity_sold"].astype(float)
                / result["true_demand"]
            ),
            1.0,
        )

        result["fulfillment_rate"] = (
            result["fulfillment_rate"]
            .clip(0.0, 1.0)
        )

        result["demand_constrained"] = (
            result["lost_demand"] > 0
        )

        return result