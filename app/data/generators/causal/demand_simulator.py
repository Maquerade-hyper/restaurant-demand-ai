from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from app.data.generators.config import (
    END_DATE,
    SEED,
    START_DATE,
)


class CausalDemandSimulator:
    """
    Generate synthetic demand using a causal chain:

        TRUE DEMAND
            ->
        AVAILABLE INVENTORY
            ->
        OBSERVED SALES
            ->
        LOST DEMAND / STOCKOUT

    The latent true demand is persisted separately from observed sales.
    """

    def __init__(
        self,
        seed: int = SEED,
        target_stockout_rate: float = 0.05,
    ):
        if not 0.01 <= target_stockout_rate <= 0.20:
            raise ValueError(
                "target_stockout_rate must be between 0.01 and 0.20."
            )

        self.seed = seed
        self.target_stockout_rate = target_stockout_rate

        self.random = random.Random(seed)
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # Demand generation
    # ---------------------------------------------------------

    def _outlet_multiplier(self, outlet) -> float:

        multiplier = 0.7 + (
            float(outlet["capacity"]) / 300.0
        )

        outlet_type = outlet["outlet_type"]

        if outlet_type == "bar":
            multiplier *= 1.15

        elif outlet_type in {
            "cloud_kitchen",
            "delivery_kitchen",
        }:
            multiplier *= 1.10

        elif outlet_type == "restaurant_bar":
            multiplier *= 1.08

        return multiplier

    def _product_multiplier(self) -> float:
        return self.random.uniform(0.5, 2.0)

    def _daily_true_demand(
        self,
        outlet,
        product_multiplier: float,
        date: pd.Timestamp,
        start_date: pd.Timestamp,
    ) -> float:

        weekday_multiplier = (
            1.15
            if date.dayofweek >= 5
            else 1.0
        )

        seasonal_multiplier = (
            1.10
            if date.month in {11, 12}
            else 1.0
        )

        trend = (
            1.0
            + (
                (date - start_date).days
                / 365.0
            )
            * 0.15
        )

        # Mild random demand variation.
        noise = max(
            float(self.rng.normal(1.0, 0.12)),
            0.20,
        )

        expected = (
            10.0
            * self._outlet_multiplier(outlet)
            * product_multiplier
            * weekday_multiplier
            * seasonal_multiplier
            * trend
            * noise
        )

        return max(
            float(expected),
            0.0,
        )

    # ---------------------------------------------------------
    # Inventory parameters
    # ---------------------------------------------------------

    def _lead_time(self) -> int:
        return self.random.choice([1, 2, 3])

    def _wastage(
        self,
        available_stock: float,
        shelf_life_days,
    ) -> float:

        if available_stock <= 0:
            return 0.0

        if shelf_life_days <= 3:
            rate = 0.005
        elif shelf_life_days <= 7:
            rate = 0.002
        else:
            rate = 0.0005

        return min(
            available_stock,
            available_stock * rate,
        )

    # ---------------------------------------------------------
    # Main simulation
    # ---------------------------------------------------------

    def generate(
        self,
        outlets: pd.DataFrame,
        products: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        dates = pd.date_range(
            START_DATE,
            END_DATE,
            freq="D",
        )

        start_date = pd.Timestamp(START_DATE)

        sales_rows = []
        inventory_rows = []
        truth_rows = []

        # Keep product demand behavior stable across the year.
        product_multipliers = {
            product["product_id"]:
                self._product_multiplier()
            for _, product in products.iterrows()
        }

        for _, outlet in outlets.iterrows():

            for _, product in products.iterrows():

                outlet_id = outlet["outlet_id"]
                product_id = product["product_id"]

                product_multiplier = (
                    product_multipliers[product_id]
                )

                shelf_life = int(
                    product.get(
                        "shelf_life_days",
                        30,
                    )
                )

                # -------------------------------------------------
                # Initial inventory.
                #
                # Deliberately modest coverage creates realistic
                # opportunities for stock constraints.
                # -------------------------------------------------

                first_demand = self._daily_true_demand(
                    outlet,
                    product_multiplier,
                    dates[0],
                    start_date,
                )

                opening_stock = max(
                    first_demand
                    * self.random.uniform(2.5, 4.5),
                    1.0,
                )

                # Incoming deliveries are scheduled by date.
                scheduled_receipts: dict[
                    pd.Timestamp,
                    float,
                ] = {}

                for date in dates:

                    true_demand = (
                        self._daily_true_demand(
                            outlet,
                            product_multiplier,
                            date,
                            start_date,
                        )
                    )

                    # -------------------------------------------------
                    # Receive previously ordered stock.
                    # -------------------------------------------------

                    received_stock = scheduled_receipts.pop(
                        date,
                        0.0,
                    )

                    available_before_sales = (
                        opening_stock
                        + received_stock
                    )

                    # -------------------------------------------------
                    # Wastage occurs before sales.
                    # -------------------------------------------------

                    wastage = self._wastage(
                        available_before_sales,
                        shelf_life,
                    )

                    available_for_sales = max(
                        available_before_sales
                        - wastage,
                        0.0,
                    )

                    # -------------------------------------------------
                    # Observed sales are constrained by stock.
                    # -------------------------------------------------

                    observed_sales = min(
                        true_demand,
                        available_for_sales,
                    )

                    lost_demand = max(
                        true_demand
                        - observed_sales,
                        0.0,
                    )

                    lost_demand = max(
                        true_demand
                        - observed_sales,
                        0.0,
                    )

                    stockout = (
                        lost_demand > 0.01
                    )

                    closing_stock = max(
                        available_for_sales
                        - observed_sales,
                        0.0,
                    )

                    # -------------------------------------------------
                    # Replenishment policy.
                    #
                    # We use a variable target coverage and random
                    # lead time. This produces genuine constraints
                    # instead of manually flipping stockout=True.
                    # -------------------------------------------------

                    target_coverage_days = (
                        self.random.randint(2, 4)
                    )

                    target_stock = (
                        true_demand
                        * target_coverage_days
                    )

                    inventory_position = (
                        closing_stock
                        + sum(
                            scheduled_receipts.values()
                        )
                    )

                    if inventory_position < target_stock:

                        order_quantity = max(
                            target_stock
                            - inventory_position,
                            0.0,
                        )

                        # Deliberately variable supplier timing.
                        lead_time = self._lead_time()

                        receipt_date = (
                            date
                            + pd.Timedelta(
                                days=lead_time
                            )
                        )

                        if receipt_date <= dates[-1]:

                            scheduled_receipts[
                                receipt_date
                            ] = (
                                scheduled_receipts.get(
                                    receipt_date,
                                    0.0,
                                )
                                + order_quantity
                            )

                    # -------------------------------------------------
                    # Persist records.
                    # -------------------------------------------------

                    sales_rows.append(
                        {
                            "outlet_id": outlet_id,
                            "date": date,
                            "product_id": product_id,
                            "quantity_sold": round(
                                observed_sales,
                                4,
                            ),
                            "unit": product["unit"],
                            "revenue": round(
                                observed_sales
                                * self.random.uniform(
                                    2,
                                    20,
                                ),
                                2,
                            ),
                        }
                    )

                    inventory_rows.append(
                        {
                            "outlet_id": outlet_id,
                            "date": date,
                            "product_id": product_id,
                            "opening_stock": round(
                                opening_stock,
                                4,
                            ),
                            "received_stock": round(
                                received_stock,
                                4,
                            ),
                            "closing_stock": round(
                                closing_stock,
                                4,
                            ),
                            "wastage": round(
                                wastage,
                                4,
                            ),
                            "stockout": stockout,
                        }
                    )

                    truth_rows.append(
                        {
                            "outlet_id": outlet_id,
                            "date": date,
                            "product_id": product_id,
                            "true_demand": round(
                                true_demand,
                                4,
                            ),
                            "observed_sales": round(
                                observed_sales,
                                4,
                            ),
                            "lost_demand": round(
                                lost_demand,
                                4,
                            ),
                            "stockout": stockout,
                        }
                    )

                    # Next day's opening inventory.
                    opening_stock = closing_stock

        sales = pd.DataFrame(sales_rows)
        inventory = pd.DataFrame(inventory_rows)
        truth = pd.DataFrame(truth_rows)

        return sales, inventory, truth