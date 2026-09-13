from __future__ import annotations

import random

import pandas as pd

from app.data.generators.config import SEED


def generate_inventory(
    sales: pd.DataFrame,
    products: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic inventory ledger consistent with
    the existing observed-sales dataset.

    Important:
    - quantity_sold is treated as observed demand for this stage.
    - We do not infer true latent demand.
    - Inventory accounting is guaranteed to satisfy:

        opening + received
        - observed_sales
        - wastage
        = closing
    """

    rng = random.Random(seed)

    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"])

    product_lookup = (
        products
        .set_index("product_id")
        .to_dict("index")
    )

    rows = []

    for (outlet_id, product_id), group in sales.groupby(
        ["outlet_id", "product_id"],
        sort=True,
    ):

        group = group.sort_values("date")

        product = product_lookup[product_id]

        shelf_life = product.get(
            "shelf_life_days",
            30,
        )

        # Initial inventory target.
        #
        # Short shelf-life products are deliberately
        # held at lower coverage.
        if shelf_life <= 3:
            coverage_days = rng.uniform(2.0, 4.0)
        elif shelf_life <= 7:
            coverage_days = rng.uniform(3.0, 6.0)
        else:
            coverage_days = rng.uniform(5.0, 10.0)

        mean_demand = max(
            group["quantity_sold"].mean(),
            1.0,
        )

        opening_stock = (
            mean_demand
            * coverage_days
        )

        for _, row in group.iterrows():

            observed_sales = float(
                row["quantity_sold"]
            )

            # Small operational wastage.
            if shelf_life <= 3:
                wastage_rate = rng.uniform(
                    0.02,
                    0.06,
                )
            elif shelf_life <= 7:
                wastage_rate = rng.uniform(
                    0.01,
                    0.04,
                )
            else:
                wastage_rate = rng.uniform(
                    0.005,
                    0.02,
                )

            wastage = min(
                max(
                    opening_stock
                    * wastage_rate,
                    0.0,
                ),
                max(
                    opening_stock
                    - observed_sales,
                    0.0,
                ),
            )

            # Maintain a target inventory position.
            #
            # Use recent demand when available.
            recent_demand = (
                group.loc[
                    group["date"]
                    <= row["date"],
                    "quantity_sold",
                ]
                .tail(7)
                .mean()
            )

            if pd.isna(recent_demand):
                recent_demand = mean_demand

            target_stock = (
                max(
                    float(recent_demand),
                    1.0,
                )
                * coverage_days
            )

            # Required receipt to keep the business
            # operational and maintain target stock.
            required_receipt = (
                observed_sales
                + wastage
                + target_stock
                - opening_stock
            )

            received_stock = max(
                required_receipt,
                0.0,
            )

            closing_stock = (
                opening_stock
                + received_stock
                - observed_sales
                - wastage
            )

            closing_stock = max(
                closing_stock,
                0.0,
            )

            # Because this synthetic ledger is built around
            # observed sales rather than latent true demand,
            # historical stockout is not inferred here.
            stockout = False

            rows.append(
                {
                    "outlet_id": outlet_id,
                    "date": row["date"],
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

            opening_stock = closing_stock

    return pd.DataFrame(rows)