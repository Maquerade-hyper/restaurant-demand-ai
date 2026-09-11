from __future__ import annotations

import random
from datetime import timedelta

import numpy as np
import pandas as pd

from app.data.generators.config import (
    END_DATE,
    NUM_OUTLETS,
    SEED,
    START_DATE,
)


def generate_sales(
    outlets: pd.DataFrame,
    products: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    dates = pd.date_range(
        START_DATE,
        END_DATE,
        freq="D",
    )

    rows = []

    for _, outlet in outlets.iterrows():

        outlet_multiplier = 0.7 + (
            outlet["capacity"] / 300.0
        )

        if outlet["outlet_type"] == "bar":
            outlet_multiplier *= 1.15

        elif outlet["outlet_type"] in {
            "cloud_kitchen",
            "delivery_kitchen",
        }:
            outlet_multiplier *= 1.10

        for _, product in products.iterrows():

            product_multiplier = rng.uniform(
                0.5,
                2.0,
            )

            for date in dates:

                dow = date.dayofweek
                month = date.month

                weekday_multiplier = (
                    1.15
                    if dow >= 5
                    else 1.0
                )

                seasonal_multiplier = (
                    1.10
                    if month in {11, 12}
                    else 1.0
                )

                trend = (
                    1.0
                    + ((date - dates[0]).days / len(dates))
                    * 0.15
                )

                expected_demand = (
                    25
                    * outlet_multiplier
                    * product_multiplier
                    * weekday_multiplier
                    * seasonal_multiplier
                    * trend
                )

                noise = np_rng.normal(
                    1.0,
                    0.12,
                )

                quantity = max(
                    0,
                    int(
                        round(
                            expected_demand
                            * noise
                        )
                    ),
                )

                rows.append(
                    {
                        "outlet_id": outlet["outlet_id"],
                        "date": date,
                        "product_id": product["product_id"],
                        "quantity_sold": quantity,
                        "unit": product["unit"],
                        "revenue": round(
                            quantity
                            * rng.uniform(2, 20),
                            2,
                        ),
                    }
                )

    return pd.DataFrame(rows)