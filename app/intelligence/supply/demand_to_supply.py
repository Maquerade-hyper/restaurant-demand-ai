from __future__ import annotations

import pandas as pd

from .bom import explode_demand_to_supply


def build_supply_demand(
    demand: pd.DataFrame,
    bom: pd.DataFrame | None = None,
) -> pd.DataFrame:

    exploded = explode_demand_to_supply(
        demand=demand,
        bom=bom,
    )

    result = (
        exploded
        .groupby(
            [
                "outlet_id",
                "date",
                "supply_product_id",
                "supply_unit",
            ],
            as_index=False,
        )["supply_requirement"]
        .sum()
        .rename(
            columns={
                "supply_requirement": "required_supply"
            }
        )
    )

    result["required_supply"] = (
        result["required_supply"].clip(lower=0.0)
    )

    return result


def aggregate_horizon_supply(
    supply_demand: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "outlet_id",
        "date",
        "supply_product_id",
        "required_supply",
    }

    missing = required - set(supply_demand.columns)

    if missing:
        raise ValueError(
            f"Supply demand missing columns: {sorted(missing)}"
        )

    result = (
        supply_demand
        .groupby(
            [
                "outlet_id",
                "supply_product_id",
            ],
            as_index=False,
        )
        .agg(
            horizon_supply=(
                "required_supply",
                "sum",
            ),
            average_daily_supply=(
                "required_supply",
                "mean",
            ),
            peak_daily_supply=(
                "required_supply",
                "max",
            ),
        )
    )

    return result