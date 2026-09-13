from pathlib import Path

import pandas as pd

from app.intelligence.demand.deconstraining.service import (
    DemandDeconstrainingService,
)
from app.intelligence.outlet.behavior import (
    build_outlet_demand_behavior,
)


ROOT = Path(__file__).resolve().parents[1]

SALES_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "sales.csv"
)

INVENTORY_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "inventory.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "outlet_demand_behavior.csv"
)


def main():
    print("=" * 70)
    print("PART 17B - OUTLET DEMAND BEHAVIOR")
    print("=" * 70)

    sales = pd.read_csv(
        SALES_PATH
    )

    inventory = pd.read_csv(
        INVENTORY_PATH
    )

    print(
        f"Sales rows               : "
        f"{len(sales):,}"
    )

    print(
        f"Inventory rows           : "
        f"{len(inventory):,}"
    )

    print(
        "\nRunning Part 16B demand deconstraining..."
    )

    demand_service = (
        DemandDeconstrainingService(
            lookback_days=56,
            minimum_clean_days=7,
        )
    )

    demand = demand_service.analyze(
        sales=sales,
        inventory=inventory,
    )

    print(
        f"Deconstrained rows       : "
        f"{len(demand):,}"
    )

    print(
        "\nBuilding outlet demand behavior..."
    )

    behavior = (
        build_outlet_demand_behavior(
            demand
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    behavior.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nBEHAVIOR SUMMARY")

    print(
        f"Outlet profiles          : "
        f"{len(behavior):,}"
    )

    print(
        "\nDEMAND SHAPE"
    )

    print(
        behavior[
            "demand_shape_class"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nDEMAND CONSISTENCY"
    )

    print(
        behavior[
            "demand_consistency_class"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nDEMAND TREND"
    )

    print(
        behavior[
            "trend_class"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nTOP PEAK-DRIVEN OUTLETS"
    )

    top = (
        behavior
        .sort_values(
            "behavior_peak_intensity",
            ascending=False,
        )
        .head(10)
    )

    columns = [
        "outlet_id",
        "behavior_average_daily_demand",
        "behavior_peak_intensity",
        "behavior_coefficient_variation",
        "behavior_weekend_weekday_ratio",
        "behavior_top_3_product_share",
        "demand_shape_class",
        "demand_consistency_class",
        "trend_class",
    ]

    print(
        top[columns].to_string(
            index=False
        )
    )

    print(
        f"\nOutput saved             : "
        f"{OUTPUT_PATH}"
    )

    print(
        "\nPART 17B RUN             : SUCCESS"
    )


if __name__ == "__main__":
    main()