from pathlib import Path

import pandas as pd

from app.intelligence.demand.deconstraining import (
    DemandDeconstrainingService,
)

from app.intelligence.demand.lost_demand import (
    LostDemandIntelligenceService,
)


SALES_PATH = Path(
    "data/synthetic/sales.csv"
)

INVENTORY_PATH = Path(
    "data/synthetic/inventory.csv"
)

OUTPUT_PATH = Path(
    "data/interim/lost_demand_intelligence.csv"
)


def main():

    print("=" * 72)
    print(
        "PART 16C - LOST DEMAND INTELLIGENCE"
    )
    print("=" * 72)

    sales = pd.read_csv(
        SALES_PATH
    )

    inventory = pd.read_csv(
        INVENTORY_PATH
    )

    sales["date"] = pd.to_datetime(
        sales["date"]
    )

    inventory["date"] = pd.to_datetime(
        inventory["date"]
    )

    # ---------------------------------------------------------
    # Part 16B
    # ---------------------------------------------------------

    deconstraining = (
        DemandDeconstrainingService(
            lookback_days=56,
            minimum_clean_days=7,
        )
    )

    demand = deconstraining.analyze(
        sales=sales,
        inventory=inventory,
    )

    # ---------------------------------------------------------
    # Part 16C
    # ---------------------------------------------------------

    intelligence = (
        LostDemandIntelligenceService()
    )

    analyzed = intelligence.analyze(
        demand
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    analyzed.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Summaries
    # ---------------------------------------------------------

    outlet_summary = (
        intelligence.summarize_outlets(
            demand
        )
    )

    product_summary = (
        intelligence.summarize_products(
            demand
        )
    )

    outlet_product_summary = (
        intelligence.summarize_outlet_products(
            demand
        )
    )

    top = intelligence.top_opportunities(
        demand,
        limit=10,
    )

    print(
        f"Daily records            : "
        f"{len(analyzed):,}"
    )

    print(
        f"Outlet summaries         : "
        f"{len(outlet_summary):,}"
    )

    print(
        f"Product summaries        : "
        f"{len(product_summary):,}"
    )

    print(
        f"Outlet-product pairs     : "
        f"{len(outlet_product_summary):,}"
    )

    print(
        f"Estimated lost demand    : "
        f"{analyzed['estimated_lost_demand'].sum():,.2f}"
    )

    print(
        f"Average lost-demand rate : "
        f"{analyzed['lost_demand_rate'].mean():.4f}"
    )

    print(
        f"Lost-demand days         : "
        f"{analyzed['lost_demand_flag'].sum():,}"
    )

    print(
        f"Material loss days       : "
        f"{analyzed['material_lost_demand_flag'].sum():,}"
    )

    print(
        f"Severe loss days         : "
        f"{analyzed['severe_lost_demand_flag'].sum():,}"
    )

    print("-" * 72)
    print("TOP LOST-DEMAND OPPORTUNITIES")
    print("-" * 72)

    columns = [
        "outlet_id",
        "product_id",
        "date",
        "estimated_lost_demand",
        "lost_demand_rate",
        "stockout_days_7d",
        "stockout_days_28d",
        "loss_severity",
        "priority_score",
        "priority_band",
    ]

    print(
        top[columns].to_string(
            index=False
        )
    )

    print("-" * 72)

    print(
        f"Output saved             : "
        f"{OUTPUT_PATH}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()