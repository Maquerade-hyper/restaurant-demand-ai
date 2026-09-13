from pathlib import Path

import pandas as pd

from app.intelligence.demand.deconstraining.service import (
    DemandDeconstrainingService,
)
from app.intelligence.outlet.performance import (
    build_outlet_performance,
)


ROOT = Path(__file__).resolve().parents[1]

SALES_PATH = ROOT / "data" / "synthetic" / "sales.csv"
INVENTORY_PATH = ROOT / "data" / "synthetic" / "inventory.csv"
OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "outlet_performance.csv"
)


def main():
    print("=" * 70)
    print("PART 17C - OUTLET PERFORMANCE INTELLIGENCE")
    print("=" * 70)

    sales = pd.read_csv(SALES_PATH)
    inventory = pd.read_csv(INVENTORY_PATH)

    print(f"Sales rows     : {len(sales):,}")
    print(f"Inventory rows : {len(inventory):,}")

    demand = DemandDeconstrainingService().analyze(
        sales=sales,
        inventory=inventory,
    )

    print(f"Demand rows    : {len(demand):,}")

    performance = build_outlet_performance(demand)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    performance.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("PERFORMANCE SUMMARY")
    print(f"Outlets generated : {len(performance)}")

    print()
    print(
        "Average fulfillment     : "
        f"{performance['fulfillment_rate'].mean():.4f}"
    )

    print(
        "Average lost-demand rate: "
        f"{performance['lost_demand_rate'].mean():.4f}"
    )

    print(
        "Average stockout rate   : "
        f"{performance['stockout_product_day_rate'].mean():.4f}"
    )

    print()
    print("TOP PERFORMING OUTLETS")

    columns = [
        "outlet_id",
        "performance_score",
        "fulfillment_rate",
        "lost_demand_rate",
        "performance_class",
    ]

    print(
        performance.sort_values(
            "performance_score",
            ascending=False,
        )[columns]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("TOP OPPORTUNITY OUTLETS")

    columns = [
        "outlet_id",
        "opportunity_index",
        "total_estimated_lost_demand",
        "lost_demand_rate",
        "opportunity_class",
    ]

    print(
        performance.sort_values(
            "opportunity_index",
            ascending=False,
        )[columns]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("TOP OPERATIONAL RISK OUTLETS")

    columns = [
        "outlet_id",
        "risk_index",
        "operational_risk_score",
        "stockout_product_day_rate",
        "lost_demand_rate",
        "operational_risk_class",
    ]

    print(
        performance.sort_values(
            "risk_index",
            ascending=False,
        )[columns]
        .head(10)
        .to_string(index=False)
    )

    print()
    print(f"Output saved : {OUTPUT_PATH}")
    print("PART 17C RUN : PASSED")


if __name__ == "__main__":
    main()