from pathlib import Path

import pandas as pd

from app.intelligence.demand.deconstraining.service import (
    DemandDeconstrainingService,
)
from app.intelligence.outlet.service import OutletProfilingService


ROOT = Path(__file__).resolve().parents[1]

SALES_PATH = ROOT / "data" / "synthetic" / "sales.csv"
INVENTORY_PATH = ROOT / "data" / "synthetic" / "inventory.csv"
OUTLETS_PATH = ROOT / "data" / "synthetic" / "outlets.csv"

OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "outlet_profiles.csv"
)


def main():
    print("=" * 70)
    print("PART 17A - OUTLET PROFILING")
    print("=" * 70)

    sales = pd.read_csv(SALES_PATH)
    inventory = pd.read_csv(INVENTORY_PATH)
    outlets = pd.read_csv(OUTLETS_PATH)

    print(f"Sales rows               : {len(sales):,}")
    print(f"Inventory rows           : {len(inventory):,}")
    print(f"Outlets                  : {len(outlets):,}")

    print("\nRunning Part 16B demand deconstraining...")

    demand_service = DemandDeconstrainingService(
        lookback_days=56,
        minimum_clean_days=7,
    )

    demand = demand_service.analyze(
        sales=sales,
        inventory=inventory,
    )

    print(
        f"Deconstrained rows       : {len(demand):,}"
    )

    print("\nBuilding outlet profiles...")

    outlet_service = OutletProfilingService()

    profiles = outlet_service.analyze(
        outlets=outlets,
        demand=demand,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profiles.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nPROFILE SUMMARY")
    print(
        f"Profiles generated       : {len(profiles):,}"
    )
    print(
        f"Unique outlet types      : "
        f"{profiles['outlet_type'].nunique():,}"
    )
    print(
        f"Countries represented    : "
        f"{profiles['country'].nunique():,}"
    )

    print("\nBEHAVIOR CLASSES")

    print(
        profiles[
            "outlet_behavior_class"
        ].value_counts()
        .to_string()
    )

    print("\nOPERATIONAL RISK")

    print(
        profiles[
            "operational_risk_class"
        ].value_counts()
        .to_string()
    )

    print("\nTOP RISK OUTLETS")

    top = outlet_service.top_risk_outlets(
        profiles,
        limit=10,
    )

    columns = [
        "outlet_id",
        "outlet_type",
        "city",
        "average_daily_demand",
        "lost_demand_rate",
        "stockout_rate",
        "demand_trend_class",
        "demand_stability_class",
        "operational_risk_class",
    ]

    print(
        top[
            [
                column
                for column in columns
                if column in top.columns
            ]
        ].to_string(index=False)
    )

    print(
        f"\nOutput saved             : {OUTPUT_PATH}"
    )

    print("\nPART 17A RUN             : SUCCESS")


if __name__ == "__main__":
    main()