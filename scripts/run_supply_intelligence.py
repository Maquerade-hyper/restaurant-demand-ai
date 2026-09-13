from __future__ import annotations

import sys
from pathlib import Path


# ==============================================================
# PROJECT ROOT
# ==============================================================
ROOT = Path(__file__).resolve().parents[1]

# Make the project root importable when this file is executed as:
# python scripts\run_supply_intelligence.py
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd

from app.intelligence.supply import (
    SupplyIntelligenceService,
)


# ==============================================================
# DATA PATHS
# ==============================================================

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

DEMAND_PATH = (
    ROOT
    / "data"
    / "interim"
    / "demand_censoring_intelligence.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "supply_intelligence.csv"
)


# ==============================================================
# DEMAND LOADING
# ==============================================================

def load_demand() -> pd.DataFrame:

    # ----------------------------------------------------------
    # Preferred source:
    # Part 16 deconstrained demand
    # ----------------------------------------------------------

    if DEMAND_PATH.exists():

        df = pd.read_csv(
            DEMAND_PATH,
            parse_dates=["date"],
        )

        if "deconstrained_demand" in df.columns:

            demand = df[
                [
                    "outlet_id",
                    "product_id",
                    "date",
                    "deconstrained_demand",
                ]
            ].rename(
                columns={
                    "deconstrained_demand": "demand"
                }
            )

            demand["demand"] = pd.to_numeric(
                demand["demand"],
                errors="coerce",
            ).fillna(0.0)

            demand["demand"] = (
                demand["demand"]
                .clip(lower=0.0)
            )

            return demand

    # ----------------------------------------------------------
    # Fallback:
    # observed sales
    #
    # This is only a fallback for environments where the
    # Part 16 artifact does not exist.
    # ----------------------------------------------------------

    if not SALES_PATH.exists():
        raise FileNotFoundError(
            "Neither Part 16 demand artifact nor "
            f"sales file exists.\n"
            f"Expected demand: {DEMAND_PATH}\n"
            f"Expected sales : {SALES_PATH}"
        )

    sales = pd.read_csv(
        SALES_PATH,
        parse_dates=["date"],
    )

    required = {
        "outlet_id",
        "product_id",
        "date",
        "quantity_sold",
    }

    missing = required - set(sales.columns)

    if missing:
        raise ValueError(
            "Sales file missing columns: "
            f"{sorted(missing)}"
        )

    demand = sales[
        [
            "outlet_id",
            "product_id",
            "date",
            "quantity_sold",
        ]
    ].rename(
        columns={
            "quantity_sold": "demand"
        }
    )

    demand["demand"] = pd.to_numeric(
        demand["demand"],
        errors="coerce",
    ).fillna(0.0)

    demand["demand"] = (
        demand["demand"]
        .clip(lower=0.0)
    )

    return demand


# ==============================================================
# MAIN
# ==============================================================

def main() -> None:

    print("=" * 70)
    print("PART 18 - SUPPLY INTELLIGENCE")
    print("=" * 70)

    # ----------------------------------------------------------
    # Load inputs
    # ----------------------------------------------------------

    demand = load_demand()

    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(
            f"Inventory file not found: {INVENTORY_PATH}"
        )

    inventory = pd.read_csv(
        INVENTORY_PATH,
        parse_dates=["date"],
    )

    print(
        f"Demand rows     : {len(demand):,}"
    )

    print(
        f"Inventory rows  : {len(inventory):,}"
    )

    print()

    # ----------------------------------------------------------
    # Create supply intelligence service
    # ----------------------------------------------------------

    service = SupplyIntelligenceService(
        safety_stock_days=1.0
    )

    # ----------------------------------------------------------
    # Run complete Part 18 analysis
    # ----------------------------------------------------------

    result = service.analyze(
        demand=demand,
        inventory=inventory,
    )

    # ----------------------------------------------------------
    # Final validation
    # ----------------------------------------------------------

    validation = service.validate(
        result
    )

    print("SUPPLY INTELLIGENCE RESULTS")
    print("-" * 70)

    print(
        f"Supply rows             : "
        f"{len(result):,}"
    )

    print(
        f"Required supply         : "
        f"{result['required_supply'].sum():,.2f}"
    )

    print(
        f"Recommended order       : "
        f"{result['recommended_order_quantity'].sum():,.2f}"
    )

    print(
        f"Orders required         : "
        f"{int(result['order_required'].sum()):,}"
    )

    print(
        f"High-risk supply rows   : "
        f"{int((result['supply_risk'] == 'high').sum()):,}"
    )

    print(
        f"Moderate-risk rows      : "
        f"{int((result['supply_risk'] == 'moderate').sum()):,}"
    )

    print(
        f"Low-risk rows           : "
        f"{int((result['supply_risk'] == 'low').sum()):,}"
    )

    print()

    print("VALIDATION")
    print("-" * 70)

    print(
        f"PASSED                  : "
        f"{validation['passed']}"
    )

    print(
        f"ERRORS                  : "
        f"{len(validation['errors'])}"
    )

    if validation["errors"]:

        for error in validation["errors"]:
            print(
                f"  ERROR: {error}"
            )

        raise SystemExit(
            "PART 18 VALIDATION FAILED"
        )

    # ----------------------------------------------------------
    # Save artifact
    # ----------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()

    print(
        f"Output saved             : "
        f"{OUTPUT_PATH}"
    )

    print()

    print("=" * 70)
    print("PART 18 SUPPLY INTELLIGENCE : PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()