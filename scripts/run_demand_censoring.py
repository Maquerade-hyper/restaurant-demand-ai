from pathlib import Path

import pandas as pd

from app.intelligence.demand.deconstraining.service import (
    DemandDeconstrainingService,
)
from app.intelligence.demand.censoring.service import (
    DemandCensoringService,
)


ROOT = Path(__file__).resolve().parents[1]

SALES_PATH = ROOT / "data" / "synthetic" / "sales.csv"
INVENTORY_PATH = ROOT / "data" / "synthetic" / "inventory.csv"

OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "demand_censoring_intelligence.csv"
)


def main() -> None:
    print("=" * 72)
    print("PART 16D - DEMAND CENSORING & STOCKOUT INTELLIGENCE")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. LOAD SOURCE DATA
    # ------------------------------------------------------------------
    sales = pd.read_csv(SALES_PATH)
    inventory = pd.read_csv(INVENTORY_PATH)

    sales["date"] = pd.to_datetime(sales["date"])
    inventory["date"] = pd.to_datetime(inventory["date"])

    print(f"Sales rows              : {len(sales):,}")
    print(f"Inventory rows          : {len(inventory):,}")

    # ------------------------------------------------------------------
    # 2. RUN PART 16B DECONSTRAINING
    #
    # IMPORTANT:
    # Use the actual public 16B service API:
    #     DemandDeconstrainingService.analyze(...)
    #
    # Do NOT use demand truth here.
    # ------------------------------------------------------------------
    deconstraining_service = DemandDeconstrainingService()

    deconstrained = deconstraining_service.analyze(
        sales=sales,
        inventory=inventory,
    )

    print(
        f"Deconstrained rows      : {len(deconstrained):,}"
    )

    # ------------------------------------------------------------------
    # 3. RUN PART 16D
    # ------------------------------------------------------------------
    censoring_service = DemandCensoringService()

    result = censoring_service.analyze(
        deconstrained,
    )

    # ------------------------------------------------------------------
    # 4. BASIC VALIDATION
    # ------------------------------------------------------------------
    if result.empty:
        raise RuntimeError(
            "Part 16D returned an empty result."
        )

    required_columns = [
        "outlet_id",
        "product_id",
        "date",
        "stockout",
        "deconstrained_demand",
        "estimated_lost_demand",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Part 16D output is missing required columns: "
            + ", ".join(missing_columns)
        )

    # ------------------------------------------------------------------
    # 5. METRICS
    # ------------------------------------------------------------------
    stockout_mask = result["stockout"].fillna(False).astype(bool)

    censored_mask = (
        result["censored_demand_flag"]
        if "censored_demand_flag" in result.columns
        else pd.Series(False, index=result.index)
    )

    censored_mask = (
        censored_mask
        .fillna(False)
        .astype(bool)
    )

    strong_mask = (
        result["censoring_class"]
        .eq("strong")
        if "censoring_class" in result.columns
        else pd.Series(False, index=result.index)
    )

    moderate_mask = (
        result["censoring_class"]
        .eq("moderate")
        if "censoring_class" in result.columns
        else pd.Series(False, index=result.index)
    )

    recovery_mask = (
        result["recovery_event"]
        if "recovery_event" in result.columns
        else pd.Series(False, index=result.index)
    )

    recovery_mask = (
        recovery_mask
        .fillna(False)
        .astype(bool)
    )

    estimated_lost_demand = (
        pd.to_numeric(
            result["estimated_lost_demand"],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )

    # ------------------------------------------------------------------
    # 6. EVENT SUMMARY
    # ------------------------------------------------------------------
    event_summary = censoring_service.summarize_events(result)

    outlet_summary = censoring_service.summarize_outlets(result)

    product_summary = censoring_service.summarize_products(result)

    top_opportunities = censoring_service.top_censored_opportunities(
        result,
        limit=10,
    )

    # ------------------------------------------------------------------
    # 7. PRINT ACCEPTANCE RESULTS
    # ------------------------------------------------------------------
    print()
    print("PART 16D VALIDATION")
    print("-" * 72)

    print(
        f"Daily records            : {len(result):,}"
    )

    print(
        f"Stockout records         : {int(stockout_mask.sum()):,}"
    )

    print(
        f"Censored records        : {int(censored_mask.sum()):,}"
    )

    print(
        f"Stockout events          : {len(event_summary):,}"
    )

    print(
        f"Outlet summaries         : {len(outlet_summary):,}"
    )

    print(
        f"Product summaries        : {len(product_summary):,}"
    )

    print(
        f"Estimated lost demand    : "
        f"{estimated_lost_demand.sum():,.2f}"
    )

    print(
        f"Strong censoring days    : {int(strong_mask.sum()):,}"
    )

    print(
        f"Moderate censoring days  : {int(moderate_mask.sum()):,}"
    )

    print(
        f"Recovery events          : {int(recovery_mask.sum()):,}"
    )

    # ------------------------------------------------------------------
    # 8. TOP CENSORED OPPORTUNITIES
    # ------------------------------------------------------------------
    print()
    print("TOP CENSORED OPPORTUNITIES")
    print("-" * 72)

    display_columns = [
        "outlet_id",
        "product_id",
        "date",
        "estimated_lost_demand",
        "censoring_ratio",
        "pre_stockout_demand_pressure",
        "censoring_strength",
        "censoring_class",
        "stockout_event_day",
        "priority_score",
    ]

    available_display_columns = [
        column
        for column in display_columns
        if column in top_opportunities.columns
    ]

    if not top_opportunities.empty:
        print(
            top_opportunities[
                available_display_columns
            ].to_string(index=False)
        )
    else:
        print("No censored opportunities found.")

    # ------------------------------------------------------------------
    # 9. SAVE RESULT
    # ------------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("-" * 72)
    print(f"Output saved             : {OUTPUT_PATH}")
    print("PART 16D RUN             : SUCCESS")
    print("=" * 72)


if __name__ == "__main__":
    main()