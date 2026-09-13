
from __future__ import annotations

# ---------------------------------------------------------------------------
# Project-root bootstrap
# ---------------------------------------------------------------------------
# When this script is executed as:
#
#     python scripts\run_daily_intelligence.py
#
# Python puts "scripts" on sys.path, not the project root.
# Add the project root explicitly so "from app..." works.
# ---------------------------------------------------------------------------

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from app.intelligence.daily.engine import DailyIntelligenceEngine
from app.intelligence.daily.service import DailyIntelligenceService


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("RESTAURANT DEMAND AI - DAILY COMMERCIAL INTELLIGENCE")
    print("=" * 72)

    print()
    print(f"Project root : {ROOT}")
    print()

    # -----------------------------------------------------------------------
    # Generate report through the service layer.
    #
    # The service owns output serialization/writing.
    # The engine owns report generation.
    # -----------------------------------------------------------------------

    service = DailyIntelligenceService()

    report = service.generate(
        include_normal=False,
        write_output=True,
    )

    # -----------------------------------------------------------------------
    # Report summary
    # -----------------------------------------------------------------------

    summary = report.get("summary", {})
    actions = report.get("actions", [])

    print("DAILY REPORT")
    print("-" * 72)

    print(f"Report date                  : {report.get('report_date')}")
    print(f"Source status                : {report.get('source_status')}")
    print(f"Outlets analyzed             : {summary.get('outlets_analyzed', 0)}")
    print(f"Products analyzed            : {summary.get('products_analyzed', 0)}")
    print(
        "Outlet-product series       : "
        f"{summary.get('outlet_product_series_analyzed', 0)}"
    )
    print(f"Actions required             : {summary.get('actions_required', 0)}")

    # -----------------------------------------------------------------------
    # Action priority summary
    # -----------------------------------------------------------------------

    priority_counts = {}

    for action in actions:
        priority = action.get("priority", "UNKNOWN")
        priority_counts[priority] = priority_counts.get(priority, 0) + 1

    print()
    print("PRIORITY SUMMARY")
    print("-" * 72)

    if priority_counts:
        for priority, count in sorted(priority_counts.items()):
            print(f"{priority:<20}: {count}")
    else:
        print("No actions required.")

    # -----------------------------------------------------------------------
    # Sample actions
    # -----------------------------------------------------------------------

    print()
    print("TOP ACTIONS")
    print("-" * 72)

    for index, action in enumerate(actions[:10], start=1):
        outlet = action.get("outlet", {})
        product = action.get("product", {})
        forecast = action.get("forecast", {})
        inventory = action.get("inventory", {})
        supply = action.get("supply", {})
        recommendation = action.get("recommendation", {})

        print()
        print(f"[{index}]")

        print(
            f"Outlet       : "
            f"{outlet.get('outlet_name', outlet.get('outlet_id', 'UNKNOWN'))}"
        )

        print(f"Outlet Type  : {outlet.get('outlet_type', 'UNKNOWN')}")

        print(
            f"Product      : "
            f"{product.get('product_name', product.get('product_id', 'UNKNOWN'))}"
        )

        print(f"Unit         : {product.get('unit', 'UNKNOWN')}")
        print(f"Date         : {action.get('date')}")

        print(
            f"Forecast     : "
            f"D+1={forecast.get('d1', 0):.2f}, "
            f"D+3={forecast.get('d3', 0):.2f}, "
            f"D+7={forecast.get('d7', 0):.2f}"
        )

        print(
            f"Inventory    : "
            f"current={inventory.get('current_inventory', 0):.2f}, "
            f"required={inventory.get('required_inventory', 0):.2f}, "
            f"shortage={inventory.get('shortage', 0):.2f}"
        )

        print(
            f"Supply       : "
            f"lead_time={supply.get('lead_time_days', 0)} days, "
            f"lead_time_demand={supply.get('lead_time_demand', 0):.2f}, "
            f"order={supply.get('recommended_order', 0):.2f} "
            f"{supply.get('order_unit', product.get('unit', ''))}"
        )

        print(f"Demand Risk  : {action.get('demand_risk', 'UNKNOWN')}")
        print(f"Stockout Risk: {action.get('stockout_risk', 'UNKNOWN')}")
        print(f"Supplier Risk: {supply.get('supplier_risk', 'UNKNOWN')}")
        print(f"Priority     : {action.get('priority', 'UNKNOWN')}")

        print(
            f"Recommendation: "
            f"{recommendation.get('action', 'UNKNOWN')}"
        )

        print(
            f"Reason       : "
            f"{recommendation.get('reason', 'UNKNOWN')}"
        )

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    validation = report.get("validation", {})

    print()
    print("VALIDATION")
    print("-" * 72)
    print(f"Passed : {validation.get('passed', False)}")
    print(f"Errors : {len(validation.get('errors', []))}")

    if validation.get("errors"):
        for error in validation["errors"][:10]:
            print(f"  - {error}")

    # -----------------------------------------------------------------------
    # Output location
    # -----------------------------------------------------------------------

    print()
    print("OUTPUT")
    print("-" * 72)

    output_candidates = [
        ROOT / "data" / "interim" / "daily_intelligence.json",
        ROOT / "data" / "interim" / "daily_commercial_intelligence.json",
    ]

    found_output = None

    for candidate in output_candidates:
        if candidate.exists():
            found_output = candidate
            break

    if found_output:
        print(f"JSON report : {found_output}")
    else:
        print("JSON report : service completed; output path not detected")

    # -----------------------------------------------------------------------
    # Final status
    # -----------------------------------------------------------------------

    print()
    print("=" * 72)

    if validation.get("passed", False):
        print("DAILY COMMERCIAL INTELLIGENCE: PASS")
    else:
        print("DAILY COMMERCIAL INTELLIGENCE: FAILED")

    print("=" * 72)


if __name__ == "__main__":
    main()

