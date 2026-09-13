from __future__ import annotations

from typing import Any, Dict


def _line(char: str = "-", width: int = 63) -> str:
    return char * width


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"

    if isinstance(value, int):
        return f"{value:,}"

    return str(value)


def format_client_report(
    result: Dict[str, Any],
) -> str:

    outlet = result["outlet"]
    product = result["product"]
    forecast = result["forecast"]
    demand = result["demand_intelligence"]
    inventory = result["inventory"]
    supply = result["supply"]
    decision = result["autonomous_decision"]

    drivers = result.get("drivers", [])

    lines = []

    lines.append("")
    lines.append(
        "=" * 63
    )
    lines.append(
        "              RESTAURANT DEMAND AI"
    )
    lines.append(
        "              CLIENT INTELLIGENCE REPORT"
    )
    lines.append(
        "=" * 63
    )

    lines.append("")
    lines.append("OUTLET")
    lines.append(_line())

    lines.append(
        f"Outlet ID          : {outlet.get('outlet_id', '-')}"
    )

    if outlet.get("outlet_type"):
        lines.append(
            f"Outlet Type        : {outlet['outlet_type']}"
        )

    if outlet.get("country"):
        lines.append(
            f"Country            : {outlet['country']}"
        )

    if outlet.get("location_type"):
        lines.append(
            f"Location Type      : {outlet['location_type']}"
        )

    lines.append(
        f"Product ID         : {product.get('product_id', '-')}"
    )

    lines.append("")
    lines.append("FORECAST")
    lines.append(_line())

    lines.append(
        f"D+1 Demand         : {_fmt(forecast['d1'])} units"
    )

    lines.append(
        f"D+3 Demand         : {_fmt(forecast['d3'])} units"
    )

    lines.append(
        f"D+7 Demand         : {_fmt(forecast['d7'])} units"
    )

    lines.append("")
    lines.append("DEMAND INTELLIGENCE")
    lines.append(_line())

    lines.append(
        f"Baseline Demand    : "
        f"{_fmt(demand['baseline_demand'])} units/day"
    )

    lines.append(
        f"Expected Change    : "
        f"{demand['demand_change_pct']:+.2f}%"
    )

    lines.append(
        f"Demand Spike       : "
        f"{str(demand['spike_severity']).upper()}"
    )

    lines.append(
        f"Demand Risk        : "
        f"{str(demand['demand_risk']).upper()}"
    )

    lines.append(
        f"Model Confidence   : "
        f"{demand['confidence'] * 100:.1f}%"
    )

    lines.append("")
    lines.append("INVENTORY")
    lines.append(_line())

    lines.append(
        f"Current Inventory  : "
        f"{_fmt(inventory['current_inventory'])}"
    )

    lines.append(
        f"Required Inventory : "
        f"{_fmt(inventory['required_inventory'])}"
    )

    lines.append(
        f"Shortage           : "
        f"{_fmt(inventory['shortage'])}"
    )

    lines.append(
        f"Safety Stock       : "
        f"{_fmt(inventory['safety_stock'])}"
    )

    lines.append(
        f"Stockout Risk      : "
        f"{str(inventory['stockout_risk']).upper()}"
    )

    lines.append("")
    lines.append("SUPPLY")
    lines.append(_line())

    lines.append(
        f"Lead Time          : "
        f"{supply['lead_time_days']} days"
    )

    lines.append(
        f"Lead-Time Demand   : "
        f"{_fmt(supply['lead_time_demand'])}"
    )

    lines.append(
        f"Recommended Order  : "
        f"{_fmt(supply['recommended_order'])} units"
    )

    lines.append(
        f"Supplier Risk      : "
        f"{str(supply['supplier_risk']).upper()}"
    )

    lines.append("")
    lines.append("KEY DRIVERS")
    lines.append(_line())

    for driver in drivers:
        lines.append(
            f"• {driver}"
        )

    lines.append("")
    lines.append(
        "=" * 63
    )

    lines.append(
        "             AUTONOMOUS RECOMMENDATION"
    )

    lines.append(
        "=" * 63
    )

    lines.append("")
    lines.append(
        f"ACTION             : "
        f"{decision['action']}"
    )

    lines.append(
        f"PRIORITY           : "
        f"{decision['priority']}"
    )

    lines.append("")
    lines.append(
        "WHY?"
    )

    lines.append(_line())

    lines.append(
        decision["reason"]
    )

    lines.append("")
    lines.append(
        "=" * 63
    )

    lines.append(
        "DATA STATUS"
    )

    lines.append(_line())

    lines.append(
        "Source             : SYNTHETIC DEMONSTRATION DATA"
    )

    lines.append(
        "Forecast            : DATA-DRIVEN"
    )

    lines.append(
        "Continuous Learning : BACKEND ARCHITECTURE READY"
    )

    lines.append(
        "Real Client Data    : FUTURE DATA-ADAPTER INPUT"
    )

    lines.append(
        "=" * 63
    )

    lines.append("")

    return "\n".join(lines)