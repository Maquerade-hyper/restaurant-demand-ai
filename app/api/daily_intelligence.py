from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.intelligence.daily.service import DailyIntelligenceService


router = APIRouter(
    prefix="/api/v1",
    tags=["Daily Commercial Intelligence"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "interim" / "daily_intelligence.json"


def _load_report() -> dict[str, Any]:
    if not DEFAULT_OUTPUT.exists():
        raise HTTPException(
            status_code=404,
            detail="Daily intelligence report does not exist yet.",
        )

    try:
        with DEFAULT_OUTPUT.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read daily intelligence report: {exc}",
        ) from exc

    if not isinstance(report, dict):
        raise HTTPException(
            status_code=500,
            detail="Daily intelligence report has an invalid structure.",
        )

    return report


def _actions(report: dict[str, Any]) -> list[dict[str, Any]]:
    actions = report.get("actions", [])

    if not isinstance(actions, list):
        raise HTTPException(
            status_code=500,
            detail="Daily intelligence report contains invalid actions.",
        )

    return actions


def _priority_summary(actions: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}

    for action in actions:
        priority = str(action.get("priority", "UNKNOWN")).upper()
        summary[priority] = summary.get(priority, 0) + 1

    return dict(sorted(summary.items()))


def _risk_summary(actions: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}

    for action in actions:
        risk = str(action.get("stockout_risk", "UNKNOWN")).upper()
        summary[risk] = summary.get(risk, 0) + 1

    return dict(sorted(summary.items()))


def _compact_action(action: dict[str, Any]) -> dict[str, Any]:
    """
    Convert the full internal action into a small business-facing object.
    """

    outlet = action.get("outlet", {})
    product = action.get("product", {})
    forecast = action.get("forecast", {})
    inventory = action.get("inventory", {})
    supply = action.get("supply", {})
    recommendation = action.get("recommendation", {})

    return {
        "outlet": {
            "outlet_id": outlet.get("outlet_id"),
            "outlet_name": outlet.get("outlet_name"),
            "outlet_type": outlet.get("outlet_type"),
        },
        "product": {
            "product_id": product.get("product_id"),
            "product_name": product.get("product_name"),
            "unit": product.get("unit"),
        },
        "date": action.get("date"),
        "forecast": {
            "d1": forecast.get("d1"),
            "d3": forecast.get("d3"),
            "d7": forecast.get("d7"),
        },
        "inventory": {
            "current_inventory": inventory.get("current_inventory"),
            "required_inventory": inventory.get("required_inventory"),
            "shortage": inventory.get("shortage"),
            "safety_stock": inventory.get("safety_stock"),
        },
        "supply": {
            "lead_time_days": supply.get("lead_time_days"),
            "lead_time_demand": supply.get("lead_time_demand"),
            "recommended_order": supply.get("recommended_order"),
            "order_unit": supply.get("order_unit"),
            "supplier_risk": supply.get("supplier_risk"),
        },
        "demand_risk": action.get("demand_risk"),
        "demand_spike": action.get("demand_spike"),
        "stockout_risk": action.get("stockout_risk"),
        "priority": action.get("priority"),
        "action": recommendation.get("action"),
        "reason": recommendation.get("reason"),
        "key_drivers": action.get("key_drivers", []),
    }


def _matches(
    action: dict[str, Any],
    priority: str | None = None,
    outlet_id: str | None = None,
    product_id: str | None = None,
) -> bool:

    outlet = action.get("outlet", {})
    product = action.get("product", {})

    if priority is not None:
        if str(action.get("priority", "")).upper() != priority.upper():
            return False

    if outlet_id is not None:
        if str(outlet.get("outlet_id")) != outlet_id:
            return False

    if product_id is not None:
        if str(product.get("product_id")) != product_id:
            return False

    return True


def _filtered_actions(
    report: dict[str, Any],
    priority: str | None = None,
    outlet_id: str | None = None,
    product_id: str | None = None,
) -> list[dict[str, Any]]:

    return [
        action
        for action in _actions(report)
        if _matches(
            action,
            priority=priority,
            outlet_id=outlet_id,
            product_id=product_id,
        )
    ]


def _summary_response(
    report: dict[str, Any],
    actions: list[dict[str, Any]],
    *,
    top_n: int = 10,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:

    validation = report.get("validation", {})

    # ---------------------------------------------------------
    # Resolve outlet/product counts.
    #
    # The internal report may not expose these fields directly,
    # so derive them from the action catalog when necessary.
    # ---------------------------------------------------------

    outlet_ids = set()
    product_ids = set()
    series_keys = set()

    for action in actions:
        outlet = action.get("outlet", {}) or {}
        product = action.get("product", {}) or {}

        outlet_id = outlet.get("outlet_id")
        product_id = product.get("product_id")

        if outlet_id is not None:
            outlet_ids.add(str(outlet_id))

        if product_id is not None:
            product_ids.add(str(product_id))

        if outlet_id is not None and product_id is not None:
            series_keys.add(
                (str(outlet_id), str(product_id))
            )

    # Prefer report-level values when available.
    outlets_analyzed = report.get("outlets_analyzed")

    if outlets_analyzed is None:
        outlets_analyzed = report.get("outlets")

    if outlets_analyzed is None:
        outlets_analyzed = len(outlet_ids)

    products_analyzed = report.get("products_analyzed")

    if products_analyzed is None:
        products_analyzed = report.get("products")

    if products_analyzed is None:
        products_analyzed = len(product_ids)

    series_analyzed = report.get("series_analyzed")

    if series_analyzed is None:
        series_analyzed = report.get("outlet_product_series")

    if series_analyzed is None:
        series_analyzed = len(series_keys)

    return {
        "service": "Restaurant Demand AI",
        "component": "Daily Commercial Intelligence",
        "report_date": report.get("report_date"),
        "source_status": report.get(
            "source_status",
            "SYNTHETIC_DEMONSTRATION_DATA",
        ),
        "outlets_analyzed": outlets_analyzed,
        "products_analyzed": products_analyzed,
        "series_analyzed": series_analyzed,
        "actions_required": len(actions),
        "priority_summary": _priority_summary(actions),
        "stockout_risk_summary": _risk_summary(actions),
        "top_actions": [
            _compact_action(action)
            for action in actions[:top_n]
        ],
        "validation": {
            "passed": validation.get("passed"),
            "errors": validation.get("errors", []),
        },
        "filters": filters or {},
    }


@router.get("/daily-intelligence")
def get_daily_intelligence(
    report_date: str | None = Query(
        default=None,
        description="Expected report date in YYYY-MM-DD format.",
    ),
    priority: str | None = Query(
        default=None,
        description="Filter by priority, e.g. HIGH or MODERATE.",
    ),
    outlet_id: str | None = Query(
        default=None,
        description="Filter by outlet ID.",
    ),
    product_id: str | None = Query(
        default=None,
        description="Filter by product ID.",
    ),
    top_n: int = Query(
        default=10,
        ge=1,
        le=50,
        description="Number of top actions returned.",
    ),
) -> dict[str, Any]:

    report = _load_report()

    actual_date = report.get("report_date")

    if report_date is not None and report_date != actual_date:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No daily intelligence report exists for "
                f"{report_date}."
            ),
        )

    filtered = _filtered_actions(
        report,
        priority=priority,
        outlet_id=outlet_id,
        product_id=product_id,
    )

    response = _summary_response(
        report,
        filtered,
        top_n=top_n,
        filters={
            "report_date": report_date,
            "priority": priority,
            "outlet_id": outlet_id,
            "product_id": product_id,
        },
    )

    return response


@router.get("/daily-intelligence/actions")
def get_daily_actions(
    page: int = Query(
        default=1,
        ge=1,
        description="Page number.",
    ),
    page_size: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Number of actions per page.",
    ),
    priority: str | None = Query(
        default=None,
        description="Filter by priority.",
    ),
    outlet_id: str | None = Query(
        default=None,
        description="Filter by outlet ID.",
    ),
    product_id: str | None = Query(
        default=None,
        description="Filter by product ID.",
    ),
    report_date: str | None = Query(
        default=None,
        description="Expected report date.",
    ),
) -> dict[str, Any]:

    report = _load_report()

    actual_date = report.get("report_date")

    if report_date is not None and report_date != actual_date:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No daily intelligence report exists for "
                f"{report_date}."
            ),
        )

    filtered = _filtered_actions(
        report,
        priority=priority,
        outlet_id=outlet_id,
        product_id=product_id,
    )

    total = len(filtered)

    start = (page - 1) * page_size
    end = start + page_size

    page_actions = filtered[start:end]

    total_pages = (
        (total + page_size - 1) // page_size
        if total
        else 0
    )

    return {
        "service": "Restaurant Demand AI",
        "component": "Daily Commercial Intelligence",
        "report_date": actual_date,
        "page": page,
        "page_size": page_size,
        "total_actions": total,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_previous": page > 1 and total_pages > 0,
        "filters": {
            "priority": priority,
            "outlet_id": outlet_id,
            "product_id": product_id,
        },
        "actions": [
            _compact_action(action)
            for action in page_actions
        ],
    }


@router.get(
    "/daily-intelligence/actions/{outlet_id}/{product_id}"
)
def get_single_daily_action(
    outlet_id: str,
    product_id: str,
    report_date: str | None = Query(
        default=None,
        description="Expected report date.",
    ),
) -> dict[str, Any]:

    report = _load_report()

    actual_date = report.get("report_date")

    if report_date is not None and report_date != actual_date:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No daily intelligence report exists for "
                f"{report_date}."
            ),
        )

    matches = _filtered_actions(
        report,
        outlet_id=outlet_id,
        product_id=product_id,
    )

    if not matches:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No daily intelligence action found for "
                f"{outlet_id}/{product_id}."
            ),
        )

    return {
        "service": "Restaurant Demand AI",
        "component": "Daily Commercial Intelligence",
        "report_date": actual_date,
        "action": _compact_action(matches[0]),
    }


@router.post("/daily-intelligence/generate")
def generate_daily_intelligence(
    report_date: str | None = Query(
        default=None,
        description="Report date. Uses service default when omitted.",
    ),
    include_normal: bool = Query(
        default=False,
        description="Include NORMAL/no-action records.",
    ),
) -> dict[str, Any]:

    service = DailyIntelligenceService()

    result = service.generate(
        report_date=report_date,
        include_normal=include_normal,
        write_output=True,
    )

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=500,
            detail="Daily intelligence generation returned invalid data.",
        )

    actions = _actions(result)

    return _summary_response(
        result,
        actions,
        top_n=10,
        filters={
            "report_date": report_date,
            "include_normal": include_normal,
        },
    )


@router.get("/daily-intelligence/health")
def daily_intelligence_health() -> dict[str, Any]:

    exists = DEFAULT_OUTPUT.exists()

    report_date = None

    if exists:
        try:
            report = _load_report()
            report_date = report.get("report_date")
        except HTTPException:
            report_date = None

    return {
        "service": "Restaurant Demand AI",
        "component": "Daily Commercial Intelligence API",
        "status": "ready" if exists else "waiting",
        "report_exists": exists,
        "report_date": report_date,
        "report_path": str(DEFAULT_OUTPUT),
    }