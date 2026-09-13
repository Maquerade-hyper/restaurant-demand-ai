"""
PART 30 - AUTONOMOUS COMMERCIAL ML

Final acceptance runner.

Validates:

30A Autonomous control loop
30B Forecast -> commercial decision
30C Governance
30D Autonomous safety gates
30E Commercial decision API contract
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


# ============================================================
# IMPORTS
# ============================================================

from app.intelligence.autonomous import (
    AutonomousCommercialMLService,
    AutonomousController,
    AutonomousRequest,
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "autonomous_commercial_ml_result.json"
)


# ============================================================
# HELPERS
# ============================================================

def fail(message):

    print(
        f"\nERROR: {message}"
    )

    raise SystemExit(1)


def make_request():

    return AutonomousRequest(
        outlet_id="O001",
        product_id="P001",
        forecast_demand=100.0,
        current_inventory=50.0,
        lead_time_days=2.0,
        safety_stock=20.0,
        minimum_order_quantity=10.0,
        order_multiple=5.0,
        scenario_multiplier=1.0,
        demand_confidence=0.95,
        high_risk=False,
    )


def assert_finite(
    value,
    name,
):

    if not math.isfinite(
        float(value)
    ):
        fail(
            f"Non-finite value: {name}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PART 30 - AUTONOMOUS COMMERCIAL ML"
    )

    print(
        "=" * 72
    )

    controller = (
        AutonomousController()
    )

    service = (
        AutonomousCommercialMLService()
    )

    # ========================================================
    # 30A - CONTROL LOOP
    # ========================================================

    print(
        "\n30A - AUTONOMOUS CONTROL LOOP"
    )

    request = make_request()

    decision = controller.execute(
        request
    )

    if not decision:
        fail(
            "No autonomous decision returned."
        )

    print(
        "Outlet:",
        decision.outlet_id,
    )

    print(
        "Product:",
        decision.product_id,
    )

    print(
        "Forecast:",
        decision.raw_forecast,
    )

    print(
        "Adjusted forecast:",
        decision.scenario_adjusted_forecast,
    )

    print(
        "Control loop: PASS"
    )

    # ========================================================
    # 30B - COMMERCIAL DECISION
    # ========================================================

    print(
        "\n30B - FORECAST -> SUPPLY -> INVENTORY"
    )

    print(
        f"Lead-time demand: "
        f"{decision.lead_time_demand:.2f}"
    )

    print(
        f"Current inventory: "
        f"{decision.current_inventory:.2f}"
    )

    print(
        f"Safety stock: "
        f"{decision.safety_stock:.2f}"
    )

    print(
        f"Shortage: "
        f"{decision.shortage:.2f}"
    )

    print(
        f"Recommended order: "
        f"{decision.recommended_order:.2f}"
    )

    print(
        f"Risk: "
        f"{decision.stockout_risk}"
    )

    if (
        decision.recommended_order
        < 0
    ):
        fail(
            "Negative recommended order."
        )

    print(
        "COMMERCIAL DECISION: PASS"
    )

    # ========================================================
    # 30C - GOVERNANCE
    # ========================================================

    print(
        "\n30C - MODEL GOVERNANCE"
    )

    print(
        "Governance action:",
        decision.governance.action,
    )

    print(
        "Governance allowed:",
        decision.governance.allowed,
    )

    print(
        "Governance reason:",
        decision.governance.reason,
    )

    if not decision.health.data_available:
        fail(
            "Unexpected unhealthy data state."
        )

    if not decision.health.model_available:
        fail(
            "Unexpected unhealthy model state."
        )

    print(
        "GOVERNANCE: PASS"
    )

    # ========================================================
    # 30D - SAFETY GATES
    # ========================================================

    print(
        "\n30D - AUTONOMOUS SAFETY GATES"
    )

    # Low confidence
    low_confidence = controller.execute(
        request,
        drift_signal=False,
    )

    # Force low confidence
    low_conf_request = make_request()
    low_conf_request.demand_confidence = (
        0.30
    )

    low_confidence = (
        controller.execute(
            low_conf_request
        )
    )

    if (
        low_confidence.commercial_action
        != "REVIEW_REQUIRED"
    ):
        fail(
            "Low-confidence gate failed."
        )

    # Drift
    drift_decision = (
        controller.execute(
            request,
            drift_signal=True,
        )
    )

    if (
        drift_decision.commercial_action
        != "REVIEW_REQUIRED"
    ):
        fail(
            "Drift gate failed."
        )

    # Model unavailable
    unavailable = (
        controller.execute(
            request,
            model_available=False,
        )
    )

    if (
        unavailable.commercial_action
        != "HOLD"
    ):
        fail(
            "Model health gate failed."
        )

    # High risk
    high_risk_request = (
        make_request()
    )

    high_risk_request.high_risk = True

    high_risk = (
        controller.execute(
            high_risk_request
        )
    )

    if (
        high_risk.commercial_action
        != "REVIEW_REQUIRED"
    ):
        fail(
            "High-risk gate failed."
        )

    print(
        "Low-confidence gate: PASS"
    )

    print(
        "Drift gate: PASS"
    )

    print(
        "Model availability gate: PASS"
    )

    print(
        "High-risk gate: PASS"
    )

    print(
        "SAFETY GATES: PASS"
    )

    # ========================================================
    # 30E - SERVICE
    # ========================================================

    print(
        "\n30E - COMMERCIAL ML SERVICE"
    )

    service_decision = (
        service.decide(
            request
        )
    )

    validation = (
        service.validate(
            service_decision
        )
    )

    if not validation["passed"]:
        fail(
            "Service validation failed: "
            + str(
                validation["errors"]
            )
        )

    assert_finite(
        service_decision.raw_forecast,
        "raw_forecast",
    )

    assert_finite(
        service_decision.scenario_adjusted_forecast,
        "scenario_adjusted_forecast",
    )

    assert_finite(
        service_decision.lead_time_demand,
        "lead_time_demand",
    )

    assert_finite(
        service_decision.recommended_order,
        "recommended_order",
    )

    if not (
        0
        <= service_decision.confidence
        <= 1
    ):
        fail(
            "Invalid confidence."
        )

    print(
        "Service validation: PASS"
    )

    # ========================================================
    # BATCH
    # ========================================================

    print(
        "\n30E - BATCH DECISIONS"
    )

    batch_requests = [
        make_request(),
        make_request(),
        make_request(),
        make_request(),
    ]

    batch_result = (
        service.batch(
            batch_requests
        )
    )

    if len(batch_result) != 4:
        fail(
            "Unexpected batch result size."
        )

    for item in batch_result:

        if (
            "recommended_order"
            not in item
        ):
            fail(
                "Batch decision missing "
                "recommended_order."
            )

        assert_finite(
            item["recommended_order"],
            "batch recommended_order",
        )

    print(
        "Batch decisions: 4"
    )

    print(
        "BATCH SERVICE: PASS"
    )

    # ========================================================
    # AUDITABILITY
    # ========================================================

    print(
        "\n30E - DECISION AUDITABILITY"
    )

    result_dict = (
        service_decision.to_dict()
    )

    required_keys = {
        "outlet_id",
        "product_id",
        "raw_forecast",
        "scenario_adjusted_forecast",
        "lead_time_demand",
        "current_inventory",
        "safety_stock",
        "shortage",
        "recommended_order",
        "stockout_risk",
        "commercial_action",
        "confidence",
        "health",
        "governance",
        "explanation",
    }

    missing = (
        required_keys
        - set(result_dict)
    )

    if missing:
        fail(
            "Decision audit contract missing: "
            f"{sorted(missing)}"
        )

    if not result_dict[
        "explanation"
    ]:
        fail(
            "Decision explanation missing."
        )

    print(
        "Decision explanation: PASS"
    )

    print(
        "Audit contract: PASS"
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output = {
        "part": 30,
        "status": "PASS",
        "architecture": (
            "autonomous_commercial_ml"
        ),
        "decision": result_dict,
        "safety_tests": {
            "low_confidence":
                low_confidence.to_dict(),
            "drift":
                drift_decision.to_dict(),
            "model_unavailable":
                unavailable.to_dict(),
            "high_risk":
                high_risk.to_dict(),
        },
        "batch_count": len(
            batch_result
        ),
        "validation": validation,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PART 30 ACCEPTANCE: PASS"
    )

    print(
        "=" * 72
    )

    print(
        f"Result: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()