
from __future__ import annotations

import json

import pytest

from app.intelligence.daily.engine import DailyIntelligenceEngine
from app.intelligence.daily.service import DailyIntelligenceService


@pytest.fixture(scope="module")
def engine():
    return DailyIntelligenceEngine()


@pytest.fixture(scope="module")
def service():
    return DailyIntelligenceService()


@pytest.fixture(scope="module")
def report(engine):
    return engine.generate(include_normal=False)


@pytest.fixture(scope="module")
def service_report(service):
    return service.generate(
        include_normal=False,
        write_output=False,
    )


def test_engine_initializes(engine):
    assert engine is not None


def test_service_initializes(service):
    assert service is not None


def test_daily_report_exists(report):
    assert isinstance(report, dict)

    assert "service" in report
    assert "version" in report
    assert "report_date" in report
    assert "source_status" in report
    assert "summary" in report
    assert "actions" in report

    assert report["service"] == "Restaurant Demand AI"
    assert isinstance(report["actions"], list)


def test_daily_report_structure(report):
    summary = report["summary"]

    assert "outlets_analyzed" in summary
    assert "products_analyzed" in summary
    assert "outlet_product_series_analyzed" in summary
    assert "actions_required" in summary

    assert summary["outlets_analyzed"] > 0
    assert summary["products_analyzed"] > 0
    assert summary["outlet_product_series_analyzed"] > 0
    assert summary["actions_required"] >= 0

    assert len(report["actions"]) == summary["actions_required"]


def test_daily_actions_have_business_identity(report):
    actions = report["actions"]

    assert len(actions) > 0

    for action in actions[:20]:
        assert "outlet" in action
        assert "product" in action
        assert "date" in action

        outlet = action["outlet"]
        product = action["product"]

        assert "outlet_id" in outlet
        assert "outlet_name" in outlet
        assert "outlet_type" in outlet

        assert "product_id" in product
        assert "product_name" in product
        assert "unit" in product

        assert outlet["outlet_id"]
        assert outlet["outlet_name"]
        assert outlet["outlet_type"]

        assert product["product_id"]
        assert product["product_name"]
        assert product["unit"]

        assert action["date"]


def test_forecast_contract(report):
    actions = report["actions"]

    assert len(actions) > 0

    for action in actions[:20]:
        assert "forecast" in action

        forecast = action["forecast"]

        assert "d1" in forecast
        assert "d3" in forecast
        assert "d7" in forecast

        assert forecast["d1"] >= 0
        assert forecast["d3"] >= 0
        assert forecast["d7"] >= 0


def test_inventory_supply_contract(report):
    actions = report["actions"]

    assert len(actions) > 0

    for action in actions[:20]:
        assert "inventory" in action
        assert "supply" in action

        inventory = action["inventory"]
        supply = action["supply"]

        assert "current_inventory" in inventory
        assert "required_inventory" in inventory
        assert "shortage" in inventory
        assert "safety_stock" in inventory

        # Actual supply contract.
        assert "lead_time_days" in supply
        assert "lead_time_demand" in supply
        assert "recommended_order" in supply
        assert "order_unit" in supply

        assert inventory["current_inventory"] >= 0
        assert inventory["required_inventory"] >= 0
        assert inventory["shortage"] >= 0
        assert inventory["safety_stock"] >= 0

        assert supply["lead_time_days"] >= 0
        assert supply["lead_time_demand"] >= 0
        assert supply["recommended_order"] >= 0
        assert supply["order_unit"]


def test_demand_intelligence_contract(report):
    actions = report["actions"]

    assert len(actions) > 0

    for action in actions[:20]:
        assert "demand_risk" in action
        assert "demand_spike" in action
        assert "model_confidence" in action

        assert action["demand_risk"] is not None
        assert action["demand_spike"] is not None

        confidence = action["model_confidence"]

        assert isinstance(confidence, (int, float))
        assert 0 <= confidence <= 100


def test_business_action_contract(report):
    actions = report["actions"]

    assert len(actions) > 0

    for action in actions[:20]:
        assert "recommendation" in action
        assert "priority" in action

        recommendation = action["recommendation"]

        assert "action" in recommendation
        assert "reason" in recommendation

        assert recommendation["action"]
        assert recommendation["reason"]
        assert action["priority"]


def test_risk_contract(report):
    actions = report["actions"]

    assert len(actions) > 0

    for action in actions[:20]:
        assert "demand_risk" in action
        assert "stockout_risk" in action
        assert "supply" in action

        supply = action["supply"]

        assert "supplier_risk" in supply

        assert action["demand_risk"] is not None
        assert action["stockout_risk"] is not None
        assert supply["supplier_risk"] is not None


def test_key_drivers_contract(report):
    actions = report["actions"]

    assert len(actions) > 0

    for action in actions[:20]:
        assert "key_drivers" in action

        drivers = action["key_drivers"]

        assert isinstance(drivers, list)

        for driver in drivers:
            assert isinstance(driver, str)


def test_service_report_contains_summary(service_report):
    assert isinstance(service_report, dict)

    assert "report_date" in service_report
    assert "summary" in service_report
    assert "actions" in service_report

    assert isinstance(service_report["summary"], dict)
    assert isinstance(service_report["actions"], list)


def test_service_report_is_json_serializable(service_report):
    payload = json.dumps(service_report)

    assert isinstance(payload, str)
    assert len(payload) > 0


def test_service_output_can_be_written(service):
    result = service.generate(
        include_normal=False,
        write_output=True,
    )

    assert isinstance(result, dict)
    assert "report_date" in result
    assert "summary" in result
    assert "actions" in result


def test_daily_report_validation(report):
    validation = report.get("validation")

    assert validation is not None
    assert validation.get("passed") is True
    assert validation.get("errors") == []

