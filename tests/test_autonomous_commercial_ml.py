import math

from app.intelligence.autonomous import (
    AutonomousCommercialMLService,
    AutonomousController,
    AutonomousRequest,
    AutonomousHealthMonitor,
    AutonomousGovernance,
)


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


def test_health_monitor():

    monitor = (
        AutonomousHealthMonitor()
    )

    report = monitor.evaluate(
        data_available=True,
        model_available=True,
        feature_contract_valid=True,
        prediction=10.0,
        drift_signal=False,
    )

    assert report.prediction_valid
    assert monitor.healthy(report)


def test_health_rejects_invalid_prediction():

    monitor = (
        AutonomousHealthMonitor()
    )

    report = monitor.evaluate(
        data_available=True,
        model_available=True,
        feature_contract_valid=True,
        prediction=float("nan"),
    )

    assert not report.prediction_valid
    assert not monitor.healthy(report)


def test_commercial_decision():

    controller = (
        AutonomousController()
    )

    decision = controller.execute(
        make_request()
    )

    assert (
        decision.raw_forecast
        == 100.0
    )

    assert (
        decision.scenario_adjusted_forecast
        == 100.0
    )

    assert (
        decision.lead_time_demand
        == 200.0
    )

    assert (
        decision.shortage
        == 170.0
    )

    assert (
        decision.recommended_order
        == 170.0
    )

    assert (
        decision.stockout_risk
        == "high"
    )


def test_moq_and_multiple():

    request = make_request()

    request.forecast_demand = 3
    request.current_inventory = 0
    request.lead_time_days = 1
    request.safety_stock = 0
    request.minimum_order_quantity = 10
    request.order_multiple = 5

    controller = (
        AutonomousController()
    )

    decision = controller.execute(
        request
    )

    assert (
        decision.recommended_order
        == 10
    )


def test_no_replenishment():

    request = make_request()

    request.forecast_demand = 10
    request.current_inventory = 100
    request.lead_time_days = 1
    request.safety_stock = 10

    controller = (
        AutonomousController()
    )

    decision = controller.execute(
        request
    )

    assert (
        decision.commercial_action
        == "NO_REPLENISHMENT"
    )

    assert (
        decision.recommended_order
        == 0
    )


def test_low_confidence_requires_review():

    request = make_request()

    request.demand_confidence = 0.30

    controller = (
        AutonomousController()
    )

    decision = controller.execute(
        request
    )

    assert (
        decision.commercial_action
        == "REVIEW_REQUIRED"
    )

    assert (
        decision.governance.allowed
        is False
    )


def test_drift_requires_review():

    request = make_request()

    controller = (
        AutonomousController()
    )

    decision = controller.execute(
        request,
        drift_signal=True,
    )

    assert (
        decision.commercial_action
        == "REVIEW_REQUIRED"
    )

    assert (
        decision.governance.allowed
        is False
    )


def test_high_risk_requires_review():

    request = make_request()

    request.high_risk = True

    controller = (
        AutonomousController()
    )

    decision = controller.execute(
        request
    )

    assert (
        decision.commercial_action
        == "REVIEW_REQUIRED"
    )


def test_unhealthy_model_is_held():

    controller = (
        AutonomousController()
    )

    decision = controller.execute(
        make_request(),
        model_available=False,
    )

    assert (
        decision.commercial_action
        == "HOLD"
    )

    assert (
        decision.governance.allowed
        is False
    )


def test_service_validation():

    service = (
        AutonomousCommercialMLService()
    )

    decision = service.decide(
        make_request()
    )

    result = service.validate(
        decision
    )

    assert result["passed"] is True
    assert result["errors"] == []


def test_batch():

    service = (
        AutonomousCommercialMLService()
    )

    requests = [
        make_request(),
        make_request(),
        make_request(),
    ]

    result = service.batch(
        requests
    )

    assert len(result) == 3

    for item in result:
        assert (
            item["outlet_id"]
            == "O001"
        )

        assert math.isfinite(
            item["recommended_order"]
        )


def test_scenario_adjustment():

    request = make_request()

    request.scenario_multiplier = 1.20

    controller = (
        AutonomousController()
    )

    decision = controller.execute(
        request
    )

    assert (
        decision.scenario_adjusted_forecast
        == 120.0
    )

    assert (
        decision.lead_time_demand
        == 240.0
    )


def test_negative_forecast_rejected():

    request = make_request()

    request.forecast_demand = -1

    controller = (
        AutonomousController()
    )

    try:
        controller.execute(
            request
        )

    except ValueError:
        assert True

    else:
        assert False


def test_feature_contract_gate():

    controller = (
        AutonomousController()
    )

    decision = controller.execute(
        make_request(),
        feature_contract_valid=False,
    )

    assert (
        decision.commercial_action
        == "HOLD"
    )

    assert (
        decision.governance.allowed
        is False
    )