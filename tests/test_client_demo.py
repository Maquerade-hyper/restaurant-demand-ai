from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.demo.formatter import format_client_report
from app.demo.schemas import ClientDemoRequest
from app.demo.service import ClientDemoService


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "demand_censoring_intelligence.csv"
)


@pytest.fixture(scope="module")
def service():
    if not DATASET.exists():
        pytest.skip(
            "Synthetic Part 16 dataset is not available."
        )

    return ClientDemoService(DATASET)


@pytest.fixture(scope="module")
def demo_request(service):
    """
    Build a deterministic valid demonstration request.

    The fixture is deliberately named demo_request instead of request
    because 'request' is a reserved pytest fixture name.
    """

    series = (
        service.df[
            ["outlet_id", "product_id", "date"]
        ]
        .sort_values("date")
        .groupby(
            ["outlet_id", "product_id"],
            as_index=False,
        )
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            rows=("date", "size"),
        )
    )

    series = series[
        series["rows"] >= 60
    ]

    if series.empty:
        pytest.skip(
            "No sufficiently long series available."
        )

    selected = series.sort_values(
        ["rows", "outlet_id", "product_id"],
        ascending=[False, True, True],
    ).iloc[0]

    forecast_date = (
        pd.Timestamp(selected["last_date"])
        + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    return ClientDemoRequest(
        outlet_id=str(selected["outlet_id"]),
        product_id=str(selected["product_id"]),
        forecast_date=forecast_date,
        horizon_days=7,
        lead_time_days=3,
    )


def test_request_contract():
    demo_request = ClientDemoRequest(
        outlet_id="O001",
        product_id="P001",
        forecast_date="2025-12-31",
        horizon_days=7,
        lead_time_days=3,
    )

    assert demo_request.outlet_id == "O001"
    assert demo_request.product_id == "P001"
    assert demo_request.horizon_days == 7
    assert demo_request.lead_time_days == 3


def test_invalid_horizon():
    with pytest.raises(ValueError):
        ClientDemoRequest(
            outlet_id="O001",
            product_id="P001",
            forecast_date="2025-12-31",
            horizon_days=5,
        )


def test_service_loads_synthetic_data(service):
    assert len(service.df) > 0
    assert service.demand_col in service.df.columns
    assert service.dataset_path.exists()


def test_demo_runs_end_to_end(
    service,
    demo_request,
):
    result = service.run(
        demo_request
    )

    assert (
        result.outlet["outlet_id"]
        == demo_request.outlet_id
    )

    assert (
        result.product["product_id"]
        == demo_request.product_id
    )

    assert result.forecast["d1"] >= 0
    assert (
        result.forecast["d3"]
        >= result.forecast["d1"]
    )

    assert (
        result.forecast["d7"]
        >= result.forecast["d3"]
    )

    assert (
        result.inventory["current_inventory"]
        >= 0
    )

    assert (
        result.inventory["shortage"]
        >= 0
    )

    assert (
        result.supply["recommended_order"]
        >= 0
    )

    assert result.autonomous_decision["action"] in {
        "HOLD",
        "REPLENISH",
        "REVIEW_REQUIRED",
    }


def test_forecast_time_firewall(
    service,
    demo_request,
):
    history = service._get_series(
        demo_request
    )

    forecast_date = pd.Timestamp(
        demo_request.forecast_date
    )

    assert (
        history["date"] < forecast_date
    ).all()


def test_result_validation(
    service,
    demo_request,
):
    result = service.run(
        demo_request
    )

    validation = service.validate_result(
        result
    )

    assert validation["passed"] is True

    assert (
        validation["forecast_time_firewall"]
        is True
    )

    assert validation["rows_loaded"] > 0


def test_client_formatter(
    service,
    demo_request,
):
    result = service.run(
        demo_request
    )

    report = format_client_report(
        result.to_dict()
    )

    assert (
        "RESTAURANT DEMAND AI"
        in report
    )

    assert "FORECAST" in report
    assert "INVENTORY" in report
    assert "SUPPLY" in report

    assert (
        "AUTONOMOUS RECOMMENDATION"
        in report
    )


def test_json_output_contract(
    service,
    demo_request,
    tmp_path,
):
    result = service.run(
        demo_request
    )

    payload = result.to_dict()

    output = (
        tmp_path
        / "client_demo.json"
    )

    output.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    loaded = json.loads(
        output.read_text(
            encoding="utf-8"
        )
    )

    required_sections = {
        "demo",
        "outlet",
        "product",
        "forecast",
        "demand_intelligence",
        "inventory",
        "supply",
        "autonomous_decision",
    }

    assert required_sections.issubset(
        loaded.keys()
    )


def test_no_future_rows_used(
    service,
    demo_request,
):
    history = service._get_series(
        demo_request
    )

    forecast_date = pd.Timestamp(
        demo_request.forecast_date
    )

    assert (
        history["date"].max()
        < forecast_date
    )


def test_result_contains_data_source(
    service,
    demo_request,
):
    result = service.run(
        demo_request
    )

    assert (
        result.data_source
    )

    assert (
        "synthetic"
        in result.production_note.lower()
    )


def test_result_has_all_client_layers(
    service,
    demo_request,
):
    result = service.run(
        demo_request
    )

    payload = result.to_dict()

    layers = [
        "forecast",
        "demand_intelligence",
        "inventory",
        "supply",
        "autonomous_decision",
    ]

    for layer in layers:
        assert layer in payload
        assert payload[layer]


def test_forecast_values_are_finite(
    service,
    demo_request,
):
    result = service.run(
        demo_request
    )

    for key in [
        "d1",
        "d3",
        "d7",
    ]:
        value = float(
            result.forecast[key]
        )

        assert pd.notna(value)
        assert value >= 0


def test_inventory_supply_consistency(
    service,
    demo_request,
):
    result = service.run(
        demo_request
    )

    inventory = result.inventory
    supply = result.supply

    assert (
        inventory["shortage"]
        >= 0
    )

    assert (
        supply["recommended_order"]
        >= 0
    )

    if inventory["shortage"] > 0:
        assert (
            supply["recommended_order"]
            >= 0
        )


def test_autonomous_decision_consistency(
    service,
    demo_request,
):
    result = service.run(
        demo_request
    )

    inventory = result.inventory
    decision = result.autonomous_decision

    if (
        inventory["shortage"] > 0
        and decision["action"] != "REVIEW_REQUIRED"
    ):
        assert decision["action"] == "REPLENISH"


def test_client_output_is_json_serializable(
    service,
    demo_request,
):
    result = service.run(
        demo_request
    )

    payload = result.to_dict()

    encoded = json.dumps(
        payload,
        allow_nan=False,
    )

    assert encoded