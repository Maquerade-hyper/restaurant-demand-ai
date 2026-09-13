import pandas as pd
import pytest

from app.intelligence.scenario import (
    ScenarioEngineService,
)


def sample_forecast():

    return pd.DataFrame(
        [
            {
                "outlet_id": "O001",
                "product_id": "P001",
                "date": "2025-01-01",
                "baseline_demand": 100.0,
            },
            {
                "outlet_id": "O001",
                "product_id": "P001",
                "date": "2025-01-02",
                "baseline_demand": 120.0,
            },
            {
                "outlet_id": "O002",
                "product_id": "P002",
                "date": "2025-01-01",
                "baseline_demand": 50.0,
            },
        ]
    )


def test_complete_scenario_engine():

    service = (
        ScenarioEngineService()
    )

    baseline = service.create_scenario(
        name="baseline"
    )

    promotion = service.create_scenario(
        name="promotion",
        promotion_multiplier=1.20,
    )

    result = service.analyze(
        forecast=sample_forecast(),
        scenarios=[
            baseline,
            promotion,
        ],
    )

    assert len(result) == 6

    assert set(
        result["scenario_name"]
    ) == {
        "baseline",
        "promotion",
    }

    assert (
        result["scenario_demand"]
        >= 0
    ).all()

    validation = service.validate(
        result
    )

    assert validation["passed"] is True


def test_baseline_scenario_is_unchanged():

    service = (
        ScenarioEngineService()
    )

    scenario = service.create_scenario(
        name="baseline"
    )

    result = service.analyze(
        forecast=sample_forecast(),
        scenarios=[scenario],
    )

    assert (
        result["scenario_multiplier"]
        == 1.0
    ).all()

    assert (
        result["scenario_demand"]
        == result["baseline_demand"]
    ).all()

    assert (
        result["absolute_change"]
        == 0.0
    ).all()


def test_combined_scenario():

    service = (
        ScenarioEngineService()
    )

    scenario = service.create_scenario(
        name="festival_promotion",
        holiday_multiplier=1.10,
        promotion_multiplier=1.20,
        event_multiplier=1.05,
    )

    result = service.analyze(
        forecast=sample_forecast(),
        scenarios=[scenario],
    )

    expected_multiplier = (
        1.10 * 1.20 * 1.05
    )

    assert (
        result["scenario_multiplier"]
        .round(8)
        == round(expected_multiplier, 8)
    ).all()

    assert (
        result["scenario_demand"]
        > result["baseline_demand"]
    ).all()


def test_invalid_multiplier_rejected():

    service = (
        ScenarioEngineService()
    )

    with pytest.raises(ValueError):

        service.create_scenario(
            name="invalid",
            promotion_multiplier=3.0,
        )


def test_truth_leakage_rejected():

    service = (
        ScenarioEngineService()
    )

    result = service.analyze(
        forecast=sample_forecast(),
        scenarios=[
            service.create_scenario(
                name="baseline"
            )
        ],
    )

    result["true_demand"] = (
        result["baseline_demand"]
    )

    validation = service.validate(
        result
    )

    assert validation["passed"] is False

    assert any(
        "truth columns" in error
        for error in validation["errors"]
    )


def test_scenario_summary():

    service = (
        ScenarioEngineService()
    )

    scenarios = [
        service.create_scenario(
            name="baseline"
        ),
        service.create_scenario(
            name="promotion",
            promotion_multiplier=1.20,
        ),
    ]

    result = service.analyze(
        forecast=sample_forecast(),
        scenarios=scenarios,
    )

    summary = service.summarize(
        result
    )

    assert len(summary) == 2

    assert set(
        summary["scenario_name"]
    ) == {
        "baseline",
        "promotion",
    }

    promotion = summary[
        summary["scenario_name"]
        == "promotion"
    ].iloc[0]

    assert (
        promotion["scenario_demand"]
        > promotion["baseline_demand"]
    )