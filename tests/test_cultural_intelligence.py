import pandas as pd
import pytest

from app.intelligence.cultural import (
    CulturalDemographicIntelligenceService,
)


def sample_demand():

    return pd.DataFrame(
        [
            {
                "outlet_id": "O001",
                "product_id": "P001",
                "date": "2025-01-01",
                "demand": 20.0,
            },
            {
                "outlet_id": "O001",
                "product_id": "P001",
                "date": "2025-01-02",
                "demand": 25.0,
            },
            {
                "outlet_id": "O001",
                "product_id": "P001",
                "date": "2025-01-03",
                "demand": 30.0,
            },
        ]
    )


def sample_calendar():

    return pd.DataFrame(
        [
            {
                "outlet_id": "O001",
                "date": "2025-01-01",
                "holiday_importance": 1.0,
                "religious_importance": 0.8,
                "event_importance": 0.2,
            },
            {
                "outlet_id": "O001",
                "date": "2025-01-02",
                "holiday_importance": 0.0,
                "religious_importance": 0.0,
                "event_importance": 0.0,
            },
            {
                "outlet_id": "O001",
                "date": "2025-01-03",
                "holiday_importance": 0.0,
                "religious_importance": 0.0,
                "event_importance": 0.0,
            },
        ]
    )


def sample_demographics():

    return pd.DataFrame(
        [
            {
                "outlet_id": "O001",
                "population_density": 5000,
                "tourism_index": 0.6,
                "student_index": 0.4,
                "business_index": 0.7,
                "residential_index": 0.5,
                "religious_population_share": 0.6,
                "young_population_share": 0.5,
                "working_population_share": 0.7,
                "dominant_religious_context": "mixed",
                "demographic_context": "urban",
                "city_class": "large",
            }
        ]
    )


def test_complete_cultural_intelligence():

    service = (
        CulturalDemographicIntelligenceService()
    )

    result = service.analyze(
        demand=sample_demand(),
        calendar=sample_calendar(),
        demographics=sample_demographics(),
    )

    assert not result.empty

    required = {
        "outlet_id",
        "product_id",
        "date",
        "demand",
        "holiday_importance",
        "religious_importance",
        "event_importance",
        "religious_population_share",
        "cultural_context_score",
        "historical_demand_reference",
        "cultural_demand_pressure",
    }

    assert required.issubset(
        result.columns
    )

    assert (
        result["cultural_context_score"]
        .between(0, 1)
        .all()
    )

    assert (
        result["religious_population_share"]
        .between(0, 1)
        .all()
    )

    assert (
        result["cultural_demand_pressure"]
        >= 0
    ).all()


def test_calendar_effects_are_loaded():

    service = (
        CulturalDemographicIntelligenceService()
    )

    result = service.prepare(
        demand=sample_demand(),
        calendar=sample_calendar(),
        demographics=sample_demographics(),
    )

    first = result.iloc[0]

    assert first["holiday_importance"] == 1.0
    assert first["religious_importance"] == 0.8
    assert first["event_importance"] == 0.2


def test_demographic_context_is_loaded():

    service = (
        CulturalDemographicIntelligenceService()
    )

    result = service.prepare(
        demand=sample_demand(),
        calendar=sample_calendar(),
        demographics=sample_demographics(),
    )

    assert (
        result["religious_population_share"]
        == 0.6
    ).all()

    assert (
        result["tourism_index"]
        == 0.6
    ).all()


def test_truth_columns_are_rejected():

    service = (
        CulturalDemographicIntelligenceService()
    )

    result = service.prepare(
        demand=sample_demand(),
        calendar=sample_calendar(),
        demographics=sample_demographics(),
    )

    result["true_demand"] = result["demand"]

    validation = service.validate(
        result
    )

    assert validation["passed"] is False

    assert any(
        "truth columns" in error
        for error in validation["errors"]
    )


def test_duplicate_records_are_rejected():

    service = (
        CulturalDemographicIntelligenceService()
    )

    result = service.prepare(
        demand=sample_demand(),
        calendar=sample_calendar(),
        demographics=sample_demographics(),
    )

    result = pd.concat(
        [result, result.iloc[[0]]],
        ignore_index=True,
    )

    validation = service.validate(
        result
    )

    assert validation["passed"] is False

    assert any(
        "duplicate" in error
        for error in validation["errors"]
    )


def test_invalid_context_is_rejected():

    service = (
        CulturalDemographicIntelligenceService()
    )

    calendar = sample_calendar()

    calendar.loc[
        0,
        "religious_importance"
    ] = 2.0

    with pytest.raises(ValueError):

        service.analyze(
            demand=sample_demand(),
            calendar=calendar,
            demographics=sample_demographics(),
        )