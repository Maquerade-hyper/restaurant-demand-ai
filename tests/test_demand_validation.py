import numpy as np
import pandas as pd

from app.intelligence.demand.validation.service import (
    DemandIntelligenceValidationService,
)


def make_data():
    dates = pd.date_range(
        "2025-01-01",
        periods=10,
        freq="D",
    )

    truth = pd.DataFrame(
        {
            "outlet_id": ["O001"] * 10,
            "product_id": ["P001"] * 10,
            "date": dates,
            "true_demand": [
                10.0,
                11.0,
                12.0,
                10.0,
                14.0,
                15.0,
                13.0,
                12.0,
                16.0,
                18.0,
            ],
            "observed_sales": [
                10.0,
                11.0,
                12.0,
                10.0,
                10.0,
                15.0,
                13.0,
                12.0,
                12.0,
                18.0,
            ],
            "lost_demand": [
                0.0,
                0.0,
                0.0,
                0.0,
                4.0,
                0.0,
                0.0,
                0.0,
                4.0,
                0.0,
            ],
            "stockout": [
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
            ],
        }
    )

    intelligence = pd.DataFrame(
        {
            "outlet_id": ["O001"] * 10,
            "product_id": ["P001"] * 10,
            "date": dates,
            "quantity_sold": truth["observed_sales"],
            "deconstrained_demand": [
                10.0,
                11.0,
                12.0,
                10.0,
                13.0,
                15.0,
                13.0,
                12.0,
                15.0,
                18.0,
            ],
            "estimated_lost_demand": [
                0.0,
                0.0,
                0.0,
                0.0,
                3.0,
                0.0,
                0.0,
                0.0,
                3.0,
                0.0,
            ],
            "estimated_fulfillment_rate": [
                1.0,
                1.0,
                1.0,
                1.0,
                10.0 / 13.0,
                1.0,
                1.0,
                1.0,
                12.0 / 15.0,
                1.0,
            ],
            "demand_constrained_estimated": [
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
            ],
            "stockout": truth["stockout"],
            "censoring_strength": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.8,
                0.0,
                0.0,
                0.0,
                0.7,
                0.0,
            ],
            "censoring_class": [
                "not_censored",
                "not_censored",
                "not_censored",
                "not_censored",
                "strong",
                "not_censored",
                "not_censored",
                "not_censored",
                "moderate",
                "not_censored",
            ],
            "censored_demand_flag": [
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
            ],
        }
    )

    return intelligence, truth


def test_validation_runs():
    intelligence, truth = make_data()

    service = DemandIntelligenceValidationService()

    report = service.validate(
        intelligence,
        truth,
    )

    assert report["overall"]["rows_compared"] == 10
    assert report["overall"]["demand_mae"] >= 0
    assert report["overall"]["lost_demand_mae"] >= 0


def test_truth_validation():
    intelligence, truth = make_data()

    service = DemandIntelligenceValidationService()

    report = service.validate(
        intelligence,
        truth,
    )

    assert (
        report["truth_validation"][
            "true_demand_ge_observed"
        ]
    )

    assert (
        report["truth_validation"][
            "lost_demand_nonnegative"
        ]
    )


def test_output_integrity():
    intelligence, truth = make_data()

    service = DemandIntelligenceValidationService()

    report = service.validate(
        intelligence,
        truth,
    )

    output = report["output_validation"]

    assert output["null_demand"] == 0
    assert output["null_lost_demand"] == 0
    assert output["demand_below_observed"] == 0
    assert output["negative_lost_demand"] == 0
    assert output["invalid_censoring_strength"] == 0


def test_stockout_validation():
    intelligence, truth = make_data()

    service = DemandIntelligenceValidationService()

    report = service.validate(
        intelligence,
        truth,
    )

    assert (
        report["stockout"]["stockout_rows"]
        == 2
    )

    assert (
        report["stockout"][
            "stockout_demand_mae"
        ]
        >= 0
    )


def test_acceptance():
    intelligence, truth = make_data()

    service = DemandIntelligenceValidationService()

    report = service.validate(
        intelligence,
        truth,
    )

    acceptance = service.acceptance_check(
        report
    )

    assert acceptance["passed"] is True
    assert all(
        acceptance["checks"].values()
    )