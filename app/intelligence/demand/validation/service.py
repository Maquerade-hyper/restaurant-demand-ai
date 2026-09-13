from __future__ import annotations

import pandas as pd

from .metrics import (
    bias,
    constraint_accuracy,
    fulfillment_mae,
    mae,
    recovery_ratio,
    rmse,
    validate_output_integrity,
    validate_truth_relationships,
)


class DemandIntelligenceValidationService:

    def validate(
        self,
        intelligence: pd.DataFrame,
        truth: pd.DataFrame,
    ) -> dict:
        if intelligence.empty:
            raise ValueError(
                "Demand intelligence output is empty."
            )

        if truth.empty:
            raise ValueError(
                "Demand truth is empty."
            )

        truth_check = validate_truth_relationships(
            truth
        )

        output_check = validate_output_integrity(
            intelligence
        )

        merged = intelligence.merge(
            truth[
                [
                    "outlet_id",
                    "product_id",
                    "date",
                    "true_demand",
                    "observed_sales",
                    "lost_demand",
                ]
            ],
            on=[
                "outlet_id",
                "product_id",
                "date",
            ],
            how="inner",
            validate="one_to_one",
        )

        if merged.empty:
            raise ValueError(
                "No matching rows between "
                "intelligence output and truth."
            )

        estimated_demand = pd.to_numeric(
            merged["deconstrained_demand"],
            errors="coerce",
        ).fillna(0.0)

        true_demand = pd.to_numeric(
            merged["true_demand"],
            errors="coerce",
        ).fillna(0.0)

        estimated_lost = pd.to_numeric(
            merged["estimated_lost_demand"],
            errors="coerce",
        ).fillna(0.0)

        actual_lost = pd.to_numeric(
            merged["lost_demand"],
            errors="coerce",
        ).fillna(0.0)

        observed = pd.to_numeric(
            merged["quantity_sold"],
            errors="coerce",
        ).fillna(0.0)

        estimated_constrained = (
            merged["demand_constrained_estimated"]
            .fillna(False)
            .astype(bool)
        )

        actual_constrained = (
            actual_lost > 1e-9
        )

        stockout = (
            merged["stockout"]
            .fillna(False)
            .astype(bool)
        )

        overall = {
            "rows_compared": int(len(merged)),
            "demand_mae": mae(
                true_demand,
                estimated_demand,
            ),
            "demand_rmse": rmse(
                true_demand,
                estimated_demand,
            ),
            "demand_bias": bias(
                true_demand,
                estimated_demand,
            ),
            "demand_recovery": recovery_ratio(
                true_demand,
                estimated_demand,
            ),
            "lost_demand_mae": mae(
                actual_lost,
                estimated_lost,
            ),
            "lost_demand_rmse": rmse(
                actual_lost,
                estimated_lost,
            ),
            "lost_demand_bias": bias(
                actual_lost,
                estimated_lost,
            ),
            "lost_demand_recovery": recovery_ratio(
                actual_lost,
                estimated_lost,
            ),
            "constraint_accuracy": constraint_accuracy(
                actual_constrained,
                estimated_constrained,
            ),
            "fulfillment_mae": fulfillment_mae(
                observed,
                true_demand,
                merged[
                    "estimated_fulfillment_rate"
                ],
            ),
        }

        stockout_rows = merged.loc[
            stockout
        ].copy()

        if stockout_rows.empty:
            stockout_metrics = {
                "stockout_rows": 0,
                "stockout_demand_mae": 0.0,
                "stockout_demand_rmse": 0.0,
                "stockout_demand_bias": 0.0,
                "stockout_lost_demand_mae": 0.0,
                "stockout_lost_demand_recovery": 1.0,
            }
        else:
            stockout_true = pd.to_numeric(
                stockout_rows["true_demand"],
                errors="coerce",
            ).fillna(0.0)

            stockout_estimated = pd.to_numeric(
                stockout_rows[
                    "deconstrained_demand"
                ],
                errors="coerce",
            ).fillna(0.0)

            stockout_actual_lost = pd.to_numeric(
                stockout_rows["lost_demand"],
                errors="coerce",
            ).fillna(0.0)

            stockout_estimated_lost = pd.to_numeric(
                stockout_rows[
                    "estimated_lost_demand"
                ],
                errors="coerce",
            ).fillna(0.0)

            stockout_metrics = {
                "stockout_rows": int(
                    len(stockout_rows)
                ),
                "stockout_demand_mae": mae(
                    stockout_true,
                    stockout_estimated,
                ),
                "stockout_demand_rmse": rmse(
                    stockout_true,
                    stockout_estimated,
                ),
                "stockout_demand_bias": bias(
                    stockout_true,
                    stockout_estimated,
                ),
                "stockout_lost_demand_mae": mae(
                    stockout_actual_lost,
                    stockout_estimated_lost,
                ),
                "stockout_lost_demand_recovery": recovery_ratio(
                    stockout_actual_lost,
                    stockout_estimated_lost,
                ),
            }

        return {
            "truth_validation": truth_check,
            "output_validation": output_check,
            "overall": overall,
            "stockout": stockout_metrics,
        }

    def acceptance_check(
        self,
        report: dict,
    ) -> dict:
        truth = report["truth_validation"]
        output = report["output_validation"]
        overall = report["overall"]

        checks = {
            "truth_demand_relationship": (
                truth["true_demand_ge_observed"]
            ),
            "truth_lost_demand_relationship": (
                truth["lost_demand_nonnegative"]
            ),
            "truth_accounting": (
                truth["max_accounting_error"]
                <= 1e-3
            ),
            "output_no_missing_demand": (
                output["null_demand"] == 0
            ),
            "output_no_missing_lost_demand": (
                output["null_lost_demand"] == 0
            ),
            "output_demand_not_below_sales": (
                output["demand_below_observed"] == 0
            ),
            "output_lost_demand_nonnegative": (
                output["negative_lost_demand"] == 0
            ),
            "censoring_strength_valid": (
                output[
                    "invalid_censoring_strength"
                ]
                == 0
            ),
            "demand_recovery_positive": (
                overall["demand_recovery"] > 0
            ),
            "lost_demand_recovery_positive": (
                overall[
                    "lost_demand_recovery"
                ] > 0
            ),
        }

        return {
            "checks": checks,
            "passed": all(checks.values()),
        }