from __future__ import annotations

import pandas as pd

from .bom import normalize_bom
from .demand_to_supply import build_supply_demand
from .recommendation import calculate_supply_recommendations
from .supplier import normalize_supplier_contract


class SupplyIntelligenceService:

    def __init__(
        self,
        bom: pd.DataFrame | None = None,
        supplier_contract: pd.DataFrame | None = None,
        safety_stock_days: float = 1.0,
    ) -> None:

        self.bom = normalize_bom(bom)

        self.supplier_contract = normalize_supplier_contract(
            supplier_contract
        )

        self.safety_stock_days = float(
            safety_stock_days
        )

        if self.safety_stock_days < 0:
            raise ValueError(
                "safety_stock_days cannot be negative"
            )

    def demand_to_supply(
        self,
        demand: pd.DataFrame,
    ) -> pd.DataFrame:

        return build_supply_demand(
            demand=demand,
            bom=self.bom,
        )

    def recommend(
        self,
        demand: pd.DataFrame,
        inventory: pd.DataFrame,
    ) -> pd.DataFrame:

        supply_demand = self.demand_to_supply(
            demand
        )

        return calculate_supply_recommendations(
            supply_demand=supply_demand,
            inventory=inventory,
            supplier_contract=self.supplier_contract,
            safety_stock_days=self.safety_stock_days,
        )

    def validate(
        self,
        result: pd.DataFrame,
    ) -> dict:

        errors: list[str] = []

        required = {
            "outlet_id",
            "date",
            "supply_product_id",
            "required_supply",
            "current_inventory",
            "incoming_stock",
            "lead_time_demand",
            "safety_stock",
            "target_inventory",
            "inventory_position",
            "shortage",
            "recommended_order_quantity",
            "supply_risk_ratio",
            "supply_risk",
            "order_required",
        }

        missing = required - set(result.columns)

        if missing:
            errors.append(
                f"result missing columns: {sorted(missing)}"
            )

            return {
                "passed": False,
                "errors": errors,
                "rows": int(len(result)),
            }

        if result.empty:
            errors.append("supply intelligence result is empty")

            return {
                "passed": False,
                "errors": errors,
                "rows": 0,
            }

        # ---------------------------------------------------------
        # 1. Numeric validity
        # ---------------------------------------------------------

        numeric_columns = [
            "required_supply",
            "current_inventory",
            "incoming_stock",
            "lead_time_demand",
            "safety_stock",
            "target_inventory",
            "inventory_position",
            "shortage",
            "recommended_order_quantity",
        ]

        for column in numeric_columns:

            numeric = pd.to_numeric(
                result[column],
                errors="coerce",
            )

            if numeric.isna().any():
                errors.append(
                    f"{column} contains invalid values"
                )
                continue

            if not numeric.map(
                lambda value: pd.notna(value)
                and value != float("inf")
                and value != float("-inf")
            ).all():
                errors.append(
                    f"{column} contains infinite values"
                )

            if (numeric < 0).any():
                errors.append(
                    f"{column} contains negative values"
                )

        # ---------------------------------------------------------
        # 2. Supply risk ratio
        # ---------------------------------------------------------

        risk_ratio = pd.to_numeric(
            result["supply_risk_ratio"],
            errors="coerce",
        )

        if risk_ratio.isna().any():
            errors.append(
                "supply_risk_ratio contains invalid values"
            )
        else:
            if (risk_ratio < 0).any() or (
                risk_ratio > 1
            ).any():
                errors.append(
                    "supply_risk_ratio outside [0, 1]"
                )

        # ---------------------------------------------------------
        # 3. Risk class validation
        # ---------------------------------------------------------

        allowed_risk = {
            "low",
            "moderate",
            "high",
        }

        invalid_risk = set(
            result["supply_risk"]
            .dropna()
            .unique()
        ) - allowed_risk

        if invalid_risk:
            errors.append(
                "invalid supply risk classes: "
                f"{sorted(invalid_risk)}"
            )

        # ---------------------------------------------------------
        # 4. Order flag consistency
        # ---------------------------------------------------------

        expected_order_flag = (
            result["recommended_order_quantity"] > 0
        )

        if not (
            expected_order_flag
            == result["order_required"]
        ).all():

            errors.append(
                "order_required inconsistent with "
                "recommended_order_quantity"
            )

        # ---------------------------------------------------------
        # 5. Shortage calculation consistency
        # ---------------------------------------------------------

        calculated_shortage = (
            result["target_inventory"]
            - result["inventory_position"]
        ).clip(lower=0.0)

        shortage_difference = (
            result["shortage"]
            - calculated_shortage
        ).abs()

        if (shortage_difference > 1e-8).any():
            errors.append(
                "shortage inconsistent with "
                "target_inventory and inventory_position"
            )

        # ---------------------------------------------------------
        # 6. Recommended order must cover shortage
        # ---------------------------------------------------------

        invalid_shortage_coverage = (
            result["recommended_order_quantity"]
            + 1e-9
            < calculated_shortage
        )

        if invalid_shortage_coverage.any():
            errors.append(
                "recommended order is below calculated shortage"
            )

        # ---------------------------------------------------------
        # 7. No unnecessary order
        #
        # If inventory position already covers target inventory,
        # recommended order must be zero.
        # ---------------------------------------------------------

        already_sufficient = (
            result["inventory_position"]
            >= result["target_inventory"]
        )

        unnecessary_order = (
            already_sufficient
            & (
                result["recommended_order_quantity"]
                > 1e-9
            )
        )

        if unnecessary_order.any():
            errors.append(
                "order recommended despite sufficient inventory"
            )

        # ---------------------------------------------------------
        # 8. Target inventory consistency
        # ---------------------------------------------------------

        calculated_target = (
            result["lead_time_demand"]
            + result["safety_stock"]
        )

        target_difference = (
            result["target_inventory"]
            - calculated_target
        ).abs()

        if (target_difference > 1e-8).any():
            errors.append(
                "target_inventory inconsistent with "
                "lead_time_demand and safety_stock"
            )

        # ---------------------------------------------------------
        # 9. Inventory position consistency
        # ---------------------------------------------------------

        calculated_position = (
            result["current_inventory"]
            + result["incoming_stock"]
        )

        position_difference = (
            result["inventory_position"]
            - calculated_position
        ).abs()

        if (position_difference > 1e-8).any():
            errors.append(
                "inventory_position inconsistent with "
                "current_inventory and incoming_stock"
            )

        # ---------------------------------------------------------
        # 10. Production leakage protection
        #
        # True demand / ground-truth fields must never enter the
        # production supply intelligence output.
        # ---------------------------------------------------------

        forbidden = {
            "true_demand",
            "lost_demand_truth",
            "demand_truth",
            "actual_demand",
        }

        leakage = forbidden.intersection(
            result.columns
        )

        if leakage:
            errors.append(
                f"truth columns present: {sorted(leakage)}"
            )

        # ---------------------------------------------------------
        # 11. Identity validity
        # ---------------------------------------------------------

        if result["outlet_id"].isna().any():
            errors.append(
                "outlet_id contains null values"
            )

        if result["supply_product_id"].isna().any():
            errors.append(
                "supply_product_id contains null values"
            )

        if result["date"].isna().any():
            errors.append(
                "date contains null values"
            )

        # ---------------------------------------------------------
        # Final result
        # ---------------------------------------------------------

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "rows": int(len(result)),
        }

    def analyze(
        self,
        demand: pd.DataFrame,
        inventory: pd.DataFrame,
    ) -> pd.DataFrame:

        result = self.recommend(
            demand=demand,
            inventory=inventory,
        )

        validation = self.validate(
            result
        )

        if not validation["passed"]:
            raise ValueError(
                "; ".join(
                    validation["errors"]
                )
            )

        return result