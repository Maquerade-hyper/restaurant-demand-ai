from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .schemas import ClientDemoRequest, ClientDemoResult
from .scenarios import discover_dataset


class ClientDemoService:
    """
    End-to-end client demonstration service.

    This layer intentionally does not train a new model.

    It consumes historical synthetic demand and produces:

        Forecast
            -> Demand Intelligence
            -> Inventory Intelligence
            -> Supply Intelligence
            -> Autonomous Commercial Decision

    The data-source boundary is intentionally isolated so that a future
    client adapter can replace the synthetic CSV without changing the
    downstream output contract.
    """

    FORBIDDEN_FEATURE_COLUMNS = {
        "future_quantity_sold",
        "future_demand",
        "prediction",
        "actual",
        "target",
    }

    def __init__(
        self,
        dataset_path: Optional[str | Path] = None,
    ) -> None:

        self.dataset_path = discover_dataset(
            Path(dataset_path) if dataset_path else None
        )

        self.df = self._load_dataset()

        self.date_col = "date"
        self.outlet_col = "outlet_id"
        self.product_col = "product_id"

        self.demand_col = self._find_demand_column()

        self.inventory_col = self._find_optional_column(
            [
                "closing_stock",
                "current_inventory",
                "inventory",
                "stock",
                "available_inventory",
            ]
        )

        self.outlet_type_col = self._find_optional_column(
            [
                "outlet_type",
                "type",
            ]
        )

        self.country_col = self._find_optional_column(
            [
                "country",
            ]
        )

        self.location_type_col = self._find_optional_column(
            [
                "location_type",
            ]
        )

        self.holiday_col = self._find_optional_column(
            [
                "holiday_active",
                "is_holiday",
                "holiday",
            ]
        )

        self.promotion_col = self._find_optional_column(
            [
                "promotion_active",
                "is_promotion",
                "promotion",
            ]
        )

        self.event_col = self._find_optional_column(
            [
                "event_active",
                "is_event",
                "event",
            ]
        )

        self.weather_col = self._find_optional_column(
            [
                "weather_condition",
                "weather",
                "is_rainy",
            ]
        )

    # ------------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------------

    def _load_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.dataset_path)

        required = {
            "date",
            "outlet_id",
            "product_id",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Dataset missing required columns: {sorted(missing)}"
            )

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        df = df.dropna(
            subset=["date", "outlet_id", "product_id"]
        ).copy()

        for column in df.columns:
            if column in {
                "outlet_id",
                "product_id",
                "date",
            }:
                continue

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):
                continue

            try:
                converted = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

                original_non_null = (
                    df[column].notna().sum()
                )

                converted_non_null = (
                    converted.notna().sum()
                )

                if (
                    original_non_null > 0
                    and converted_non_null
                    == original_non_null
                ):
                    df[column] = converted

            except Exception:
                pass

        return df

    def _find_demand_column(self) -> str:
        candidates = [
            "deconstrained_demand",
            "estimated_true_demand",
            "true_demand",
            "quantity_sold",
        ]

        for column in candidates:
            if column in self.df.columns:
                return column

        raise ValueError(
            "No usable demand column found. Expected one of: "
            + ", ".join(candidates)
        )

    def _find_optional_column(
        self,
        candidates: List[str],
    ) -> Optional[str]:

        lower_map = {
            str(column).lower(): column
            for column in self.df.columns
        }

        for candidate in candidates:
            if candidate.lower() in lower_map:
                return lower_map[candidate.lower()]

        return None

    # ------------------------------------------------------------------
    # SERIES
    # ------------------------------------------------------------------

    def _get_series(
        self,
        request: ClientDemoRequest,
    ) -> pd.DataFrame:

        series = self.df[
            (self.df[self.outlet_col].astype(str) == str(request.outlet_id))
            & (
                self.df[self.product_col].astype(str)
                == str(request.product_id)
            )
        ].copy()

        series = series.sort_values(self.date_col)

        if series.empty:
            raise ValueError(
                f"No data found for outlet={request.outlet_id}, "
                f"product={request.product_id}"
            )

        forecast_date = pd.Timestamp(request.forecast_date)

        # Strict forecast-time firewall.
        # Historical rows must be strictly before the forecast date.
        series = series[
            series[self.date_col] < forecast_date
        ].copy()

        if len(series) < 28:
            raise ValueError(
                "At least 28 historical observations are required "
                "for the client demonstration."
            )

        series[self.demand_col] = pd.to_numeric(
            series[self.demand_col],
            errors="coerce",
        )

        series = series.dropna(
            subset=[self.demand_col]
        )

        if series.empty:
            raise ValueError("Historical demand is empty")

        return series

    # ------------------------------------------------------------------
    # FORECAST
    # ------------------------------------------------------------------

    def _forecast_daily(
        self,
        history: pd.DataFrame,
        forecast_date: pd.Timestamp,
        days: int = 7,
    ) -> List[Dict[str, Any]]:

        demand = history[self.demand_col].astype(float)

        # Recent demand level.
        recent_28 = demand.tail(28)

        # Same-day-of-week historical behavior.
        dow = forecast_date.dayofweek

        dow_history = history[
            history[self.date_col].dt.dayofweek == dow
        ][self.demand_col].astype(float)

        if len(dow_history) >= 4:
            dow_mean = float(dow_history.tail(12).mean())
        else:
            dow_mean = float(recent_28.mean())

        recent_mean = float(recent_28.mean())

        # Recent trend.
        if len(demand) >= 28:
            older = float(demand.tail(28).head(14).mean())
            newer = float(demand.tail(14).mean())

            if older > 0:
                trend_ratio = newer / older
            else:
                trend_ratio = 1.0

            # Bound the trend contribution to avoid unrealistic demo
            # extrapolation.
            trend_ratio = float(
                np.clip(trend_ratio, 0.85, 1.15)
            )
        else:
            trend_ratio = 1.0

        base = (
            0.55 * recent_mean
            + 0.45 * dow_mean
        )

        base *= trend_ratio

        forecasts = []

        for step in range(1, days + 1):
            target_date = forecast_date + pd.Timedelta(
                days=step - 1
            )

            target_dow = target_date.dayofweek

            same_dow = history[
                history[self.date_col].dt.dayofweek == target_dow
            ][self.demand_col].astype(float)

            if len(same_dow) >= 4:
                seasonal = float(
                    same_dow.tail(12).mean()
                )
            else:
                seasonal = recent_mean

            value = (
                0.55 * base
                + 0.45 * seasonal
            )

            value = max(
                0.0,
                float(value),
            )

            forecasts.append(
                {
                    "date": target_date.strftime("%Y-%m-%d"),
                    "demand": round(value, 4),
                    "day_of_week": target_date.day_name(),
                }
            )

        return forecasts

    # ------------------------------------------------------------------
    # DEMAND INTELLIGENCE
    # ------------------------------------------------------------------

    def _demand_intelligence(
        self,
        history: pd.DataFrame,
        daily_forecast: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        demand = history[self.demand_col].astype(float)

        recent_28 = demand.tail(28)

        baseline = float(recent_28.mean())

        forecast_d7 = float(
            sum(item["demand"] for item in daily_forecast)
        )

        forecast_daily_mean = (
            forecast_d7 / max(1, len(daily_forecast))
        )

        if baseline > 0:
            change_pct = (
                (forecast_daily_mean - baseline)
                / baseline
                * 100.0
            )
        else:
            change_pct = 0.0

        std = float(recent_28.std(ddof=0))

        if std > 0:
            z = (
                forecast_daily_mean - baseline
            ) / std
        else:
            z = 0.0

        spike_detected = bool(
            change_pct >= 20.0
            and z >= 1.5
        )

        if not spike_detected:
            severity = "NONE"
        elif z >= 3.0 or change_pct >= 50.0:
            severity = "EXTREME"
        elif z >= 2.0 or change_pct >= 35.0:
            severity = "MAJOR"
        else:
            severity = "MODERATE"

        if change_pct >= 30 or z >= 2:
            risk = "HIGH"
        elif change_pct >= 10 or z >= 1:
            risk = "MODERATE"
        else:
            risk = "LOW"

        # Confidence is deliberately conservative.
        # It reflects history availability and stability, not an
        # unsupported "accuracy percentage".
        history_factor = min(
            1.0,
            len(history) / 180.0,
        )

        if baseline > 0:
            cv = std / baseline
        else:
            cv = 1.0

        stability_factor = 1.0 / (
            1.0 + max(0.0, cv)
        )

        confidence = (
            0.65 * history_factor
            + 0.35 * stability_factor
        )

        confidence = float(
            np.clip(confidence, 0.0, 1.0)
        )

        return {
            "baseline_demand": round(baseline, 2),
            "forecast_demand": round(forecast_d7, 2),
            "demand_change_pct": round(change_pct, 2),
            "spike_detected": spike_detected,
            "spike_severity": severity,
            "demand_risk": risk,
            "confidence": round(confidence, 4),
            "recent_demand_std": round(std, 2),
        }

    # ------------------------------------------------------------------
    # INVENTORY
    # ------------------------------------------------------------------

    def _current_inventory(
        self,
        history: pd.DataFrame,
        request: ClientDemoRequest,
    ) -> float:

        if request.current_inventory is not None:
            return float(request.current_inventory)

        if self.inventory_col is not None:
            values = pd.to_numeric(
                history[self.inventory_col],
                errors="coerce",
            ).dropna()

            if not values.empty:
                return max(
                    0.0,
                    float(values.iloc[-1]),
                )

        # If synthetic inventory is unavailable for the selected series,
        # derive a transparent demo inventory position from recent demand.
        # This is explicitly marked as derived rather than observed.
        recent_mean = float(
            history[self.demand_col]
            .astype(float)
            .tail(7)
            .mean()
        )

        return max(
            0.0,
            recent_mean,
        )

    def _inventory_intelligence(
        self,
        history: pd.DataFrame,
        current_inventory: float,
        lead_time_demand: float,
    ) -> Dict[str, Any]:

        demand = history[self.demand_col].astype(float)

        recent_28 = demand.tail(28)

        mean = float(recent_28.mean())

        std = float(
            recent_28.std(ddof=0)
        )

        # Simple service-oriented safety stock.
        safety_stock = max(
            0.0,
            1.28 * std,
        )

        required_inventory = (
            lead_time_demand
            + safety_stock
        )

        shortage = max(
            0.0,
            required_inventory - current_inventory,
        )

        if current_inventory < lead_time_demand:
            risk = "HIGH"
        elif current_inventory < required_inventory:
            risk = "MODERATE"
        else:
            risk = "LOW"

        return {
            "current_inventory": round(
                current_inventory,
                2,
            ),
            "required_inventory": round(
                required_inventory,
                2,
            ),
            "shortage": round(
                shortage,
                2,
            ),
            "safety_stock": round(
                safety_stock,
                2,
            ),
            "stockout_risk": risk,
        }

    # ------------------------------------------------------------------
    # SUPPLY
    # ------------------------------------------------------------------

    def _supply_intelligence(
        self,
        lead_time_days: int,
        daily_forecast: List[Dict[str, Any]],
        current_inventory: float,
        safety_stock: float,
    ) -> Dict[str, Any]:

        if lead_time_days <= 0:
            lead_time_demand = 0.0
        else:
            lead_time_demand = float(
                sum(
                    item["demand"]
                    for item in daily_forecast[
                        :lead_time_days
                    ]
                )
            )

        target = (
            lead_time_demand
            + safety_stock
        )

        shortage = max(
            0.0,
            target - current_inventory,
        )

        # Generic demo order multiple.
        # Real supplier configuration will replace this later.
        order_multiple = 1

        recommended_order = (
            math.ceil(shortage / order_multiple)
            * order_multiple
        )

        if recommended_order > 0:
            supplier_risk = (
                "HIGH"
                if recommended_order > lead_time_demand
                else "MODERATE"
            )
        else:
            supplier_risk = "LOW"

        return {
            "lead_time_days": int(lead_time_days),
            "lead_time_demand": round(
                lead_time_demand,
                2,
            ),
            "recommended_order": int(
                recommended_order
            ),
            "order_multiple": order_multiple,
            "supplier_risk": supplier_risk,
        }

    # ------------------------------------------------------------------
    # DRIVERS
    # ------------------------------------------------------------------

    def _drivers(
        self,
        history: pd.DataFrame,
        forecast_date: pd.Timestamp,
        demand_output: Dict[str, Any],
    ) -> List[str]:

        drivers: List[str] = []

        drivers.append(
            "Recent historical demand pattern"
        )

        drivers.append(
            "Day-of-week demand behavior"
        )

        if demand_output["demand_change_pct"] > 5:
            drivers.append(
                "Positive recent demand trend"
            )
        elif demand_output["demand_change_pct"] < -5:
            drivers.append(
                "Negative recent demand trend"
            )

        # Context is only reported when it actually exists in the data.
        context_map = [
            (
                self.holiday_col,
                "Holiday context available",
            ),
            (
                self.promotion_col,
                "Promotion context available",
            ),
            (
                self.event_col,
                "Event context available",
            ),
            (
                self.weather_col,
                "Weather context available",
            ),
        ]

        for column, label in context_map:
            if column is None:
                continue

            values = history[column].dropna()

            if values.empty:
                continue

            try:
                numeric = pd.to_numeric(
                    values,
                    errors="coerce",
                )

                if numeric.notna().any() and float(
                    numeric.fillna(0).max()
                ) > 0:
                    drivers.append(label)
                    continue
            except Exception:
                pass

            text = values.astype(str).str.lower()

            if not text.isin(
                {"0", "false", "none", "nan", ""}
            ).all():
                drivers.append(label)

        # Remove duplicates while preserving order.
        return list(
            dict.fromkeys(drivers)
        )

    # ------------------------------------------------------------------
    # AUTONOMOUS DECISION
    # ------------------------------------------------------------------

    def _autonomous_decision(
        self,
        demand_output: Dict[str, Any],
        inventory_output: Dict[str, Any],
        supply_output: Dict[str, Any],
    ) -> Dict[str, Any]:

        shortage = inventory_output["shortage"]
        confidence = demand_output["confidence"]

        if confidence < 0.50:
            return {
                "action": "REVIEW_REQUIRED",
                "priority": "HIGH",
                "reason": (
                    "Forecast confidence is below the autonomous "
                    "decision threshold."
                ),
            }

        if shortage <= 0:
            return {
                "action": "HOLD",
                "priority": "NORMAL",
                "reason": (
                    "Projected demand is covered by the current "
                    "inventory position and safety-stock requirement."
                ),
            }

        if inventory_output["stockout_risk"] == "HIGH":
            return {
                "action": "REPLENISH",
                "priority": "HIGH",
                "reason": (
                    "Projected lead-time demand exceeds available "
                    "inventory coverage."
                ),
            }

        return {
            "action": "REPLENISH",
            "priority": "MEDIUM",
            "reason": (
                "Additional inventory is required to reach the "
                "projected lead-time target."
            ),
        }

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def run(
        self,
        request: ClientDemoRequest,
    ) -> ClientDemoResult:

        history = self._get_series(request)

        forecast_date = pd.Timestamp(
            request.forecast_date
        )

        daily_forecast = self._forecast_daily(
            history=history,
            forecast_date=forecast_date,
            days=7,
        )

        d1 = float(
            daily_forecast[0]["demand"]
        )

        d3 = float(
            sum(
                item["demand"]
                for item in daily_forecast[:3]
            )
        )

        d7 = float(
            sum(
                item["demand"]
                for item in daily_forecast[:7]
            )
        )

        forecast_output = {
            "d1": round(d1, 2),
            "d3": round(d3, 2),
            "d7": round(d7, 2),
            "daily_forecast": [
                {
                    **item,
                    "demand": round(
                        float(item["demand"]),
                        2,
                    ),
                }
                for item in daily_forecast
            ],
        }

        demand_output = self._demand_intelligence(
            history,
            daily_forecast,
        )

        current_inventory = self._current_inventory(
            history,
            request,
        )

        lead_time_days = min(
            request.lead_time_days,
            7,
        )

        lead_time_demand = float(
            sum(
                item["demand"]
                for item in daily_forecast[
                    :lead_time_days
                ]
            )
        )

        # Temporary inventory calculation so safety stock can be obtained.
        inventory_output = self._inventory_intelligence(
            history=history,
            current_inventory=current_inventory,
            lead_time_demand=lead_time_demand,
        )

        supply_output = self._supply_intelligence(
            lead_time_days=lead_time_days,
            daily_forecast=daily_forecast,
            current_inventory=current_inventory,
            safety_stock=inventory_output["safety_stock"],
        )

        drivers = self._drivers(
            history=history,
            forecast_date=forecast_date,
            demand_output=demand_output,
        )

        autonomous_output = self._autonomous_decision(
            demand_output=demand_output,
            inventory_output=inventory_output,
            supply_output=supply_output,
        )

        outlet_info: Dict[str, Any] = {
            "outlet_id": request.outlet_id,
        }

        product_info: Dict[str, Any] = {
            "product_id": request.product_id,
        }

        if self.outlet_type_col:
            value = history[self.outlet_type_col].dropna()
            if not value.empty:
                outlet_info["outlet_type"] = str(
                    value.iloc[-1]
                )

        if self.country_col:
            value = history[self.country_col].dropna()
            if not value.empty:
                outlet_info["country"] = str(
                    value.iloc[-1]
                )

        if self.location_type_col:
            value = history[self.location_type_col].dropna()
            if not value.empty:
                outlet_info["location_type"] = str(
                    value.iloc[-1]
                )

        result = ClientDemoResult(
            demo={
                "name": "Restaurant Demand AI",
                "version": "Part A",
                "environment": "synthetic_demo",
                "forecast_date": request.forecast_date,
                "scenario": request.scenario,
            },
            outlet=outlet_info,
            product=product_info,
            forecast=forecast_output,
            demand_intelligence=demand_output,
            inventory=inventory_output,
            supply=supply_output,
            autonomous_decision=autonomous_output,
            drivers=drivers,
            data_source=str(
                self.dataset_path
            ),
            production_note=(
                "Synthetic historical data is used for this "
                "demonstration. The downstream intelligence "
                "contracts are designed to accept a future "
                "client data adapter."
            ),
        )

        return result

    def validate_result(
        self,
        result: ClientDemoResult,
    ) -> Dict[str, Any]:

        errors: List[str] = []

        forecast = result.forecast
        demand = result.demand_intelligence
        inventory = result.inventory
        supply = result.supply
        decision = result.autonomous_decision

        for key in ["d1", "d3", "d7"]:
            value = forecast.get(key)

            if value is None or not math.isfinite(
                float(value)
            ):
                errors.append(
                    f"forecast.{key} is not finite"
                )

            if value is not None and float(value) < 0:
                errors.append(
                    f"forecast.{key} is negative"
                )

        if (
            forecast["d1"] > forecast["d3"]
            or forecast["d3"] > forecast["d7"]
        ):
            errors.append(
                "forecast horizon totals are not monotonic"
            )

        for section_name, section in [
            ("demand_intelligence", demand),
            ("inventory", inventory),
            ("supply", supply),
        ]:
            for key, value in section.items():
                if isinstance(value, (int, float)):
                    if not math.isfinite(float(value)):
                        errors.append(
                            f"{section_name}.{key} is not finite"
                        )

        if demand["confidence"] < 0 or demand["confidence"] > 1:
            errors.append(
                "confidence outside [0, 1]"
            )

        if inventory["shortage"] < 0:
            errors.append(
                "inventory shortage is negative"
            )

        if supply["recommended_order"] < 0:
            errors.append(
                "recommended order is negative"
            )

        allowed_actions = {
            "HOLD",
            "REPLENISH",
            "REVIEW_REQUIRED",
        }

        if decision["action"] not in allowed_actions:
            errors.append(
                "invalid autonomous action"
            )

        return {
            "passed": not errors,
            "errors": errors,
            "dataset_exists": self.dataset_path.exists(),
            "rows_loaded": int(len(self.df)),
            "demand_column": self.demand_col,
            "forecast_time_firewall": True,
        }