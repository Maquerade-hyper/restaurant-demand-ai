from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.intelligence.daily.identity import (
    build_outlet_identity,
    build_product_identity,
    load_outlet_catalog,
    load_product_catalog,
)
from app.intelligence.daily.ranking import (
    rank_actions,
    summarize_priorities,
)


ROOT = Path(__file__).resolve().parents[3]

DATASET = (
    ROOT
    / "data"
    / "interim"
    / "demand_censoring_intelligence.csv"
)


class DailyIntelligenceEngine:
    """
    Batch daily commercial intelligence engine.

    Converts historical outlet/product information into a
    machine-readable daily operational report.

    This layer does not replace the established ML/intelligence
    layers. It is the commercial response/orchestration layer.
    """

    MIN_HISTORY = 28
    LEAD_TIME_DAYS = 3
    ORDER_MULTIPLE = 1.0

    def __init__(
        self,
        dataset_path: str | Path = DATASET,
    ) -> None:
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Daily intelligence dataset not found: "
                f"{self.dataset_path}"
            )

        self.df = pd.read_csv(self.dataset_path)

        required = {
            "date",
            "outlet_id",
            "product_id",
        }

        missing = required - set(self.df.columns)

        if missing:
            raise ValueError(
                "Daily intelligence dataset missing columns: "
                f"{sorted(missing)}"
            )

        self.df["date"] = pd.to_datetime(
            self.df["date"],
            errors="coerce",
        )

        self.df = self.df.dropna(
            subset=[
                "date",
                "outlet_id",
                "product_id",
            ]
        ).copy()

        self.df["outlet_id"] = (
            self.df["outlet_id"].astype(str)
        )

        self.df["product_id"] = (
            self.df["product_id"].astype(str)
        )

        self.demand_column = self._find_demand_column()
        self.inventory_column = self._find_inventory_column()

        self.df[self.demand_column] = pd.to_numeric(
            self.df[self.demand_column],
            errors="coerce",
        )

        self.df = self.df.dropna(
            subset=[self.demand_column]
        ).copy()

        self.df = self.df.sort_values(
            [
                "outlet_id",
                "product_id",
                "date",
            ]
        ).reset_index(drop=True)

        self.outlet_catalog = load_outlet_catalog()
        self.product_catalog = load_product_catalog()

        self.latest_date = self.df["date"].max()

        # Pre-build series once.
        # This prevents thousands of repeated full-DataFrame scans.
        self._series_groups = {
            (
                str(outlet_id),
                str(product_id),
            ): group.reset_index(drop=True)
            for (
                outlet_id,
                product_id,
            ), group in self.df.groupby(
                ["outlet_id", "product_id"],
                sort=True,
            )
        }

        # Cache outlet/product identities.
        self._outlet_identity_cache = {}
        self._product_identity_cache = {}

    # ------------------------------------------------------------------
    # COLUMN DISCOVERY
    # ------------------------------------------------------------------

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
            "No demand column found. Expected one of: "
            + ", ".join(candidates)
        )

    def _find_inventory_column(self) -> str | None:
        candidates = [
            "closing_stock",
            "current_inventory",
            "inventory",
            "stock",
            "available_inventory",
        ]

        for column in candidates:
            if column in self.df.columns:
                return column

        return None

    # ------------------------------------------------------------------
    # IDENTITY
    # ------------------------------------------------------------------

    def _outlet_identity(
        self,
        outlet_id: str,
    ) -> dict:
        outlet_id = str(outlet_id)

        if outlet_id not in self._outlet_identity_cache:
            self._outlet_identity_cache[outlet_id] = (
                build_outlet_identity(
                    outlet_id,
                    self.outlet_catalog,
                )
            )

        return dict(
            self._outlet_identity_cache[outlet_id]
        )

    def _product_identity(
        self,
        product_id: str,
    ) -> dict:
        product_id = str(product_id)

        if product_id not in self._product_identity_cache:
            self._product_identity_cache[product_id] = (
                build_product_identity(
                    product_id,
                    self.product_catalog,
                )
            )

        return dict(
            self._product_identity_cache[product_id]
        )

    # ------------------------------------------------------------------
    # HISTORY
    # ------------------------------------------------------------------

    def _series_history(
        self,
        outlet_id: str,
        product_id: str,
        forecast_date: pd.Timestamp,
    ) -> pd.DataFrame:
        key = (
            str(outlet_id),
            str(product_id),
        )

        series = self._series_groups.get(key)

        if series is None:
            return pd.DataFrame()

        # Strict forecast-time firewall.
        history = series[
            series["date"] < forecast_date
        ].copy()

        return history

    # ------------------------------------------------------------------
    # FORECAST
    # ------------------------------------------------------------------

    def _forecast(
        self,
        history: pd.DataFrame,
        forecast_date: pd.Timestamp,
    ) -> list[dict[str, Any]]:
        demand = (
            history[self.demand_column]
            .astype(float)
        )

        recent_28 = demand.tail(28)

        recent_mean = float(
            recent_28.mean()
        )

        dow = forecast_date.dayofweek

        dow_mask = (
            history["date"].dt.dayofweek
            == dow
        )

        dow_history = (
            history.loc[
                dow_mask,
                self.demand_column,
            ]
            .astype(float)
        )

        if len(dow_history) >= 4:
            dow_mean = float(
                dow_history.tail(12).mean()
            )
        else:
            dow_mean = recent_mean

        if len(demand) >= 28:
            older = float(
                demand.tail(28)
                .head(14)
                .mean()
            )

            newer = float(
                demand.tail(14).mean()
            )

            if older > 0:
                trend_ratio = newer / older
            else:
                trend_ratio = 1.0

            trend_ratio = float(
                np.clip(
                    trend_ratio,
                    0.85,
                    1.15,
                )
            )
        else:
            trend_ratio = 1.0

        base = (
            0.55 * recent_mean
            + 0.45 * dow_mean
        )

        base *= trend_ratio

        forecasts = []

        for step in range(7):
            target_date = (
                forecast_date
                + pd.Timedelta(days=step)
            )

            target_dow = (
                target_date.dayofweek
            )

            same_dow = (
                history.loc[
                    history["date"].dt.dayofweek
                    == target_dow,
                    self.demand_column,
                ]
                .astype(float)
            )

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

            forecasts.append(
                {
                    "date": target_date.strftime(
                        "%Y-%m-%d"
                    ),
                    "demand": max(
                        0.0,
                        float(value),
                    ),
                }
            )

        return forecasts

    # ------------------------------------------------------------------
    # INVENTORY
    # ------------------------------------------------------------------

    def _current_inventory(
        self,
        history: pd.DataFrame,
    ) -> float:
        if self.inventory_column is None:
            return 0.0

        values = pd.to_numeric(
            history[self.inventory_column],
            errors="coerce",
        ).dropna()

        if values.empty:
            return 0.0

        return max(
            0.0,
            float(values.iloc[-1]),
        )

    # ------------------------------------------------------------------
    # SINGLE ACTION
    # ------------------------------------------------------------------

    def _build_action(
        self,
        outlet_id: str,
        product_id: str,
        forecast_date: pd.Timestamp,
    ) -> dict | None:
        history = self._series_history(
            outlet_id,
            product_id,
            forecast_date,
        )

        if len(history) < self.MIN_HISTORY:
            return None

        daily = self._forecast(
            history,
            forecast_date,
        )

        d1 = float(
            daily[0]["demand"]
        )

        d3 = float(
            sum(
                item["demand"]
                for item in daily[:3]
            )
        )

        d7 = float(
            sum(
                item["demand"]
                for item in daily[:7]
            )
        )

        recent = (
            history[self.demand_column]
            .astype(float)
            .tail(28)
        )

        baseline = float(
            recent.mean()
        )

        std = float(
            recent.std(ddof=0)
        )

        forecast_daily_mean = (
            d7 / 7.0
        )

        if baseline > 0:
            change_pct = (
                (
                    forecast_daily_mean
                    - baseline
                )
                / baseline
                * 100.0
            )
        else:
            change_pct = 0.0

        if std > 0:
            z_score = (
                forecast_daily_mean
                - baseline
            ) / std
        else:
            z_score = 0.0

        spike = bool(
            change_pct >= 20.0
            and z_score >= 1.5
        )

        if not spike:
            spike_severity = "NONE"
        elif (
            change_pct >= 50.0
            or z_score >= 3.0
        ):
            spike_severity = "EXTREME"
        elif (
            change_pct >= 35.0
            or z_score >= 2.0
        ):
            spike_severity = "MAJOR"
        else:
            spike_severity = "MODERATE"

        if (
            change_pct >= 30.0
            or z_score >= 2.0
        ):
            demand_risk = "HIGH"
        elif (
            change_pct >= 10.0
            or z_score >= 1.0
        ):
            demand_risk = "MODERATE"
        else:
            demand_risk = "LOW"

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
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        )

        current_inventory = (
            self._current_inventory(history)
        )

        lead_time = min(
            self.LEAD_TIME_DAYS,
            len(daily),
        )

        lead_time_demand = float(
            sum(
                item["demand"]
                for item in daily[:lead_time]
            )
        )

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
            required_inventory
            - current_inventory,
        )

        if current_inventory < lead_time_demand:
            stockout_risk = "HIGH"
        elif current_inventory < required_inventory:
            stockout_risk = "MODERATE"
        else:
            stockout_risk = "LOW"

        recommended_order = (
            math.ceil(
                shortage
                / self.ORDER_MULTIPLE
            )
            * self.ORDER_MULTIPLE
        )

        if recommended_order <= 0:
            supplier_risk = "LOW"
        elif recommended_order > lead_time_demand:
            supplier_risk = "HIGH"
        else:
            supplier_risk = "MODERATE"

        if confidence < 0.50:
            action = "REVIEW_REQUIRED"
            priority = "HIGH"
            reason = (
                "Forecast confidence is below the "
                "autonomous decision threshold."
            )
        elif shortage <= 0:
            action = "HOLD"
            priority = "NORMAL"
            reason = (
                "Projected demand is covered by the "
                "current inventory position and "
                "safety-stock requirement."
            )
        elif stockout_risk == "HIGH":
            action = "REPLENISH"
            priority = "HIGH"
            reason = (
                "Projected lead-time demand exceeds "
                "available inventory coverage."
            )
        else:
            action = "REPLENISH"
            priority = "MODERATE"
            reason = (
                "Additional inventory is required to "
                "reach the projected lead-time target."
            )

        drivers = [
            "Recent historical demand pattern",
            "Day-of-week demand behavior",
        ]

        if change_pct > 5:
            drivers.append(
                "Positive recent demand trend"
            )
        elif change_pct < -5:
            drivers.append(
                "Negative recent demand trend"
            )

        context_columns = [
            (
                "holiday_active",
                "Holiday context available",
            ),
            (
                "promotion_active",
                "Promotion context available",
            ),
            (
                "event_active",
                "Event context available",
            ),
            (
                "is_rainy",
                "Weather context available",
            ),
        ]

        for column, label in context_columns:
            if column not in history.columns:
                continue

            values = pd.to_numeric(
                history[column],
                errors="coerce",
            )

            if (
                values.notna().any()
                and float(
                    values.fillna(0).max()
                ) > 0
            ):
                drivers.append(label)

        outlet = self._outlet_identity(
            outlet_id
        )

        product = self._product_identity(
            product_id
        )

        return {
            "outlet": outlet,
            "product": product,
            "date": forecast_date.strftime(
                "%Y-%m-%d"
            ),
            "forecast": {
                "d1": round(d1, 2),
                "d3": round(d3, 2),
                "d7": round(d7, 2),
            },
            "demand_risk": demand_risk,
            "demand_spike": spike_severity,
            "model_confidence": round(
                confidence,
                4,
            ),
            "inventory": {
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
            },
            "supply": {
                "lead_time_days": int(
                    lead_time
                ),
                "lead_time_demand": round(
                    lead_time_demand,
                    2,
                ),
                "recommended_order": round(
                    recommended_order,
                    2,
                ),
                "order_unit": str(
                    product["unit"]
                ),
                "order_multiple": float(
                    self.ORDER_MULTIPLE
                ),
                "supplier_risk": supplier_risk,
            },
            "stockout_risk": stockout_risk,
            "priority": priority,
            "recommendation": {
                "action": action,
                "priority": priority,
                "reason": reason,
            },
            "key_drivers": list(
                dict.fromkeys(drivers)
            ),
        }

    # ------------------------------------------------------------------
    # DAILY REPORT
    # ------------------------------------------------------------------

    def generate(
        self,
        report_date: str | None = None,
        include_normal: bool = False,
    ) -> dict:
        if report_date is None:
            forecast_date = (
                self.latest_date
                + pd.Timedelta(days=1)
            )
        else:
            forecast_date = pd.Timestamp(
                report_date
            )

        if forecast_date <= self.latest_date:
            raise ValueError(
                "report_date must be after the latest "
                f"historical date "
                f"{self.latest_date.date()}."
            )

        actions = []

        for (
            outlet_id,
            product_id,
        ) in self._series_groups.keys():
            action = self._build_action(
                outlet_id=outlet_id,
                product_id=product_id,
                forecast_date=forecast_date,
            )

            if action is None:
                continue

            if (
                not include_normal
                and action["priority"] == "NORMAL"
            ):
                continue

            actions.append(action)

        actions = rank_actions(actions)

        priorities = summarize_priorities(
            actions
        )

        replenish = sum(
            1
            for item in actions
            if item["recommendation"]["action"]
            == "REPLENISH"
        )

        review = sum(
            1
            for item in actions
            if item["recommendation"]["action"]
            == "REVIEW_REQUIRED"
        )

        hold = sum(
            1
            for item in actions
            if item["recommendation"]["action"]
            == "HOLD"
        )

        summary = {
            "outlets_analyzed": int(
                self.df["outlet_id"].nunique()
            ),
            "products_analyzed": int(
                self.df["product_id"].nunique()
            ),
            "outlet_product_series_analyzed": int(
                len(self._series_groups)
            ),
            "actions_required": int(
                len(actions)
            ),
            "critical": int(
                priorities["CRITICAL"]
            ),
            "high": int(
                priorities["HIGH"]
            ),
            "moderate": int(
                priorities["MODERATE"]
            ),
            "low": int(
                priorities["LOW"]
            ),
            "normal": int(
                priorities["NORMAL"]
            ),
            "replenish_actions": int(
                replenish
            ),
            "review_actions": int(
                review
            ),
            "hold_actions": int(
                hold
            ),
        }

        return {
            "service": "Restaurant Demand AI",
            "version": (
                "Daily Commercial Intelligence V1"
            ),
            "report_date": forecast_date.strftime(
                "%Y-%m-%d"
            ),
            "source_status": (
                "SYNTHETIC_DEMONSTRATION_DATA"
            ),
            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "summary": summary,
            "actions": actions,
            "validation": self.validate_report(
                actions,
                summary,
            ),
        }

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def validate_report(
        self,
        actions: list[dict],
        summary: dict,
    ) -> dict:
        errors = []

        if summary["outlets_analyzed"] <= 0:
            errors.append(
                "No outlets analyzed."
            )

        if summary["products_analyzed"] <= 0:
            errors.append(
                "No products analyzed."
            )

        if (
            summary["actions_required"]
            != len(actions)
        ):
            errors.append(
                "Action count mismatch."
            )

        for index, action in enumerate(
            actions
        ):
            forecast = action["forecast"]
            inventory = action["inventory"]
            supply = action["supply"]

            if (
                forecast["d1"] < 0
                or forecast["d3"] < 0
                or forecast["d7"] < 0
            ):
                errors.append(
                    f"Action {index}: negative forecast."
                )

            if not (
                forecast["d1"]
                <= forecast["d3"]
                <= forecast["d7"]
            ):
                errors.append(
                    f"Action {index}: invalid "
                    "forecast horizon ordering."
                )

            if inventory["shortage"] < 0:
                errors.append(
                    f"Action {index}: negative shortage."
                )

            if supply["recommended_order"] < 0:
                errors.append(
                    f"Action {index}: negative order."
                )

            if not (
                0
                <= action["model_confidence"]
                <= 1
            ):
                errors.append(
                    f"Action {index}: invalid confidence."
                )

            for value in [
                forecast["d1"],
                forecast["d3"],
                forecast["d7"],
                inventory["current_inventory"],
                inventory["required_inventory"],
                inventory["shortage"],
                supply["lead_time_demand"],
                supply["recommended_order"],
            ]:
                if not math.isfinite(
                    float(value)
                ):
                    errors.append(
                        f"Action {index}: "
                        "non-finite numeric value."
                    )

        return {
            "passed": bool(not errors),
            "errors": errors,
            "dataset_exists": bool(
                self.dataset_path.exists()
            ),
            "rows_loaded": int(
                len(self.df)
            ),
            "demand_column": str(
                self.demand_column
            ),
            "forecast_time_firewall": True,
            "identity_source": (
                "canonical_outlet_product_catalog"
            ),
        }