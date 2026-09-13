from __future__ import annotations

import math
import uuid
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from app.features.feature_pipeline import build_features
from app.models.xgboost_model import XGBoostDemandModel


PRODUCTION_FEATURE_COLUMNS = [
    "year", "month", "day", "day_of_week", "week_of_year", "day_of_year",
    "quarter", "is_weekend", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_28",
    "rolling_mean_3", "rolling_std_3", "rolling_mean_7", "rolling_std_7",
    "rolling_mean_14", "rolling_std_14", "rolling_mean_28", "rolling_std_28",
    "trend_1d", "trend_7d", "trend_14d", "trend_ratio_7d", "trend_ratio_14d",
    "trend_acceleration", "volatility_3", "coefficient_variation_3",
    "volatility_7", "coefficient_variation_7", "volatility_14",
    "coefficient_variation_14", "volatility_28", "coefficient_variation_28",
    "order_velocity_3d", "order_velocity_7d", "order_velocity_14d",
    "order_velocity_change", "dow_sin_2", "dow_cos_2", "month_sin_2",
    "month_cos_2", "quarter_sin", "quarter_cos", "weekend_month_interaction",
]

MIN_HISTORY_DAYS = 35


class ClientIntelligenceService:
    """Synchronous client JSON inference/training proof.

    This endpoint trains a temporary Part-9 XGBoost model from the supplied
    history. It is intentionally a bridge for client integration testing.
    Production should load a governed champion model rather than retraining
    on every request.
    """

    VERSION = "Client Intelligence V1"

    def _frame(self, payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        sales = pd.DataFrame(payload["sales"])
        inventory = pd.DataFrame(payload.get("inventory", []))
        outlets = pd.DataFrame(payload.get("outlets", []))
        products = pd.DataFrame(payload.get("products", []))

        sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
        sales["quantity_sold"] = pd.to_numeric(sales["quantity_sold"], errors="coerce")
        sales = sales.dropna(subset=["date", "outlet_id", "product_id", "quantity_sold"]).copy()
        sales = sales[sales["quantity_sold"] >= 0].copy()
        sales = (
            sales.groupby(["date", "outlet_id", "product_id"], as_index=False)["quantity_sold"]
            .sum()
            .sort_values(["outlet_id", "product_id", "date"])
            .reset_index(drop=True)
        )

        if not sales.empty:
            for column in ["current_inventory", "safety_stock", "minimum_order_quantity", "order_multiple"]:
                if column in inventory.columns:
                    inventory[column] = pd.to_numeric(inventory[column], errors="coerce")
            if "lead_time_days" in inventory.columns:
                inventory["lead_time_days"] = pd.to_numeric(inventory["lead_time_days"], errors="coerce")

        return sales, inventory, outlets, products

    def _validate_history(self, sales: pd.DataFrame) -> dict[str, Any]:
        counts = sales.groupby(["outlet_id", "product_id"])["date"].nunique()
        insufficient = int((counts < MIN_HISTORY_DAYS).sum())
        if insufficient:
            examples = [
                {"outlet_id": k[0], "product_id": k[1], "days": int(v)}
                for k, v in counts[counts < MIN_HISTORY_DAYS].head(10).items()
            ]
            raise ValueError(
                f"Each outlet/product series requires at least {MIN_HISTORY_DAYS} unique history days; "
                f"{insufficient} series are insufficient. Examples: {examples}"
            )
        return {
            "series": int(len(counts)),
            "minimum_history_days": int(counts.min()),
            "maximum_history_days": int(counts.max()),
        }

    def _prepare_training(self, sales: pd.DataFrame) -> pd.DataFrame:
        features = build_features(
            sales.copy(),
            date_column="date",
            target_column="quantity_sold",
            group_columns=["outlet_id", "product_id"],
        )
        missing = [c for c in PRODUCTION_FEATURE_COLUMNS if c not in features.columns]
        if missing:
            raise ValueError(f"Part-9 feature contract missing columns: {missing}")
        return features

    def _train(self, features: pd.DataFrame) -> XGBoostDemandModel:
        model = XGBoostDemandModel()
        X = features[PRODUCTION_FEATURE_COLUMNS].copy()
        y = pd.to_numeric(features["quantity_sold"], errors="coerce")
        valid = y.notna() & np.isfinite(y.to_numpy())
        model.fit(X.loc[valid], y.loc[valid])
        if len(model.feature_columns) != 51:
            raise RuntimeError(f"Production feature contract violation: {len(model.feature_columns)} features")
        return model

    def _predict_future(
        self,
        model: XGBoostDemandModel,
        history: pd.DataFrame,
        horizon: int = 7,
    ) -> pd.DataFrame:
        work = history.copy()
        last_date = work["date"].max()
        series = work[["outlet_id", "product_id"]].drop_duplicates().reset_index(drop=True)
        predictions: list[pd.DataFrame] = []

        for step in range(1, horizon + 1):
            target_date = last_date + timedelta(days=step)
            future = series.copy()
            future["date"] = target_date
            future["quantity_sold"] = np.nan

            combined = pd.concat([work, future], ignore_index=True, sort=False)
            combined = combined.sort_values(["outlet_id", "product_id", "date"]).reset_index(drop=True)
            features = build_features(
                combined,
                date_column="date",
                target_column="quantity_sold",
                group_columns=["outlet_id", "product_id"],
            )
            mask = features["date"].eq(target_date)
            current = features.loc[mask, ["outlet_id", "product_id", "date"] + PRODUCTION_FEATURE_COLUMNS].copy()
            current["prediction"] = model.predict(current[PRODUCTION_FEATURE_COLUMNS])
            current["prediction"] = np.maximum(current["prediction"].astype(float), 0.0)

            predictions.append(current[["outlet_id", "product_id", "date", "prediction"]])

            future = future.drop(columns=["quantity_sold"])
            future = future.merge(
                current[["outlet_id", "product_id", "prediction"]],
                on=["outlet_id", "product_id"],
                how="left",
            ).rename(columns={"prediction": "quantity_sold"})
            work = pd.concat([work, future], ignore_index=True, sort=False)

        return pd.concat(predictions, ignore_index=True)

    @staticmethod
    def _risk(change_pct: float) -> str:
        if change_pct >= 0.20:
            return "HIGH"
        if change_pct >= 0.08:
            return "MODERATE"
        return "LOW"

    @staticmethod
    def _round(value: float) -> float:
        return round(float(value), 2)

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        sales, inventory, outlets, products = self._frame(request)
        if sales.empty:
            raise ValueError("No valid sales rows remain after validation")

        history_quality = self._validate_history(sales)
        report_date = pd.Timestamp(request.get("as_of_date")) if request.get("as_of_date") else sales["date"].max() + pd.Timedelta(days=1)
        report_date = pd.Timestamp(report_date).date()

        features = self._prepare_training(sales)
        model = self._train(features)
        future = self._predict_future(model, sales, horizon=7)

        recent = (
            sales.sort_values("date")
            .groupby(["outlet_id", "product_id"], as_index=False)
            .tail(28)
        )
        baseline = recent.groupby(["outlet_id", "product_id"])["quantity_sold"].mean().rename("baseline").reset_index()
        variability = recent.groupby(["outlet_id", "product_id"])["quantity_sold"].std().fillna(0).rename("std").reset_index()

        future["step"] = (future["date"] - pd.Timestamp(sales["date"].max())).dt.days
        fwide = future.pivot_table(index=["outlet_id", "product_id"], columns="step", values="prediction", aggfunc="first").reset_index()
        fwide.columns = [str(c) if isinstance(c, int) else c for c in fwide.columns]
        result = baseline.merge(variability, on=["outlet_id", "product_id"], how="left").merge(fwide, on=["outlet_id", "product_id"], how="left")

        inv = inventory.copy()
        if not inv.empty:
            inv = inv.sort_values(["outlet_id", "product_id"]).drop_duplicates(["outlet_id", "product_id"], keep="last")
        outlet_map = outlets.set_index("outlet_id").to_dict("index") if not outlets.empty else {}
        product_map = products.set_index("product_id").to_dict("index") if not products.empty else {}
        inventory_map = inv.set_index(["outlet_id", "product_id"]).to_dict("index") if not inv.empty else {}

        actions: list[dict[str, Any]] = []
        for row in result.to_dict("records"):
            key = (row["outlet_id"], row["product_id"])
            meta_i = inventory_map.get(key, {})
            meta_o = outlet_map.get(row["outlet_id"], {})
            meta_p = product_map.get(row["product_id"], {})

            d1 = float(row.get("1", 0.0))
            d3 = sum(float(row.get(str(i), 0.0)) for i in range(1, 4))
            d7 = sum(float(row.get(str(i), 0.0)) for i in range(1, 8))
            baseline_value = max(float(row["baseline"]), 1e-6)
            change_pct = (d1 - baseline_value) / baseline_value
            demand_risk = self._risk(change_pct)

            lead_time_raw = meta_i.get("lead_time_days", 1)

            if lead_time_raw is None:
                lead_time = 1
            else:
                try:
                    lead_time_float = float(lead_time_raw)
                    lead_time = int(lead_time_float) if np.isfinite(lead_time_float) else 1
                except (TypeError, ValueError):
                    lead_time = 1

            lead_time = max(1, min(lead_time, 7))

            lead_time_demand = sum(
                float(row.get(str(i), 0.0))
                for i in range(1, lead_time + 1)
            )
            safety = meta_i.get("safety_stock")
            if safety is None or not np.isfinite(float(safety)):
                safety = 1.65 * float(row.get("std", 0.0))
            safety = max(0.0, float(safety))
            required = lead_time_demand + safety
            current = meta_i.get("current_inventory")
            inventory_known = current is not None and np.isfinite(float(current))
            current_value = max(0.0, float(current)) if inventory_known else 0.0
            shortage = max(0.0, required - current_value) if inventory_known else None

            if inventory_known:
                order = max(0.0, required - current_value)
                moq = meta_i.get("minimum_order_quantity")
                multiple = meta_i.get("order_multiple")

                try:
                    moq = float(moq) if moq is not None and np.isfinite(float(moq)) else None
                except (TypeError, ValueError):
                    moq = None

                try:
                    multiple = float(multiple) if multiple is not None and np.isfinite(float(multiple)) else None
                except (TypeError, ValueError):
                    multiple = None

                if moq is not None and moq > 0 and order > 0:
                    order = max(order, moq)

                if multiple is not None and multiple > 0 and order > 0:
                    order = math.ceil(order / multiple) * multiple
                action = "REPLENISH" if order > 0 else "NO_ACTION"
                stockout_risk = "HIGH" if current_value < lead_time_demand else "LOW"
                reason = (
                    "Projected lead-time demand plus safety stock exceeds available inventory."
                    if order > 0 else
                    "Available inventory covers projected lead-time demand and safety stock."
                )
            else:
                order = None
                stockout_risk = "UNKNOWN"
                action = "INVENTORY_DATA_REQUIRED"
                reason = "Inventory position was not supplied; forecast is available but replenishment quantity cannot be safely calculated."

            priority = "HIGH" if action == "REPLENISH" and (stockout_risk == "HIGH" or demand_risk == "HIGH") else ("MODERATE" if action == "REPLENISH" else "LOW")
            drivers = ["recent demand pattern", "day-of-week demand behavior"]
            if change_pct > 0.08:
                drivers.insert(0, "recent demand acceleration")

            actions.append({
                "outlet": {
                    "outlet_id": row["outlet_id"],
                    "outlet_name": meta_o.get("outlet_name") or f"Outlet {row['outlet_id']}",
                    "outlet_type": meta_o.get("outlet_type"),
                    "country": meta_o.get("country"),
                    "region": meta_o.get("region"),
                    "city": meta_o.get("city"),
                },
                "product": {
                    "product_id": row["product_id"],
                    "product_name": meta_p.get("product_name") or f"Product {row['product_id']}",
                    "category": meta_p.get("category"),
                    "unit": meta_p.get("unit"),
                },
                "date": report_date.isoformat(),
                "forecast": {
                    "d1": self._round(d1),
                    "d3": self._round(d3),
                    "d7": self._round(d7),
                    "unit": meta_p.get("unit"),
                },
                "demand_risk": demand_risk,
                "demand_change_pct": self._round(change_pct * 100.0),
                "demand_spike": "WATCH" if change_pct >= 0.20 else "NONE",
                "model_confidence": None,
                "confidence_status": "NOT_CALIBRATED",
                "inventory": {
                    "current_inventory": self._round(current_value) if inventory_known else None,
                    "required_inventory": self._round(required) if inventory_known else None,
                    "shortage": self._round(shortage) if shortage is not None else None,
                    "safety_stock": self._round(safety),
                },
                "supply": {
                    "lead_time_days": lead_time,
                    "lead_time_demand": self._round(lead_time_demand),
                    "recommended_order": self._round(order) if order is not None else None,
                    "order_unit": meta_p.get("unit"),
                    "supplier_risk": meta_i.get("supplier_risk") or "UNKNOWN",
                },
                "stockout_risk": stockout_risk,
                "priority": priority,
                "recommendation": {"action": action, "reason": reason},
                "key_drivers": drivers,
            })

        actions.sort(key=lambda x: {"HIGH": 0, "MODERATE": 1, "LOW": 2}.get(x["priority"], 3))
        if not request.get("include_normal", False):
            actions = [a for a in actions if a["priority"] != "LOW"]
        actions = actions[: int(request.get("top_n", 50))]

        high = sum(1 for a in actions if a["priority"] == "HIGH")
        moderate = sum(1 for a in actions if a["priority"] == "MODERATE")
        replenishment = sum(1 for a in actions if a["recommendation"]["action"] == "REPLENISH")

        return {
            "service": "Restaurant Demand AI",
            "version": self.VERSION,
            "request_id": f"req_{uuid.uuid4().hex[:12]}",
            "client_id": request["client_id"],
            "report_date": report_date,
            "source_status": "CLIENT_PROVIDED_DATA",
            "model": {
                "model_type": "XGBoostDemandModel",
                "feature_contract": "Part 9 / 51 features",
                "training_mode": "REQUEST_TIME_TEMPORARY_MODEL",
                "production_champion": False,
                "confidence_calibrated": False,
            },
            "summary": {
                "outlets_analyzed": int(sales["outlet_id"].nunique()),
                "products_analyzed": int(sales["product_id"].nunique()),
                "outlet_product_series_analyzed": int(sales.groupby(["outlet_id", "product_id"]).ngroups),
                "actions_returned": len(actions),
                "high_priority": high,
                "moderate_priority": moderate,
                "replenishment_actions": replenishment,
            },
            "actions": actions,
            "data_quality": {
                "sales_rows": int(len(sales)),
                "history": history_quality,
                "inventory_rows": int(len(inventory)),
                "outlet_metadata_rows": int(len(outlets)),
                "product_metadata_rows": int(len(products)),
                "synthetic_data_used": False,
            },
        }
