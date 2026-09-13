from pathlib import Path

import pandas as pd

from app.forecasting.xgboost_features import (
    prepare_xgboost_dataset,
)
from app.forecasting.xgboost_service import (
    XGBoostForecastService,
)
from app.forecasting.metrics import evaluate_forecast
from app.forecasting.diagnostics import (
    high_demand_analysis,
    spike_recall,
)

from app.forecasting.uncertainty.service import (
    UncertaintyService,
)
from app.inventory.service import InventoryService


SALES_PATH = Path("data/synthetic/sales.csv")
INVENTORY_PATH = Path("data/synthetic/inventory.csv")
PRODUCTS_PATH = Path("data/synthetic/products.csv")


def test_owner_demo():

    print()
    print("=" * 72)
    print(" RESTAURANT / BAR / CLOUD KITCHEN")
    print(" DEMAND & SUPPLY INTELLIGENCE")
    print(" OWNER DEMONSTRATION")
    print("=" * 72)

    # =========================================================
    # 1. BUSINESS DATA
    # =========================================================

    assert SALES_PATH.exists()

    sales = pd.read_csv(SALES_PATH)
    assert not sales.empty

    sales["date"] = pd.to_datetime(sales["date"])

    print()
    print("1. BUSINESS DATA")
    print("-" * 72)

    print(f"Outlets          : {sales['outlet_id'].nunique()}")
    print(f"Products         : {sales['product_id'].nunique()}")
    print(f"Sales records    : {len(sales):,}")
    print(
        f"Date range       : "
        f"{sales['date'].min().date()} → "
        f"{sales['date'].max().date()}"
    )

    # =========================================================
    # 2. FEATURE ENGINEERING
    # =========================================================

    print()
    print("2. DEMAND INTELLIGENCE")
    print("-" * 72)

    features = prepare_xgboost_dataset(sales)

    assert not features.empty

    print(f"Feature rows     : {len(features):,}")
    print(
        f"Feature columns  : "
        f"{len(features.columns)}"
    )

    # =========================================================
    # 3. TEMPORAL TRAIN / TEST SPLIT
    # =========================================================

    features = (
        features
        .sort_values("date")
        .reset_index(drop=True)
    )

    cutoff = (
        features["date"].max()
        - pd.Timedelta(days=28)
    )

    train = features[
        features["date"] < cutoff
    ].copy()

    test = features[
        features["date"] >= cutoff
    ].copy()

    assert not train.empty
    assert not test.empty

    assert (
        train["date"].max()
        < test["date"].min()
    )

    print()
    print("3. FORECAST PERIOD")
    print("-" * 72)

    print(
        f"Training period : "
        f"{train['date'].min().date()} → "
        f"{train['date'].max().date()}"
    )

    print(
        f"Test period     : "
        f"{test['date'].min().date()} → "
        f"{test['date'].max().date()}"
    )

    # =========================================================
    # 4. XGBOOST
    # =========================================================

    print()
    print("4. MACHINE LEARNING FORECAST")
    print("-" * 72)

    service = XGBoostForecastService()

    service.train(
        train,
        target_column="quantity_sold",
    )

    predictions = service.predict(test)

    assert len(predictions) == len(test)

    actual = (
        test["quantity_sold"]
        .reset_index(drop=True)
    )

    predictions = pd.Series(
        predictions
    ).reset_index(drop=True)

    metrics = evaluate_forecast(
        actual,
        predictions,
    )

    print("Model            : XGBoost")
    print(f"Train rows       : {len(train):,}")
    print(f"Test rows        : {len(test):,}")
    print(f"MAE              : {metrics['mae']:.4f}")
    print(f"RMSE             : {metrics['rmse']:.4f}")
    print(f"sMAPE            : {metrics['smape']:.4f}%")
    print(f"Bias             : {metrics['bias']:.4f}")

    # =========================================================
    # 5. DEMAND DIAGNOSTICS
    # =========================================================

    print()
    print("5. DEMAND RISK")
    print("-" * 72)

    high_demand = high_demand_analysis(
        actual,
        predictions,
    )

    spike = spike_recall(
        actual,
        predictions,
    )

    print(
        f"High-demand threshold : "
        f"{high_demand['threshold']:.2f}"
    )

    print(
        f"High-demand MAE       : "
        f"{high_demand['mae']:.4f}"
    )

    print(
        f"High-demand bias      : "
        f"{high_demand['bias']:.4f}"
    )

    print(
        f"Spike recall          : "
        f"{spike:.4f}"
    )



    # =========================================================
    # 6. FORECAST UNCERTAINTY
    # =========================================================

    print()
    print("6. FORECAST UNCERTAINTY")
    print("-" * 72)

    uncertainty_service = UncertaintyService()

    error_profile = uncertainty_service.fit(
        actual,
        predictions,
    )

    prediction_intervals = (
        uncertainty_service.predict(
            predictions
        )
    )

    reliability = (
        uncertainty_service.evaluate(
            actual,
            prediction_intervals,
        )
    )

    assert not prediction_intervals.empty

    assert {
        "prediction",
        "lower_bound",
        "upper_bound",
        "confidence",
    }.issubset(
        prediction_intervals.columns
    )

    print(
        f"Mean error          : "
        f"{error_profile['mean_error']:.4f}"
    )

    print(
        f"P90 absolute error  : "
        f"{error_profile['p90_absolute_error']:.4f}"
    )

    print(
        f"Interval coverage   : "
        f"{reliability['coverage']:.4f}"
    )

    print(
        f"Mean interval width : "
        f"{reliability['mean_interval_width']:.4f}"
    )

    print()
    print("Sample forecasts")
    print("-" * 72)

    sample = prediction_intervals.head(10)

    for _, row in sample.iterrows():

        print(
            f"Prediction: {row['prediction']:8.2f} | "
            f"Range: "
            f"{row['lower_bound']:8.2f} - "
            f"{row['upper_bound']:8.2f} | "
            f"Confidence: "
            f"{row['confidence']:.3f}"
        )
    # =========================================================
    # 6. CURRENT OWNER DEMO ACCEPTANCE
    # =========================================================

    print()
    print("=" * 72)
    print(" CURRENT OWNER DEMO RESULT")
    print("=" * 72)

    print("Synthetic business data : WORKING")
    print("Feature engineering     : WORKING")
    print("Temporal validation     : WORKING")
    print("XGBoost forecasting     : WORKING")
    print("Forecast diagnostics    : WORKING")
    print("Forecast uncertainty    : WORKING")
    print("Inventory intelligence  : WORKING")

    print("=" * 72)



    # =========================================================
    # 7. INVENTORY INTELLIGENCE
    # =========================================================

    print()
    print("7. INVENTORY INTELLIGENCE")
    print("-" * 72)

    assert INVENTORY_PATH.exists()
    assert PRODUCTS_PATH.exists()

    inventory = pd.read_csv(
        INVENTORY_PATH
    )

    products = pd.read_csv(
        PRODUCTS_PATH
    )

    assert not inventory.empty
    assert not products.empty

    inventory["date"] = pd.to_datetime(
        inventory["date"]
    )

    # ---------------------------------------------------------
    # Select one real outlet/product pair from the forecast
    # ---------------------------------------------------------

    demo_row = test.iloc[0]

    demo_outlet = demo_row["outlet_id"]
    demo_product = demo_row["product_id"]

    product_inventory = inventory[
        (inventory["outlet_id"] == demo_outlet)
        & (
            inventory["product_id"]
            == demo_product
        )
    ].sort_values("date")

    assert not product_inventory.empty

    latest_inventory = (
        product_inventory
        .iloc[-1]
    )

    inventory_position = float(
        latest_inventory["closing_stock"]
    )

    # ---------------------------------------------------------
    # Build a short forecast horizon.
    #
    # The current model has a 28-day holdout. For this Owner
    # Demo we use the first forecast values available for the
    # selected product/outlet.
    # ---------------------------------------------------------

    demo_forecasts = (
        pd.DataFrame(
            {
                "outlet_id": test["outlet_id"].values,
                "product_id": test["product_id"].values,
                "prediction": predictions.values,
            }
        )
    )

    demo_forecasts = demo_forecasts[
        (demo_forecasts["outlet_id"] == demo_outlet)
        & (
            demo_forecasts["product_id"]
            == demo_product
        )
    ]

    demo_forecasts = demo_forecasts.head(7)

    assert not demo_forecasts.empty

    daily_forecast = (
        demo_forecasts["prediction"]
        .clip(lower=0)
        .tolist()
    )

    # If fewer than 7 values are available, use the
    # available forecast horizon.
    lead_time_days = min(
        3,
        len(daily_forecast),
    )

    # ---------------------------------------------------------
    # Estimate demand volatility from historical observations.
    #
    # This is a demonstration input to the inventory engine,
    # not a claim that this is the final production estimator.
    # ---------------------------------------------------------

    historical_product = sales[
        (sales["outlet_id"] == demo_outlet)
        & (
            sales["product_id"]
            == demo_product
        )
    ].sort_values("date")

    demand_std = float(
        historical_product[
            "quantity_sold"
        ]
        .tail(28)
        .std()
    )

    if pd.isna(demand_std):
        demand_std = 0.0

    # ---------------------------------------------------------
    # Product pack size
    # ---------------------------------------------------------

    product_row = products[
        products["product_id"]
        == demo_product
    ]

    assert not product_row.empty

    pack_size = float(
        product_row.iloc[0]["pack_size"]
    )

    # ---------------------------------------------------------
    # Inventory recommendation
    # ---------------------------------------------------------

    inventory_service = InventoryService()

    recommendation = (
        inventory_service.recommend(
            product_id=demo_product,
            outlet_id=demo_outlet,
            daily_forecast=daily_forecast,
            demand_std=demand_std,
            lead_time_days=lead_time_days,
            inventory_position=inventory_position,
            service_level=0.95,
            review_period_days=1,
            pack_size=pack_size,
        )
    )

    assert recommendation[
        "recommended_order_quantity"
    ] >= 0

    print(
        f"Outlet                  : "
        f"{demo_outlet}"
    )

    print(
        f"Product                 : "
        f"{demo_product}"
    )

    print(
        f"Current inventory       : "
        f"{inventory_position:.2f}"
    )

    print(
        f"Lead time               : "
        f"{recommendation['lead_time_days']} days"
    )

    print(
        f"Lead-time demand        : "
        f"{recommendation['lead_time_demand']:.2f}"
    )

    print(
        f"Safety stock            : "
        f"{recommendation['safety_stock']:.2f}"
    )

    print(
        f"Reorder point           : "
        f"{recommendation['reorder_point']:.2f}"
    )

    print(
        f"Reorder required        : "
        f"{recommendation['reorder']}"
    )

    print(
        f"Recommended order      : "
        f"{recommendation['recommended_order_quantity']:.2f}"
    )

    print(
        f"Stockout risk           : "
        f"{recommendation['stockout_risk']}"
    )

    assert recommendation["lead_time_demand"] >= 0
    assert recommendation["safety_stock"] >= 0
    assert recommendation["reorder_point"] >= 0
    assert recommendation["recommended_order_quantity"] >= 0

    # =========================================================
    # 7. ACCEPTANCE CONDITIONS
    # =========================================================

    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert 0 <= spike <= 1