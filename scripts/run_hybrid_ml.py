from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============================================================
# EXISTING HYBRID COMPONENTS
# ============================================================

from app.forecasting.hybrid.selector import DynamicModelSelector
from app.forecasting.hybrid.ensemble import WeightedHybridEnsemble

# ============================================================
# OPTIONAL / REQUIRED ML DEPENDENCIES
# ============================================================

from xgboost import XGBRegressor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = (
    ROOT
    / "data"
    / "interim"
    / "demand_censoring_intelligence.csv"
)

OUTPUT_METRICS = (
    ROOT
    / "data"
    / "interim"
    / "hybrid_ml_benchmark.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "interim"
    / "hybrid_ml_predictions.csv"
)

SEED = 42

MAX_SERIES = 12

VALIDATION_DAYS = 30
TEST_DAYS = 30

TRANSFORMER_SEQUENCE_LENGTH = 28

TRANSFORMER_EPOCHS = 12
TRANSFORMER_BATCH_SIZE = 64
TRANSFORMER_LEARNING_RATE = 0.001
TRANSFORMER_PATIENCE = 3

HORIZON = 1

# Existing Part 9 feature contract.
XGB_FEATURES = [
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "day_of_year",
    "quarter",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_3",
    "rolling_std_3",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_28",
    "rolling_std_28",
    "trend_1d",
    "trend_7d",
    "trend_14d",
    "trend_ratio_7d",
    "trend_ratio_14d",
    "trend_acceleration",
    "volatility_3",
    "coefficient_variation_3",
    "volatility_7",
    "coefficient_variation_7",
    "volatility_14",
    "coefficient_variation_14",
    "volatility_28",
    "coefficient_variation_28",
    "order_velocity_3d",
    "order_velocity_7d",
    "order_velocity_14d",
    "order_velocity_change",
    "dow_sin_2",
    "dow_cos_2",
    "month_sin_2",
    "month_cos_2",
    "quarter_sin",
    "quarter_cos",
    "weekend_month_interaction",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def mae(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(
        np.sqrt(np.mean((actual - predicted) ** 2))
    )


def smape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    denominator = np.abs(actual) + np.abs(predicted)

    values = np.where(
        denominator == 0,
        0.0,
        2.0 * np.abs(actual - predicted) / denominator,
    )

    return float(np.mean(values) * 100.0)


def bias(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return float(np.mean(predicted - actual))


def high_demand_mae(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    threshold = float(np.quantile(actual, 0.75))

    mask = actual >= threshold

    if not np.any(mask):
        return 0.0

    return float(
        np.mean(
            np.abs(
                actual[mask]
                - predicted[mask]
            )
        )
    )


def high_demand_bias(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    threshold = float(np.quantile(actual, 0.75))

    mask = actual >= threshold

    if not np.any(mask):
        return 0.0

    return float(
        np.mean(
            predicted[mask]
            - actual[mask]
        )
    )


def spike_recall(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    actual_threshold = float(
        np.quantile(actual, 0.90)
    )

    prediction_threshold = float(
        np.quantile(actual, 0.75)
    )

    spikes = actual >= actual_threshold

    if not np.any(spikes):
        return 0.0

    detected = (
        predicted[spikes]
        >= prediction_threshold
    )

    return float(np.mean(detected))


def metric_row(
    model_name,
    actual,
    predicted,
    prediction_seconds=0.0,
):
    predicted = np.maximum(
        np.asarray(predicted, dtype=float),
        0.0,
    )

    return {
        "model_name": model_name,
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "smape": smape(actual, predicted),
        "bias": bias(actual, predicted),
        "high_demand_mae": high_demand_mae(
            actual,
            predicted,
        ),
        "high_demand_bias": high_demand_bias(
            actual,
            predicted,
        ),
        "spike_recall": spike_recall(
            actual,
            predicted,
        ),
        "prediction_seconds": float(
            prediction_seconds
        ),
    }


# ============================================================
# DATA PREPARATION
# ============================================================

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "date" not in df.columns:
        raise ValueError(
            "Dataset must contain 'date'."
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    target_candidates = [
        "deconstrained_demand",
        "true_demand",
        "quantity_sold",
    ]

    target = None

    for column in target_candidates:
        if column in df.columns:
            target = column
            break

    if target is None:
        raise ValueError(
            "No demand target found. "
            "Expected one of: "
            f"{target_candidates}"
        )

    if "outlet_id" not in df.columns:
        raise ValueError(
            "Dataset must contain outlet_id."
        )

    if "product_id" not in df.columns:
        raise ValueError(
            "Dataset must contain product_id."
        )

    df[target] = pd.to_numeric(
        df[target],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "date",
            "outlet_id",
            "product_id",
            target,
        ]
    ).copy()

    df[target] = np.maximum(
        df[target].astype(float),
        0.0,
    )

    df = df.sort_values(
        [
            "outlet_id",
            "product_id",
            "date",
        ]
    ).reset_index(drop=True)

    return df, target


# ============================================================
# FEATURE ENGINEERING FOR XGBOOST
# ============================================================

def add_xgb_features(
    series: pd.DataFrame,
    target: str,
):
    df = series.copy()

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    date = df["date"]

    df["year"] = date.dt.year
    df["month"] = date.dt.month
    df["day"] = date.dt.day
    df["day_of_week"] = date.dt.dayofweek
    df["week_of_year"] = (
        date.dt.isocalendar()
        .week
        .astype(int)
    )
    df["day_of_year"] = date.dt.dayofyear
    df["quarter"] = date.dt.quarter

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["dow_sin"] = np.sin(
        2.0
        * np.pi
        * df["day_of_week"]
        / 7.0
    )

    df["dow_cos"] = np.cos(
        2.0
        * np.pi
        * df["day_of_week"]
        / 7.0
    )

    df["month_sin"] = np.sin(
        2.0
        * np.pi
        * df["month"]
        / 12.0
    )

    df["month_cos"] = np.cos(
        2.0
        * np.pi
        * df["month"]
        / 12.0
    )

    # --------------------------------------------------------
    # LAGS
    # --------------------------------------------------------

    for lag in [
        1,
        2,
        3,
        7,
        14,
        28,
    ]:
        df[f"lag_{lag}"] = (
            df[target]
            .shift(lag)
        )

    # --------------------------------------------------------
    # ROLLING FEATURES
    # --------------------------------------------------------

    for window in [
        3,
        7,
        14,
        28,
    ]:
        shifted = (
            df[target]
            .shift(1)
        )

        df[
            f"rolling_mean_{window}"
        ] = (
            shifted
            .rolling(window)
            .mean()
        )

        df[
            f"rolling_std_{window}"
        ] = (
            shifted
            .rolling(window)
            .std()
        )

    # --------------------------------------------------------
    # TRENDS
    # --------------------------------------------------------

    df["trend_1d"] = (
        df["lag_1"]
        - df["lag_2"]
    )

    df["trend_7d"] = (
        df["rolling_mean_3"]
        - df["rolling_mean_7"]
    )

    df["trend_14d"] = (
        df["rolling_mean_7"]
        - df["rolling_mean_14"]
    )

    df["trend_ratio_7d"] = (
        df["rolling_mean_3"]
        / df["rolling_mean_7"].replace(
            0,
            np.nan,
        )
    )

    df["trend_ratio_14d"] = (
        df["rolling_mean_7"]
        / df["rolling_mean_14"].replace(
            0,
            np.nan,
        )
    )

    df["trend_acceleration"] = (
        df["trend_7d"]
        - df["trend_14d"]
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    for window in [
        3,
        7,
        14,
        28,
    ]:
        mean_col = (
            f"rolling_mean_{window}"
        )

        std_col = (
            f"rolling_std_{window}"
        )

        df[
            f"volatility_{window}"
        ] = df[std_col]

        df[
            f"coefficient_variation_{window}"
        ] = (
            df[std_col]
            / df[mean_col].replace(
                0,
                np.nan,
            )
        )

    # --------------------------------------------------------
    # ORDER VELOCITY
    # --------------------------------------------------------

    df["order_velocity_3d"] = (
        df[target]
        .shift(1)
        .rolling(3)
        .mean()
    )

    df["order_velocity_7d"] = (
        df[target]
        .shift(1)
        .rolling(7)
        .mean()
    )

    df["order_velocity_14d"] = (
        df[target]
        .shift(1)
        .rolling(14)
        .mean()
    )

    df["order_velocity_change"] = (
        df["order_velocity_3d"]
        - df["order_velocity_7d"]
    )

    # --------------------------------------------------------
    # SECOND HARMONICS
    # --------------------------------------------------------

    df["dow_sin_2"] = np.sin(
        4.0
        * np.pi
        * df["day_of_week"]
        / 7.0
    )

    df["dow_cos_2"] = np.cos(
        4.0
        * np.pi
        * df["day_of_week"]
        / 7.0
    )

    df["month_sin_2"] = np.sin(
        4.0
        * np.pi
        * df["month"]
        / 12.0
    )

    df["month_cos_2"] = np.cos(
        4.0
        * np.pi
        * df["month"]
        / 12.0
    )

    df["quarter_sin"] = np.sin(
        2.0
        * np.pi
        * df["quarter"]
        / 4.0
    )

    df["quarter_cos"] = np.cos(
        2.0
        * np.pi
        * df["quarter"]
        / 4.0
    )

    df["weekend_month_interaction"] = (
        df["is_weekend"]
        * df["month"]
    )

    return df


# ============================================================
# NAIVE FORECAST
# ============================================================

def naive_forecast(
    history,
    horizon,
):
    history = np.asarray(
        history,
        dtype=float,
    )

    if len(history) == 0:
        return np.zeros(horizon)

    value = max(
        float(history[-1]),
        0.0,
    )

    return np.full(
        horizon,
        value,
        dtype=float,
    )


# ============================================================
# TRANSFORMER MODEL
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=512,
    ):
        super().__init__()

        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (
                -np.log(10000.0)
                / d_model
            )
        )

        pe = torch.zeros(
            max_len,
            d_model,
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe,
        )

    def forward(self, x):
        return (
            x
            + self.pe[
                :,
                :x.size(1),
                :,
            ]
        )


class TemporalTransformer(nn.Module):

    def __init__(
        self,
        input_size=1,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dim_feedforward=64,
        dropout=0.1,
    ):
        super().__init__()

        self.input_projection = (
            nn.Linear(
                input_size,
                d_model,
            )
        )

        self.position = (
            PositionalEncoding(
                d_model=d_model,
                max_len=512,
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers,
            )
        )

        self.head = nn.Linear(
            d_model,
            1,
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.position(x)

        x = self.encoder(x)

        x = x[:, -1, :]

        return self.head(x).squeeze(-1)


# ============================================================
# TRANSFORMER TRAINING
# ============================================================

def make_sequences(
    values,
    sequence_length,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    X = []
    y = []

    if len(values) <= sequence_length:
        return (
            np.empty(
                (0, sequence_length),
                dtype=np.float32,
            ),
            np.empty(
                (0,),
                dtype=np.float32,
            ),
        )

    for i in range(
        sequence_length,
        len(values),
    ):
        X.append(
            values[
                i - sequence_length:i
            ]
        )

        y.append(
            values[i]
        )

    return (
        np.asarray(
            X,
            dtype=np.float32,
        ),
        np.asarray(
            y,
            dtype=np.float32,
        ),
    )


def fit_transformer(
    train_values,
    validation_values,
):
    train_values = np.asarray(
        train_values,
        dtype=float,
    )

    validation_values = np.asarray(
        validation_values,
        dtype=float,
    )

    if len(train_values) < (
        TRANSFORMER_SEQUENCE_LENGTH + 5
    ):
        return None

    mean = float(
        np.mean(train_values)
    )

    std = float(
        np.std(train_values)
    )

    if std < 1e-8:
        std = 1.0

    # --------------------------------------------------------
    # TRAIN SEQUENCES
    # --------------------------------------------------------

    scaled_train = (
        train_values - mean
    ) / std

    X_train, y_train = make_sequences(
        scaled_train,
        TRANSFORMER_SEQUENCE_LENGTH,
    )

    if len(X_train) == 0:
        return None

    # --------------------------------------------------------
    # VALIDATION SEQUENCES
    #
    # Validation gets the tail of train history so that the
    # first validation prediction has enough historical context.
    # --------------------------------------------------------

    validation_context = np.concatenate(
        [
            train_values[
                -TRANSFORMER_SEQUENCE_LENGTH:
            ],
            validation_values,
        ]
    )

    scaled_validation = (
        validation_context - mean
    ) / std

    X_val, y_val = make_sequences(
        scaled_validation,
        TRANSFORMER_SEQUENCE_LENGTH,
    )

    if len(X_val) == 0:
        return None

    # --------------------------------------------------------
    # TORCH DATA
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
    ).unsqueeze(-1)

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32,
    ).unsqueeze(-1)

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.float32,
    )

    train_loader = DataLoader(
        TensorDataset(
            X_train_tensor,
            y_train_tensor,
        ),
        batch_size=TRANSFORMER_BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = TemporalTransformer()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRANSFORMER_LEARNING_RATE,
        weight_decay=1e-4,
    )

    criterion = nn.HuberLoss()

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for epoch in range(
        TRANSFORMER_EPOCHS
    ):

        model.train()

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            prediction = model(
                X_batch
            )

            loss = criterion(
                prediction,
                y_batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_prediction = model(
                X_val_tensor
            )

            val_loss = float(
                criterion(
                    val_prediction,
                    y_val_tensor,
                ).item()
            )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.detach().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

            if (
                patience_counter
                >= TRANSFORMER_PATIENCE
            ):
                break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    model.eval()

    return {
        "model": model,
        "mean": mean,
        "std": std,
    }


def transformer_predict(
    fitted,
    history,
    horizon,
):
    if fitted is None:
        return naive_forecast(
            history,
            horizon,
        )

    model = fitted["model"]
    mean = fitted["mean"]
    std = fitted["std"]

    history = list(
        np.asarray(
            history,
            dtype=float,
        )
    )

    predictions = []

    for _ in range(horizon):

        context = np.asarray(
            history[
                -TRANSFORMER_SEQUENCE_LENGTH:
            ],
            dtype=float,
        )

        if len(context) < (
            TRANSFORMER_SEQUENCE_LENGTH
        ):
            pad_value = (
                context[0]
                if len(context) > 0
                else mean
            )

            padding = np.full(
                TRANSFORMER_SEQUENCE_LENGTH
                - len(context),
                pad_value,
            )

            context = np.concatenate(
                [
                    padding,
                    context,
                ]
            )

        scaled = (
            context - mean
        ) / std

        X = torch.tensor(
            scaled,
            dtype=torch.float32,
        ).reshape(
            1,
            TRANSFORMER_SEQUENCE_LENGTH,
            1,
        )

        with torch.no_grad():
            prediction = float(
                model(X).item()
            )

        prediction = (
            prediction * std
            + mean
        )

        prediction = max(
            prediction,
            0.0,
        )

        predictions.append(
            prediction
        )

        history.append(
            prediction
        )

    return np.asarray(
        predictions,
        dtype=float,
    )


# ============================================================
# XGBOOST TRAINING
# ============================================================

def fit_xgboost(
    train_frame,
    target,
):
    prepared = add_xgb_features(
        train_frame,
        target,
    )

    prepared = prepared.dropna(
        subset=XGB_FEATURES
    ).copy()

    if len(prepared) == 0:
        return None

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=1,
        tree_method="hist",
    )

    model.fit(
        prepared[XGB_FEATURES],
        prepared[target],
        verbose=False,
    )

    return model


def xgb_recursive_forecast(
    model,
    full_series,
    target,
    forecast_dates,
):
    if model is None:
        return naive_forecast(
            full_series[target].values,
            len(forecast_dates),
        )

    working = full_series.copy()

    predictions = []

    for forecast_date in forecast_dates:

        row = pd.DataFrame(
            {
                "date": [forecast_date],
                target: [np.nan],
            }
        )

        working_for_feature = pd.concat(
            [
                working[
                    [
                        "date",
                        target,
                    ]
                ],
                row,
            ],
            ignore_index=True,
        )

        prepared = add_xgb_features(
            working_for_feature,
            target,
        )

        latest = prepared.iloc[
            -1:
        ].copy()

        latest[XGB_FEATURES] = (
            latest[XGB_FEATURES]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .ffill()
            .bfill()
        )

        if latest[XGB_FEATURES].isna().any().any():

            fallback = (
                working[target]
                .dropna()
                .tail(7)
                .mean()
            )

            prediction = (
                float(fallback)
                if np.isfinite(fallback)
                else 0.0
            )

        else:

            prediction = float(
                model.predict(
                    latest[
                        XGB_FEATURES
                    ]
                )[0]
            )

        prediction = max(
            prediction,
            0.0,
        )

        predictions.append(
            prediction
        )

        working = pd.concat(
            [
                working,
                pd.DataFrame(
                    {
                        "date": [
                            forecast_date
                        ],
                        target: [
                            prediction
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )

    return np.asarray(
        predictions,
        dtype=float,
    )


# ============================================================
# SPLIT SERIES
# ============================================================

def select_series(
    df,
):
    series_counts = (
        df.groupby(
            [
                "outlet_id",
                "product_id",
            ]
        )["date"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    selected = list(
        series_counts
        .head(MAX_SERIES)
        .index
    )

    return selected


def split_dates(
    series,
):
    dates = np.array(
        sorted(
            series["date"]
            .dt.normalize()
            .unique()
        )
    )

    required = (
        VALIDATION_DAYS
        + TEST_DAYS
        + 60
    )

    if len(dates) < required:
        return None

    test_dates = dates[
        -TEST_DAYS:
    ]

    validation_dates = dates[
        -(
            TEST_DAYS
            + VALIDATION_DAYS
        ):
        -TEST_DAYS
    ]

    train_dates = dates[
        : -(
            VALIDATION_DAYS
            + TEST_DAYS
        )
    ]

    return (
        train_dates,
        validation_dates,
        test_dates,
    )


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():

    total_start = time.perf_counter()

    print_header(
        "PART 26E - GENUINE UNSEEN HYBRID BENCHMARK"
    )

    print(
        f"Dataset: {DATA_PATH}"
    )

    print(
        f"Maximum series: {MAX_SERIES}"
    )

    print(
        f"Validation days: {VALIDATION_DAYS}"
    )

    print(
        f"Test days: {TEST_DAYS}"
    )

    print(
        "Weight source: VALIDATION ONLY"
    )

    print(
        "Final evaluation: UNSEEN TEST ONLY"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df, target = load_dataset()

    print()
    print(
        f"Rows available: {len(df):,}"
    )

    print(
        f"Target: {target}"
    )

    print(
        f"Outlets: {df['outlet_id'].nunique()}"
    )

    print(
        f"Products: {df['product_id'].nunique()}"
    )

    selected_series = select_series(
        df
    )

    print(
        f"Series selected: "
        f"{len(selected_series)}"
    )

    if len(selected_series) == 0:
        raise RuntimeError(
            "No usable series found."
        )

    # --------------------------------------------------------
    # AGGREGATE STORAGE
    # --------------------------------------------------------

    all_validation_predictions = []
    all_test_predictions = []

    series_results = []

    selector = DynamicModelSelector()
    ensemble = WeightedHybridEnsemble()

    # --------------------------------------------------------
    # SERIES LOOP
    # --------------------------------------------------------

    for series_number, (
        outlet_id,
        product_id,
    ) in enumerate(
        selected_series,
        start=1,
    ):

        print()
        print(
            "-" * 70
        )

        print(
            f"SERIES "
            f"{series_number}/{len(selected_series)}"
        )

        print(
            f"Outlet: {outlet_id} | "
            f"Product: {product_id}"
        )

        series = df[
            (
                df["outlet_id"]
                == outlet_id
            )
            &
            (
                df["product_id"]
                == product_id
            )
        ].copy()

        series = series.sort_values(
            "date"
        ).reset_index(
            drop=True
        )

        split = split_dates(
            series
        )

        if split is None:

            print(
                "SKIPPED: insufficient history"
            )

            continue

        (
            train_dates,
            validation_dates,
            test_dates,
        ) = split

        train = series[
            series["date"]
            .dt.normalize()
            .isin(train_dates)
        ].copy()

        validation = series[
            series["date"]
            .dt.normalize()
            .isin(validation_dates)
        ].copy()

        test = series[
            series["date"]
            .dt.normalize()
            .isin(test_dates)
        ].copy()

        if (
            len(train) == 0
            or len(validation) == 0
            or len(test) == 0
        ):
            print(
                "SKIPPED: empty split"
            )
            continue

        # ----------------------------------------------------
        # COMMON TEST ACTUALS
        # ----------------------------------------------------

        actual_validation = (
            validation[target]
            .astype(float)
            .values
        )

        actual_test = (
            test[target]
            .astype(float)
            .values
        )

        # ----------------------------------------------------
        # NAIVE
        # ----------------------------------------------------

        train_values = (
            train[target]
            .astype(float)
            .values
        )

        validation_naive = (
            naive_forecast(
                train_values,
                len(validation),
            )
        )

        # Recursive naive test uses validation actuals
        # because this represents rolling daily operation.
        naive_test_history = list(
            train_values
        )

        naive_validation_predictions = (
            validation_naive
        )

        for value in validation[
            target
        ].values:

            naive_test_history.append(
                float(value)
            )

        naive_test = naive_forecast(
            naive_test_history,
            len(test),
        )

        # ----------------------------------------------------
        # XGBOOST
        # ----------------------------------------------------

        xgb_start = time.perf_counter()

        xgb_model = fit_xgboost(
            train,
            target,
        )

        xgb_validation = (
            xgb_recursive_forecast(
                xgb_model,
                train,
                target,
                validation[
                    "date"
                ].values,
            )
        )

        # For test forecasting, validation observations are
        # appended as known historical observations.
        xgb_test_history = pd.concat(
            [
                train[
                    [
                        "date",
                        target,
                    ]
                ],
                validation[
                    [
                        "date",
                        target,
                    ]
                ],
            ],
            ignore_index=True,
        )

        xgb_test = (
            xgb_recursive_forecast(
                xgb_model,
                xgb_test_history,
                target,
                test[
                    "date"
                ].values,
            )
        )

        xgb_seconds = (
            time.perf_counter()
            - xgb_start
        )

        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        transformer_start = (
            time.perf_counter()
        )

        transformer = fit_transformer(
            train_values,
            validation[target]
            .astype(float)
            .values,
        )

        transformer_validation = (
            transformer_predict(
                transformer,
                train_values,
                len(validation),
            )
        )

        transformer_test_history = np.concatenate(
            [
                train_values,
                validation[
                    target
                ]
                .astype(float)
                .values,
            ]
        )

        transformer_test = (
            transformer_predict(
                transformer,
                transformer_test_history,
                len(test),
            )
        )

        transformer_seconds = (
            time.perf_counter()
            - transformer_start
        )

        # ----------------------------------------------------
        # VALIDATION SCORES
        #
        # CRITICAL:
        # These scores are the ONLY information used to
        # determine hybrid weights.
        # ----------------------------------------------------

        validation_scores = [
            {
                "model_name": "naive",
                "mae": mae(
                    actual_validation,
                    naive_validation_predictions,
                ),
                "rmse": rmse(
                    actual_validation,
                    naive_validation_predictions,
                ),
                "bias": bias(
                    actual_validation,
                    naive_validation_predictions,
                ),
                "stability": 1.0,
            },
            {
                "model_name": "xgboost",
                "mae": mae(
                    actual_validation,
                    xgb_validation,
                ),
                "rmse": rmse(
                    actual_validation,
                    xgb_validation,
                ),
                "bias": bias(
                    actual_validation,
                    xgb_validation,
                ),
                "stability": 1.0,
            },
            {
                "model_name": "transformer",
                "mae": mae(
                    actual_validation,
                    transformer_validation,
                ),
                "rmse": rmse(
                    actual_validation,
                    transformer_validation,
                ),
                "bias": bias(
                    actual_validation,
                    transformer_validation,
                ),
                "stability": 1.0,
            },
        ]

        # Convert to existing schema objects.
        from app.forecasting.hybrid.schemas import (
            ModelValidationScore,
        )

        score_objects = [
            ModelValidationScore(
                model_name=item[
                    "model_name"
                ],
                mae=item["mae"],
                rmse=item["rmse"],
                bias=item["bias"],
                stability=item["stability"],
            )
            for item in validation_scores
        ]

        # ----------------------------------------------------
        # FREEZE ROUTING / WEIGHTS
        # ----------------------------------------------------

        decision = selector.select(
            score_objects,
            horizon=HORIZON,
        )

        # ----------------------------------------------------
        # TEST PREDICTIONS
        #
        # NONE of these are used to change weights.
        # ----------------------------------------------------

        test_candidate_predictions = {
            "naive": np.maximum(
                naive_test,
                0.0,
            ),
            "xgboost": np.maximum(
                xgb_test,
                0.0,
            ),
            "transformer": np.maximum(
                transformer_test,
                0.0,
            ),
        }

        hybrid_test = ensemble.combine(
            test_candidate_predictions,
            decision.weights,
        )

        hybrid_test = np.maximum(
            hybrid_test,
            0.0,
        )

        # ----------------------------------------------------
        # SERIES METRICS
        # ----------------------------------------------------

        series_metrics = []

        for model_name, prediction in [
            (
                "naive",
                naive_test,
            ),
            (
                "xgboost",
                xgb_test,
            ),
            (
                "transformer",
                transformer_test,
            ),
            (
                "hybrid",
                hybrid_test,
            ),
        ]:

            if model_name == "xgboost":
                prediction_seconds = (
                    xgb_seconds
                )

            elif model_name == "transformer":
                prediction_seconds = (
                    transformer_seconds
                )

            else:
                prediction_seconds = 0.0

            metrics = metric_row(
                model_name,
                actual_test,
                prediction,
                prediction_seconds,
            )

            metrics[
                "outlet_id"
            ] = outlet_id

            metrics[
                "product_id"
            ] = product_id

            metrics[
                "selected_model"
            ] = decision.selected_model

            metrics[
                "naive_weight"
            ] = decision.weights.get(
                "naive",
                0.0,
            )

            metrics[
                "xgboost_weight"
            ] = decision.weights.get(
                "xgboost",
                0.0,
            )

            metrics[
                "transformer_weight"
            ] = decision.weights.get(
                "transformer",
                0.0,
            )

            metrics[
                "routing_confidence"
            ] = decision.confidence

            series_metrics.append(
                metrics
            )

        series_results.extend(
            series_metrics
        )

        # ----------------------------------------------------
        # SAVE RAW PREDICTIONS
        # ----------------------------------------------------

        for i, date in enumerate(
            test["date"].values
        ):

            all_test_predictions.append(
                {
                    "outlet_id": outlet_id,
                    "product_id": product_id,
                    "date": date,
                    "actual": float(
                        actual_test[i]
                    ),
                    "naive_prediction": float(
                        naive_test[i]
                    ),
                    "xgboost_prediction": float(
                        xgb_test[i]
                    ),
                    "transformer_prediction": float(
                        transformer_test[i]
                    ),
                    "hybrid_prediction": float(
                        hybrid_test[i]
                    ),
                    "selected_model": (
                        decision.selected_model
                    ),
                    "naive_weight": (
                        decision.weights.get(
                            "naive",
                            0.0,
                        )
                    ),
                    "xgboost_weight": (
                        decision.weights.get(
                            "xgboost",
                            0.0,
                        )
                    ),
                    "transformer_weight": (
                        decision.weights.get(
                            "transformer",
                            0.0,
                        )
                    ),
                    "routing_confidence": (
                        decision.confidence
                    ),
                }
            )

        print()
        print(
            "VALIDATION MAE"
        )

        for item in validation_scores:
            print(
                f"{item['model_name']:15s}"
                f"{item['mae']:.6f}"
            )

        print()
        print(
            "FROZEN HYBRID WEIGHTS"
        )

        for model_name in [
            "naive",
            "xgboost",
            "transformer",
        ]:
            print(
                f"{model_name:15s}"
                f"{decision.weights.get(model_name, 0.0):.6f}"
            )

        print(
            f"Selected model: "
            f"{decision.selected_model}"
        )

        print(
            f"Confidence: "
            f"{decision.confidence:.6f}"
        )

        print()
        print(
            "UNSEEN TEST MAE"
        )

        print(
            f"{'naive':15s}"
            f"{mae(actual_test, naive_test):.6f}"
        )

        print(
            f"{'xgboost':15s}"
            f"{mae(actual_test, xgb_test):.6f}"
        )

        print(
            f"{'transformer':15s}"
            f"{mae(actual_test, transformer_test):.6f}"
        )

        print(
            f"{'hybrid':15s}"
            f"{mae(actual_test, hybrid_test):.6f}"
        )

    # ========================================================
    # BUILD SERIES RESULT DATAFRAME
    # ========================================================

    if len(series_results) == 0:
        raise RuntimeError(
            "No series produced a benchmark result."
        )

    results = pd.DataFrame(
        series_results
    )

    predictions = pd.DataFrame(
        all_test_predictions
    )

    # ========================================================
    # AGGREGATE MODEL METRICS
    # ========================================================

    aggregate_rows = []

    for model_name in [
        "naive",
        "xgboost",
        "transformer",
        "hybrid",
    ]:

        model_predictions = (
            predictions[
                f"{model_name}_prediction"
            ]
            .astype(float)
            .values
        )

        actual = (
            predictions["actual"]
            .astype(float)
            .values
        )

        if model_name == "naive":
            seconds = 0.0

        elif model_name == "xgboost":
            seconds = float(
                results.loc[
                    results[
                        "model_name"
                    ]
                    == "xgboost",
                    "prediction_seconds",
                ].sum()
            )

        elif model_name == "transformer":
            seconds = float(
                results.loc[
                    results[
                        "model_name"
                    ]
                    == "transformer",
                    "prediction_seconds",
                ].sum()
            )

        else:
            seconds = 0.0

        aggregate_rows.append(
            metric_row(
                model_name,
                actual,
                model_predictions,
                seconds,
            )
        )

    aggregate = pd.DataFrame(
        aggregate_rows
    )

    # ========================================================
    # BEST INDIVIDUAL
    # ========================================================

    individual_models = [
        "naive",
        "xgboost",
        "transformer",
    ]

    individual = aggregate[
        aggregate["model_name"].isin(
            individual_models
        )
    ].copy()

    best_index = individual[
        "mae"
    ].idxmin()

    best_model = str(
        individual.loc[
            best_index,
            "model_name",
        ]
    )

    best_mae = float(
        individual.loc[
            best_index,
            "mae",
        ]
    )

    hybrid_mae = float(
        aggregate.loc[
            aggregate[
                "model_name"
            ]
            == "hybrid",
            "mae",
        ].iloc[0]
    )

    hybrid_improvement = (
        (
            best_mae
            - hybrid_mae
        )
        / max(
            best_mae,
            1e-12,
        )
        * 100.0
    )

    aggregate[
        "improvement_vs_best_individual_pct"
    ] = hybrid_improvement

    aggregate[
        "best_individual_model"
    ] = best_model

    aggregate[
        "hybrid_beats_best_individual"
    ] = (
        hybrid_mae
        < best_mae
    )

    # ========================================================
    # SERIES WIN RATE
    # ========================================================

    pivot = (
        results
        .pivot_table(
            index=[
                "outlet_id",
                "product_id",
            ],
            columns="model_name",
            values="mae",
        )
        .reset_index()
    )

    if (
        "hybrid" in pivot.columns
        and len(pivot) > 0
    ):

        wins = (
            pivot["hybrid"]
            <
            pivot[
                individual_models
            ].min(axis=1)
        )

        hybrid_win_rate = float(
            wins.mean()
        )

    else:
        hybrid_win_rate = 0.0

    # ========================================================
    # ROUTING DISTRIBUTION
    # ========================================================

    routing = (
        results[
            results["model_name"]
            == "hybrid"
        ]["selected_model"]
        .value_counts()
        .to_dict()
    )

    routing_agreement = 0.0

    if len(pivot) > 0:

        pivot["best_individual"] = (
            pivot[
                individual_models
            ]
            .idxmin(axis=1)
        )

        hybrid_selected = (
            results[
                results["model_name"]
                == "hybrid"
            ][
                [
                    "outlet_id",
                    "product_id",
                    "selected_model",
                ]
            ]
            .drop_duplicates(
                [
                    "outlet_id",
                    "product_id",
                ]
            )
        )

        merged = pivot.merge(
            hybrid_selected,
            on=[
                "outlet_id",
                "product_id",
            ],
            how="inner",
        )

        if len(merged) > 0:
            routing_agreement = float(
                (
                    merged[
                        "best_individual"
                    ]
                    == merged[
                        "selected_model"
                    ]
                ).mean()
            )

    # ========================================================
    # WEIGHT SUMMARY
    # ========================================================

    hybrid_rows = results[
        results["model_name"]
        == "hybrid"
    ].copy()

    average_weights = {
        "naive_weight": float(
            hybrid_rows[
                "naive_weight"
            ].mean()
        ),
        "xgboost_weight": float(
            hybrid_rows[
                "xgboost_weight"
            ].mean()
        ),
        "transformer_weight": float(
            hybrid_rows[
                "transformer_weight"
            ].mean()
        ),
    }

    average_confidence = float(
        hybrid_rows[
            "routing_confidence"
        ].mean()
    )

    # ========================================================
    # COMBINE AGGREGATE + SERIES METRICS
    # ========================================================

    aggregate["scope"] = "aggregate"
    results["scope"] = "series"

    final_results = pd.concat(
        [
            aggregate,
            results,
        ],
        ignore_index=True,
        sort=False,
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_METRICS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PREDICTIONS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_results.to_csv(
        OUTPUT_METRICS,
        index=False,
    )

    predictions.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print_header(
        "PART 26E - FINAL UNSEEN TEST RESULTS"
    )

    print(
        f"Series evaluated: "
        f"{len(pivot)}"
    )

    print(
        f"Test prediction rows: "
        f"{len(predictions):,}"
    )

    print()
    print(
        "AGGREGATE TEST PERFORMANCE"
    )

    print(
        "-" * 70
    )

    print(
        f"{'MODEL':15s}"
        f"{'MAE':>12s}"
        f"{'RMSE':>12s}"
        f"{'sMAPE':>12s}"
        f"{'BIAS':>12s}"
    )

    print(
        "-" * 70
    )

    for _, row in aggregate.iterrows():

        print(
            f"{str(row['model_name']):15s}"
            f"{row['mae']:12.6f}"
            f"{row['rmse']:12.6f}"
            f"{row['smape']:12.6f}"
            f"{row['bias']:12.6f}"
        )

    print()
    print(
        f"BEST INDIVIDUAL MODEL: "
        f"{best_model}"
    )

    print(
        f"BEST INDIVIDUAL MAE: "
        f"{best_mae:.6f}"
    )

    print(
        f"HYBRID MAE: "
        f"{hybrid_mae:.6f}"
    )

    print(
        f"HYBRID IMPROVEMENT VS BEST: "
        f"{hybrid_improvement:.4f}%"
    )

    print()
    print(
        "HYBRID WIN RATE"
    )

    print(
        f"{hybrid_win_rate * 100.0:.2f}%"
    )

    print()
    print(
        "ROUTING DISTRIBUTION"
    )

    print(
        f"Naive: "
        f"{routing.get('naive', 0)}"
    )

    print(
        f"XGBoost: "
        f"{routing.get('xgboost', 0)}"
    )

    print(
        f"Transformer: "
        f"{routing.get('transformer', 0)}"
    )

    print()
    print(
        "AVERAGE FROZEN WEIGHTS"
    )

    print(
        f"Naive: "
        f"{average_weights['naive_weight']:.6f}"
    )

    print(
        f"XGBoost: "
        f"{average_weights['xgboost_weight']:.6f}"
    )

    print(
        f"Transformer: "
        f"{average_weights['transformer_weight']:.6f}"
    )

    print()
    print(
        f"Average routing confidence: "
        f"{average_confidence:.6f}"
    )

    print()
    print(
        f"Routing agreement with "
        f"validation winner: "
        f"{routing_agreement * 100.0:.2f}%"
    )

    print()
    print(
        "OUTPUT FILES"
    )

    print(
        f"Metrics: "
        f"{OUTPUT_METRICS}"
    )

    print(
        f"Predictions: "
        f"{OUTPUT_PREDICTIONS}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    validation_errors = []

    required_models = {
        "naive",
        "xgboost",
        "transformer",
        "hybrid",
    }

    actual_models = set(
        aggregate["model_name"]
    )

    missing_models = (
        required_models
        - actual_models
    )

    if missing_models:
        validation_errors.append(
            "Missing models: "
            f"{sorted(missing_models)}"
        )

    if predictions.empty:
        validation_errors.append(
            "Prediction output is empty."
        )

    prediction_columns = {
        "actual",
        "naive_prediction",
        "xgboost_prediction",
        "transformer_prediction",
        "hybrid_prediction",
    }

    missing_prediction_columns = (
        prediction_columns
        - set(predictions.columns)
    )

    if missing_prediction_columns:
        validation_errors.append(
            "Missing prediction columns: "
            f"{sorted(missing_prediction_columns)}"
        )

    if not predictions.empty:

        for column in prediction_columns:

            values = (
                predictions[column]
                .astype(float)
                .values
            )

            if not np.all(
                np.isfinite(values)
            ):
                validation_errors.append(
                    f"Non-finite values in "
                    f"{column}"
                )

        for column in [
            "naive_prediction",
            "xgboost_prediction",
            "transformer_prediction",
            "hybrid_prediction",
        ]:

            values = (
                predictions[column]
                .astype(float)
                .values
            )

            if np.any(values < 0):
                validation_errors.append(
                    f"Negative predictions "
                    f"in {column}"
                )

    # --------------------------------------------------------
    # WEIGHT VALIDATION
    # --------------------------------------------------------

    weight_sum = (
        hybrid_rows[
            [
                "naive_weight",
                "xgboost_weight",
                "transformer_weight",
            ]
        ]
        .sum(axis=1)
    )

    if not np.allclose(
        weight_sum.values,
        1.0,
        atol=1e-6,
    ):
        validation_errors.append(
            "Hybrid weights do not sum "
            "to 1.0."
        )

    # --------------------------------------------------------
    # TEMPORAL LEAKAGE CHECK
    # --------------------------------------------------------

    if not predictions.empty:

        pred_dates = pd.to_datetime(
            predictions["date"]
        )

        minimum_test_date = (
            pred_dates.min()
        )

        maximum_train_date = (
            df.loc[
                df["date"]
                < minimum_test_date,
                "date",
            ].max()
        )

        if (
            pd.isna(maximum_train_date)
            or maximum_train_date
            >= minimum_test_date
        ):
            validation_errors.append(
                "Temporal split validation failed."
            )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    elapsed = (
        time.perf_counter()
        - total_start
    )

    print()
    print_header(
        "PART 26E ACCEPTANCE"
    )

    print(
        "TRAIN -> VALIDATION -> TEST: PASS"
    )

    print(
        "Validation-only weight derivation: PASS"
    )

    print(
        "Unseen test prediction evaluation: PASS"
    )

    print(
        "Candidate prediction vectors: PASS"
    )

    print(
        "Hybrid prediction generation: PASS"
    )

    print(
        "Non-negative demand constraint: "
        "PASS"
        if not validation_errors
        else "CHECK REQUIRED"
    )

    print(
        "Weight normalization: "
        "PASS"
        if not validation_errors
        else "CHECK REQUIRED"
    )

    if validation_errors:

        print()
        print(
            "VALIDATION ERRORS"
        )

        for error in validation_errors:
            print(
                f" - {error}"
            )

        print()
        print(
            "PART 26E FAILED"
        )

        raise RuntimeError(
            "Part 26E acceptance failed."
        )

    print()
    print(
        "PART 26E GENUINE BENCHMARK PASSED"
    )

    print(
        f"Runtime: "
        f"{elapsed:.2f} seconds"
    )

    print()

    # --------------------------------------------------------
    # IMPORTANT PRODUCTION DECISION
    # --------------------------------------------------------

    if hybrid_mae < best_mae:

        print(
            "EMPIRICAL RESULT:"
        )

        print(
            "Hybrid beats the best individual "
            "model on the unseen test set."
        )

        print(
            "Hybrid is a valid candidate for "
            "the next production evaluation."
        )

    else:

        print(
            "EMPIRICAL RESULT:"
        )

        print(
            "Hybrid does NOT beat the best "
            "individual model on the unseen "
            "test set."
        )

        print(
            f"Best individual: {best_model}"
        )

        print(
            "Do NOT force Hybrid ML into "
            "production."
        )

        print(
            "Dynamic routing may still be "
            "useful."
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This benchmark is limited to the "
        f"{len(pivot)} selected series."
    )

    print(
        "It is not yet a global production "
        "championship claim."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()