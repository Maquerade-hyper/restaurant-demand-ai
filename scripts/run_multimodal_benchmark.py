"""
PART 27E V5
Genuine Low-Memory Multimodal Benchmark

Purpose
-------
Benchmark whether genuinely available:
    1. historical demand
    2. safe numeric context
    3. safe text context
    4. genuine visual features
    5. multimodal combination

actually improve unseen chronological forecasting.

Design constraints
------------------
- CPU / low-memory friendly
- Maximum 24 outlet-product series
- Compact dataset only
- Global LightGBM model
- One model at a time
- No pandas apply()
- Strict forecast-time leakage firewall
- No fake visual features
- No target-derived context
- Common chronological train/validation/test split
- Validation is used for model selection only
- Test is unseen
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from lightgbm import LGBMRegressor


# ============================================================================
# PATHS / CONSTANTS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "interim" / "demand_censoring_intelligence.csv"
OUTPUT_PATH = ROOT / "data" / "interim" / "multimodal_ablation_v5.csv"

MAX_SERIES = 24
VALIDATION_DAYS = 30
TEST_DAYS = 30

RANDOM_STATE = 42

TARGET = "deconstrained_demand"

REQUIRED_COLUMNS = [
    "date",
    "outlet_id",
    "product_id",
    TARGET,
]

# ---------------------------------------------------------------------------
# Forecast-time firewall
#
# These are never allowed to become model features.
# ---------------------------------------------------------------------------

FORBIDDEN_FEATURES = {
    # target / direct target aliases
    "quantity_sold",
    "deconstrained_demand",
    "true_demand",
    "actual_demand",
    "future_demand",
    "target",
    "prediction",
    "predicted_demand",

    # sales / revenue
    "revenue",

    # inventory / future operational state
    "closing_stock",
    "opening_stock",
    "ending_stock",
    "inventory",
    "inventory_position",
    "wastage",

    # stockout / censoring outputs derived from realized demand
    "stockout",
    "stockout_duration",
    "estimated_lost_demand",
    "estimated_fulfillment_rate",
    "demand_constrained_estimated",
    "stockout_event_start",
    "stockout_event_id",
    "stockout_event_active",
    "stockout_event_day",
    "pre_stockout_demand",
    "pre_stockout_sales",
    "sales_demand_gap",
    "censoring_ratio",
    "pre_stockout_demand_pressure",
    "censored_demand_flag",
    "censoring_strength",
    "stockout_end",
    "recovery_event",
    "days_since_stockout",
    "clean_reference_demand",
    "recovery_ratio",
    "post_stockout_recovery_flag",
    "recovery_strength",

    # truth / leakage
    "demand_truth",
    "lost_demand_truth",
    "future_quantity_sold",
}

# Explicitly unsafe keywords.
FORBIDDEN_KEYWORDS = (
    "actual",
    "truth",
    "future",
    "target",
    "prediction",
    "predicted",
    "closing_stock",
    "opening_stock",
    "ending_stock",
    "inventory_position",
    "stockout",
    "censor",
    "lost_demand",
    "recovery",
    "sales_demand_gap",
)


# ============================================================================
# LOGGING
# ============================================================================

def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def section(text: str) -> None:
    print("\n" + "-" * 72)
    print(text)
    print("-" * 72)


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def ensure_columns(df: pd.DataFrame, columns: List[str]) -> None:
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Required columns missing from dataset: {missing}"
        )


def is_forbidden_column(name: str) -> bool:
    """
    Strict forecast-time firewall.

    A column is unsafe if:
      - explicitly forbidden
      - contains a forbidden semantic keyword
    """
    n = str(name).strip().lower()

    if n in FORBIDDEN_FEATURES:
        return True

    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in n:
            return True

    return False


def is_known_context_column(name: str) -> bool:
    """
    Detect context that can legitimately be known at forecast time.

    This function is intentionally conservative.
    """
    n = str(name).strip().lower()

    if is_forbidden_column(n):
        return False

    context_keywords = (
        "holiday",
        "promotion",
        "promo",
        "event",
        "weather",
        "temperature",
        "temp",
        "rain",
        "rainfall",
        "snow",
        "wind",
        "tourism",
        "tourist",
        "student",
        "university",
        "calendar",
        "cultural",
        "religious",
        "demographic",
        "festival",
        "weekend",
        "season",
        "location",
    )

    return any(k in n for k in context_keywords)


def is_visual_column(name: str) -> bool:
    """
    Genuine visual feature detection.

    We only accept explicitly visual/CV/embedding fields.

    We DO NOT manufacture image information from outlet_id,
    product_id, hashes, or arbitrary numeric identifiers.
    """
    n = str(name).strip().lower()

    if is_forbidden_column(n):
        return False

    visual_keywords = (
        "image_",
        "image",
        "vision_",
        "vision",
        "visual_",
        "visual",
        "embedding_",
        "embedding",
        "scene_",
        "scene",
        "cv_",
        "cv_",
        "computer_vision",
    )

    return any(k in n for k in visual_keywords)


# ============================================================================
# CONTEXT DISCOVERY
# ============================================================================

def discover_numeric_context_columns(
    df: pd.DataFrame,
) -> List[str]:
    """
    Discover safe numeric forecast-time context.

    IDs and target-derived fields are excluded.
    """
    result = []

    excluded = {
        "date",
        "outlet_id",
        "product_id",
        TARGET,
    }

    for column in df.columns:
        if column in excluded:
            continue

        if is_forbidden_column(column):
            continue

        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        if is_known_context_column(column):
            result.append(column)

    return sorted(result)


def discover_text_context_columns(
    df: pd.DataFrame,
) -> List[str]:
    """
    Discover safe textual forecast-time context.
    """
    result = []

    excluded = {
        "date",
        "outlet_id",
        "product_id",
        TARGET,
    }

    for column in df.columns:
        if column in excluded:
            continue

        if is_forbidden_column(column):
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            continue

        if is_known_context_column(column):
            result.append(column)

    return sorted(result)


def discover_visual_columns(
    df: pd.DataFrame,
) -> List[str]:
    result = []

    excluded = {
        "date",
        "outlet_id",
        "product_id",
        TARGET,
    }

    for column in df.columns:
        if column in excluded:
            continue

        if is_visual_column(column):
            result.append(column)

    return sorted(result)


# ============================================================================
# SERIES SELECTION
# ============================================================================

def select_series(
    df: pd.DataFrame,
    max_series: int = MAX_SERIES,
) -> List[Tuple[str, str]]:
    """
    Select deterministic, sufficiently long series.

    Selection is based on history availability, not future test performance.
    """
    counts = (
        df.groupby(
            ["outlet_id", "product_id"],
            sort=False,
            observed=True,
        )
        .size()
        .reset_index(name="rows")
    )

    counts = counts.sort_values(
        ["rows", "outlet_id", "product_id"],
        ascending=[False, True, True],
        kind="stable",
    )

    selected = []

    for row in counts.itertuples(index=False):
        selected.append(
            (
                str(row.outlet_id),
                str(row.product_id),
            )
        )

        if len(selected) >= max_series:
            break

    return selected


def compact_to_series(
    df: pd.DataFrame,
    selected_series: List[Tuple[str, str]],
) -> pd.DataFrame:
    """
    Vectorized filtering.
    """
    keys = pd.MultiIndex.from_arrays(
        [
            df["outlet_id"].astype(str).to_numpy(),
            df["product_id"].astype(str).to_numpy(),
        ]
    )

    selected_index = pd.MultiIndex.from_tuples(
        selected_series,
        names=["outlet_id", "product_id"],
    )

    mask = keys.isin(selected_index)

    out = df.loc[mask].copy()

    out["outlet_id"] = out["outlet_id"].astype(str)
    out["product_id"] = out["product_id"].astype(str)

    return out


# ============================================================================
# HISTORICAL DEMAND FEATURES
# ============================================================================

def build_historical_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build strictly historical demand features.

    Every lag/rolling value uses only observations before the
    current forecast date.

    IMPORTANT:
    The target itself is shifted before rolling calculations.
    """
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["outlet_id", "product_id", "date"],
        kind="stable",
    ).reset_index(drop=True)

    group_cols = ["outlet_id", "product_id"]

    grouped = df.groupby(
        group_cols,
        sort=False,
        observed=True,
    )[TARGET]

    # ------------------------------------------------------------------
    # LAGS
    # ------------------------------------------------------------------

    for lag in (1, 2, 3, 7, 14, 21, 28):
        df[f"lag_{lag}"] = grouped.shift(lag)

    # ------------------------------------------------------------------
    # ROLLING FEATURES
    #
    # Shift by 1 FIRST.
    # ------------------------------------------------------------------

    shifted = grouped.shift(1)

    for window in (3, 7, 14, 28):
        rolling_mean = (
            shifted.groupby(
                [
                    df["outlet_id"],
                    df["product_id"],
                ],
                sort=False,
            )
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

        rolling_std = (
            shifted.groupby(
                [
                    df["outlet_id"],
                    df["product_id"],
                ],
                sort=False,
            )
            .rolling(window=window, min_periods=2)
            .std()
            .reset_index(level=[0, 1], drop=True)
        )

        rolling_max = (
            shifted.groupby(
                [
                    df["outlet_id"],
                    df["product_id"],
                ],
                sort=False,
            )
            .rolling(window=window, min_periods=1)
            .max()
            .reset_index(level=[0, 1], drop=True)
        )

        df[f"rolling_mean_{window}"] = (
            rolling_mean.to_numpy()
        )

        df[f"rolling_std_{window}"] = (
            rolling_std.to_numpy()
        )

        df[f"rolling_max_{window}"] = (
            rolling_max.to_numpy()
        )

    # ------------------------------------------------------------------
    # TREND / VOLATILITY
    # ------------------------------------------------------------------

    df["trend_7_28"] = (
        df["rolling_mean_7"]
        / df["rolling_mean_28"].replace(0, np.nan)
    )

    df["trend_3_14"] = (
        df["rolling_mean_3"]
        / df["rolling_mean_14"].replace(0, np.nan)
    )

    df["volatility_7"] = (
        df["rolling_std_7"]
        / df["rolling_mean_7"].replace(0, np.nan)
    )

    df["volatility_28"] = (
        df["rolling_std_28"]
        / df["rolling_mean_28"].replace(0, np.nan)
    )

    # ------------------------------------------------------------------
    # CALENDAR
    # ------------------------------------------------------------------

    df["calendar_dow"] = df["date"].dt.dayofweek.astype(np.int8)
    df["calendar_month"] = df["date"].dt.month.astype(np.int8)
    df["calendar_day"] = df["date"].dt.day.astype(np.int8)
    df["calendar_week"] = (
        df["date"].dt.isocalendar().week.astype(np.int16)
    )
    df["calendar_quarter"] = (
        df["date"].dt.quarter.astype(np.int8)
    )
    df["calendar_day_of_year"] = (
        df["date"].dt.dayofyear.astype(np.int16)
    )
    df["calendar_weekend"] = (
        df["calendar_dow"] >= 5
    ).astype(np.int8)

    # ------------------------------------------------------------------
    # CYCLICAL CALENDAR FEATURES
    # ------------------------------------------------------------------

    df["dow_sin"] = np.sin(
        2.0 * np.pi * df["calendar_dow"] / 7.0
    ).astype(np.float32)

    df["dow_cos"] = np.cos(
        2.0 * np.pi * df["calendar_dow"] / 7.0
    ).astype(np.float32)

    df["month_sin"] = np.sin(
        2.0 * np.pi * df["calendar_month"] / 12.0
    ).astype(np.float32)

    df["month_cos"] = np.cos(
        2.0 * np.pi * df["calendar_month"] / 12.0
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # CLEAN NUMERIC FEATURES
    # ------------------------------------------------------------------

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    for column in numeric_cols:
        if column == TARGET:
            continue

        df[column] = (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )

    return df


# ============================================================================
# VISUAL FEATURES
# ============================================================================

def build_visual_features(
    df: pd.DataFrame,
    visual_columns: List[str],
) -> pd.DataFrame:
    """
    Convert genuine visual columns into numeric features.

    No synthetic image proxies are generated.
    """
    if not visual_columns:
        return pd.DataFrame(index=df.index)

    result = pd.DataFrame(index=df.index)

    for column in visual_columns:
        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        result[f"visual_{column}"] = (
            series
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )

    return result


# ============================================================================
# TEXT FEATURES
# ============================================================================

def build_text_features(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_columns: List[str],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Fit text representation on TRAIN only.

    Validation/test are transformed, never fitted.
    """
    if not text_columns:
        return (
            pd.DataFrame(index=train_df.index),
            pd.DataFrame(index=validation_df.index),
            pd.DataFrame(index=test_df.index),
        )

    def make_text(frame: pd.DataFrame) -> pd.Series:
        pieces = []

        for column in text_columns:
            pieces.append(
                frame[column]
                .fillna("")
                .astype(str)
            )

        if not pieces:
            return pd.Series(
                "",
                index=frame.index,
            )

        text = pieces[0].copy()

        for piece in pieces[1:]:
            text = text + " " + piece

        return text

    train_text = make_text(train_df)
    validation_text = make_text(validation_df)
    test_text = make_text(test_df)

    # Avoid TF-IDF failure when all text is empty.
    if train_text.str.strip().eq("").all():
        return (
            pd.DataFrame(index=train_df.index),
            pd.DataFrame(index=validation_df.index),
            pd.DataFrame(index=test_df.index),
        )

    vectorizer = TfidfVectorizer(
        max_features=64,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
    )

    try:
        train_matrix = vectorizer.fit_transform(
            train_text
        )

        validation_matrix = vectorizer.transform(
            validation_text
        )

        test_matrix = vectorizer.transform(
            test_text
        )
    except ValueError:
        return (
            pd.DataFrame(index=train_df.index),
            pd.DataFrame(index=validation_df.index),
            pd.DataFrame(index=test_df.index),
        )

    # Small text corpus: TF-IDF itself is enough.
    if train_matrix.shape[1] <= 1:
        return (
            pd.DataFrame(
                train_matrix.toarray(),
                index=train_df.index,
                columns=["text_0"],
                dtype=np.float32,
            ),
            pd.DataFrame(
                validation_matrix.toarray(),
                index=validation_df.index,
                columns=["text_0"],
                dtype=np.float32,
            ),
            pd.DataFrame(
                test_matrix.toarray(),
                index=test_df.index,
                columns=["text_0"],
                dtype=np.float32,
            ),
        )

    n_components = min(
        8,
        train_matrix.shape[1] - 1,
        max(1, train_matrix.shape[0] - 1),
    )

    try:
        svd = TruncatedSVD(
            n_components=n_components,
            random_state=RANDOM_STATE,
        )

        train_matrix = svd.fit_transform(
            train_matrix
        )

        validation_matrix = svd.transform(
            validation_matrix
        )

        test_matrix = svd.transform(
            test_matrix
        )

    except Exception:
        # Fall back to raw TF-IDF if SVD is not numerically useful.
        train_matrix = train_matrix.toarray()
        validation_matrix = validation_matrix.toarray()
        test_matrix = test_matrix.toarray()

        n_features = min(
            train_matrix.shape[1],
            8,
        )

        train_matrix = train_matrix[:, :n_features]
        validation_matrix = validation_matrix[:, :n_features]
        test_matrix = test_matrix[:, :n_features]

    columns = [
        f"text_{i}"
        for i in range(train_matrix.shape[1])
    ]

    train_features = pd.DataFrame(
        np.asarray(train_matrix, dtype=np.float32),
        index=train_df.index,
        columns=columns,
    )

    validation_features = pd.DataFrame(
        np.asarray(validation_matrix, dtype=np.float32),
        index=validation_df.index,
        columns=columns,
    )

    test_features = pd.DataFrame(
        np.asarray(test_matrix, dtype=np.float32),
        index=test_df.index,
        columns=columns,
    )

    return (
        train_features,
        validation_features,
        test_features,
    )


# ============================================================================
# TEMPORAL SPLITS
# ============================================================================

def build_temporal_splits(
    df: pd.DataFrame,
    validation_days: int = VALIDATION_DAYS,
    test_days: int = TEST_DAYS,
):
    """
    Build common chronological train/validation/test split.

    The last test_days are completely unseen.
    """
    dates = np.sort(
        df["date"].dropna().unique()
    )

    if len(dates) <= validation_days + test_days:
        raise ValueError(
            "Not enough dates for chronological split."
        )

    test_dates = dates[-test_days:]

    validation_dates = dates[
        -(validation_days + test_days):-test_days
    ]

    train_dates = dates[
        :-(validation_days + test_days)
    ]

    train = df[
        df["date"].isin(train_dates)
    ].copy()

    validation = df[
        df["date"].isin(validation_dates)
    ].copy()

    test = df[
        df["date"].isin(test_dates)
    ].copy()

    return train, validation, test


# ============================================================================
# BACKWARD-COMPATIBILITY SPLIT
# ============================================================================

def build_series_splits(
    series_df: pd.DataFrame,
    validation_days: int = VALIDATION_DAYS,
    test_days: int = TEST_DAYS,
):
    """
    Compatibility helper used by the Part 27E tests.

    Returns:
        train, validation, test
    """
    frame = series_df.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    return build_temporal_splits(
        frame,
        validation_days=validation_days,
        test_days=test_days,
    )


# ============================================================================
# MODEL
# ============================================================================

def create_model() -> LGBMRegressor:
    """
    Small CPU-friendly global LightGBM model.
    """
    return LGBMRegressor(
        objective="regression",
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.10,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1,
    )


# ============================================================================
# FEATURE MATRIX
# ============================================================================

BASE_FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",

    "rolling_mean_3",
    "rolling_std_3",
    "rolling_max_3",

    "rolling_mean_7",
    "rolling_std_7",
    "rolling_max_7",

    "rolling_mean_14",
    "rolling_std_14",
    "rolling_max_14",

    "rolling_mean_28",
    "rolling_std_28",
    "rolling_max_28",

    "trend_7_28",
    "trend_3_14",
    "volatility_7",
    "volatility_28",

    "calendar_dow",
    "calendar_month",
    "calendar_day",
    "calendar_week",
    "calendar_quarter",
    "calendar_day_of_year",
    "calendar_weekend",

    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]


def safe_feature_columns(
    df: pd.DataFrame,
    candidate_columns: List[str],
) -> List[str]:
    """
    Final feature firewall.
    """
    result = []

    for column in candidate_columns:
        if column not in df.columns:
            continue

        if is_forbidden_column(column):
            continue

        if column in {
            "date",
            "outlet_id",
            "product_id",
            TARGET,
        }:
            continue

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            continue

        result.append(column)

    return result


# ============================================================================
# METRICS
# ============================================================================

def mae(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> float:
    return float(
        np.mean(
            np.abs(actual - prediction)
        )
    )


def rmse(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    actual - prediction
                )
            )
        )
    )


def smape(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> float:
    denominator = (
        np.abs(actual)
        + np.abs(prediction)
        + 1e-8
    )

    return float(
        np.mean(
            2.0
            * np.abs(actual - prediction)
            / denominator
        )
        * 100.0
    )


def bias(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> float:
    return float(
        np.mean(prediction - actual)
    )


def high_demand_mae(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> float:
    if len(actual) == 0:
        return 0.0

    threshold = np.quantile(
        actual,
        0.80,
    )

    mask = actual >= threshold

    if not np.any(mask):
        return 0.0

    return mae(
        actual[mask],
        prediction[mask],
    )


def spike_recall(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """
    Simple relative-spike recall.

    Spike threshold is determined from actual test data only
    for diagnostic reporting. It is NOT a model feature.
    """
    if len(actual) < 2:
        return 0.0

    threshold = np.quantile(
        actual,
        0.90,
    )

    actual_spike = actual >= threshold
    predicted_spike = prediction >= threshold

    positives = np.sum(actual_spike)

    if positives == 0:
        return 0.0

    return float(
        np.sum(
            actual_spike & predicted_spike
        )
        / positives
    )


def evaluate_predictions(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:
    return {
        "mae": mae(actual, prediction),
        "rmse": rmse(actual, prediction),
        "smape": smape(actual, prediction),
        "bias": bias(actual, prediction),
        "high_demand_mae": high_demand_mae(
            actual,
            prediction,
        ),
        "spike_recall": spike_recall(
            actual,
            prediction,
        ),
    }


# ============================================================================
# HISTORY BASELINE
# ============================================================================

def history_prediction(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Conservative historical baseline.

    Uses lag_1 where available.

    Missing history falls back to rolling_mean_7,
    then rolling_mean_28,
    then zero.
    """
    def predict(frame: pd.DataFrame):
        prediction = frame["lag_1"].to_numpy(
            dtype=np.float32
        )

        fallback_7 = frame[
            "rolling_mean_7"
        ].to_numpy(dtype=np.float32)

        fallback_28 = frame[
            "rolling_mean_28"
        ].to_numpy(dtype=np.float32)

        prediction = np.where(
            np.isfinite(prediction)
            & (prediction >= 0),
            prediction,
            fallback_7,
        )

        prediction = np.where(
            np.isfinite(prediction)
            & (prediction >= 0),
            prediction,
            fallback_28,
        )

        prediction = np.nan_to_num(
            prediction,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return np.maximum(
            prediction,
            0.0,
        ).astype(np.float32)

    return (
        predict(validation),
        predict(test),
    )


# ============================================================================
# MODE FEATURE BUILDING
# ============================================================================

def build_mode_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    mode: str,
    numeric_context: List[str],
    text_context: List[str],
    visual_columns: List[str],
):
    """
    Build features for exactly one benchmark mode.
    """
    train_parts = []
    validation_parts = []
    test_parts = []

    # ------------------------------------------------------------------
    # Historical features
    # ------------------------------------------------------------------

    history_features = safe_feature_columns(
        train,
        BASE_FEATURES,
    )

    if mode in {
        "history",
        "numeric",
        "numeric_text",
        "numeric_visual",
        "multimodal",
    }:
        train_parts.append(
            train[history_features]
        )

        validation_parts.append(
            validation[history_features]
        )

        test_parts.append(
            test[history_features]
        )

    # ------------------------------------------------------------------
    # Numeric context
    # ------------------------------------------------------------------

    if mode in {
        "numeric",
        "numeric_text",
        "numeric_visual",
        "multimodal",
    }:
        numeric_features = safe_feature_columns(
            train,
            numeric_context,
        )

        if numeric_features:
            train_parts.append(
                train[numeric_features]
            )

            validation_parts.append(
                validation[numeric_features]
            )

            test_parts.append(
                test[numeric_features]
            )

    # ------------------------------------------------------------------
    # Text context
    # ------------------------------------------------------------------

    if mode in {
        "numeric_text",
        "multimodal",
    }:
        (
            train_text,
            validation_text,
            test_text,
        ) = build_text_features(
            train,
            validation,
            test,
            text_context,
        )

        if not train_text.empty:
            train_parts.append(
                train_text
            )

            validation_parts.append(
                validation_text
            )

            test_parts.append(
                test_text
            )

    # ------------------------------------------------------------------
    # Genuine visual features
    # ------------------------------------------------------------------

    if mode in {
        "numeric_visual",
        "multimodal",
    }:
        train_visual = build_visual_features(
            train,
            visual_columns,
        )

        validation_visual = build_visual_features(
            validation,
            visual_columns,
        )

        test_visual = build_visual_features(
            test,
            visual_columns,
        )

        if not train_visual.empty:
            train_parts.append(
                train_visual
            )

            validation_parts.append(
                validation_visual
            )

            test_parts.append(
                test_visual
            )

    # ------------------------------------------------------------------
    # Concatenate
    # ------------------------------------------------------------------

    if not train_parts:
        raise ValueError(
            f"No usable features for mode={mode}"
        )

    train_matrix = pd.concat(
        train_parts,
        axis=1,
    )

    validation_matrix = pd.concat(
        validation_parts,
        axis=1,
    )

    test_matrix = pd.concat(
        test_parts,
        axis=1,
    )

    # Remove duplicate columns.
    train_matrix = train_matrix.loc[
        :,
        ~train_matrix.columns.duplicated(),
    ]

    validation_matrix = validation_matrix.loc[
        :,
        ~validation_matrix.columns.duplicated(),
    ]

    test_matrix = test_matrix.loc[
        :,
        ~test_matrix.columns.duplicated(),
    ]

    # Final firewall.
    allowed = [
        c
        for c in train_matrix.columns
        if not is_forbidden_column(c)
    ]

    train_matrix = train_matrix[allowed]
    validation_matrix = validation_matrix[allowed]
    test_matrix = test_matrix[allowed]

    # float32 keeps memory low.
    train_matrix = (
        train_matrix
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )

    validation_matrix = (
        validation_matrix
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )

    test_matrix = (
        test_matrix
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )

    return (
        train_matrix,
        validation_matrix,
        test_matrix,
    )


# ============================================================================
# SINGLE MODE BENCHMARK
# ============================================================================

def run_model_mode(
    mode: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    numeric_context: List[str],
    text_context: List[str],
    visual_columns: List[str],
):
    section(f"RUNNING: {mode}")

    train_features, validation_features, test_features = (
        build_mode_features(
            train,
            validation,
            test,
            mode,
            numeric_context,
            text_context,
            visual_columns,
        )
    )

    target_train = (
        train[TARGET]
        .to_numpy(dtype=np.float32)
    )

    target_validation = (
        validation[TARGET]
        .to_numpy(dtype=np.float32)
    )

    target_test = (
        test[TARGET]
        .to_numpy(dtype=np.float32)
    )

    # --------------------------------------------------------------
    # Train
    # --------------------------------------------------------------

    model = create_model()

    model.fit(
        train_features,
        target_train,
    )

    # --------------------------------------------------------------
    # Predict
    # --------------------------------------------------------------

    validation_prediction = model.predict(
        validation_features
    )

    test_prediction = model.predict(
        test_features
    )

    validation_prediction = np.maximum(
        np.nan_to_num(
            validation_prediction,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
    )

    test_prediction = np.maximum(
        np.nan_to_num(
            test_prediction,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
    )

    validation_metrics = evaluate_predictions(
        target_validation,
        validation_prediction,
    )

    test_metrics = evaluate_predictions(
        target_test,
        test_prediction,
    )

    print(
        f"Validation MAE: "
        f"{validation_metrics['mae']:.6f}"
    )

    print(
        f"Test MAE: "
        f"{test_metrics['mae']:.6f}"
    )

    print(
        f"Test RMSE: "
        f"{test_metrics['rmse']:.6f}"
    )

    result = {
        "mode": mode,

        "validation_mae": validation_metrics["mae"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_smape": validation_metrics["smape"],
        "validation_bias": validation_metrics["bias"],

        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_smape": test_metrics["smape"],
        "test_bias": test_metrics["bias"],

        "high_demand_mae": test_metrics[
            "high_demand_mae"
        ],

        "spike_recall": test_metrics[
            "spike_recall"
        ],

        "feature_count": int(
            train_features.shape[1]
        ),
    }

    # Explicitly release memory.
    del (
        train_features,
        validation_features,
        test_features,
        model,
        validation_prediction,
        test_prediction,
    )

    gc.collect()

    return result


# ============================================================================
# VALIDATION
# ============================================================================

def validate_results(
    results_df: pd.DataFrame,
    numeric_context: List[str],
    text_context: List[str],
    visual_columns: List[str],
) -> Dict:
    """
    Final Part 27E acceptance validation.

    This function NEVER accesses a missing column before checking
    that it exists. This fixes the previous V3/V5 validation bug.
    """
    errors = []

    required_columns = {
        "mode",
        "validation_mae",
        "validation_rmse",
        "test_mae",
        "test_rmse",
    }

    missing = required_columns - set(
        results_df.columns
    )

    if missing:
        errors.append(
            f"Missing result columns: {sorted(missing)}"
        )

    if results_df.empty:
        errors.append(
            "Benchmark produced no results."
        )

    expected_modes = {
        "history",
        "numeric",
        "numeric_text",
        "numeric_visual",
        "multimodal",
    }

    if "mode" in results_df.columns:
        actual_modes = set(
            results_df["mode"].astype(str)
        )

        missing_modes = (
            expected_modes - actual_modes
        )

        if missing_modes:
            errors.append(
                f"Missing benchmark modes: "
                f"{sorted(missing_modes)}"
            )

    metric_columns = [
        c
        for c in [
            "validation_mae",
            "validation_rmse",
            "test_mae",
            "test_rmse",
            "test_smape",
            "test_bias",
        ]
        if c in results_df.columns
    ]

    for column in metric_columns:
        values = pd.to_numeric(
            results_df[column],
            errors="coerce",
        ).to_numpy()

        if not np.isfinite(values).all():
            errors.append(
                f"Non-finite values found in {column}"
            )

    # --------------------------------------------------------------
    # Firewall validation
    # --------------------------------------------------------------

    discovered_feature_names = (
        list(BASE_FEATURES)
        + list(numeric_context)
    )

    for column in discovered_feature_names:
        if is_forbidden_column(column):
            errors.append(
                f"Forecast-time firewall violation: {column}"
            )

    for column in text_context:
        if is_forbidden_column(column):
            errors.append(
                f"Text firewall violation: {column}"
            )

    for column in visual_columns:
        if is_forbidden_column(column):
            errors.append(
                f"Visual firewall violation: {column}"
            )

    return {
        "passed": len(errors) == 0,
        "errors": errors,
    }


# ============================================================================
# DECISION LOGIC
# ============================================================================

def determine_decision(
    results_df: pd.DataFrame,
) -> Tuple[str, str]:
    """
    Production decision.

    Multimodal must demonstrate actual unseen-test value.

    No improvement => keep optional.
    """
    if results_df.empty:
        return (
            "REJECT",
            "No benchmark results.",
        )

    result_lookup = (
        results_df
        .set_index("mode")
    )

    if (
        "multimodal" not in result_lookup.index
    ):
        return (
            "REJECT",
            "Multimodal result missing.",
        )

    if (
        "history" not in result_lookup.index
    ):
        return (
            "REJECT",
            "History baseline missing.",
        )

    multimodal_mae = float(
        result_lookup.loc[
            "multimodal",
            "test_mae",
        ]
    )

    history_mae = float(
        result_lookup.loc[
            "history",
            "test_mae",
        ]
    )

    improvement = (
        history_mae - multimodal_mae
    ) / max(history_mae, 1e-8) * 100.0

    if improvement >= 2.0:
        return (
            "MULTIMODAL_CANDIDATE",
            (
                "Multimodal achieved at least "
                "2% unseen-test MAE improvement "
                "over the history baseline."
            ),
        )

    return (
        "KEEP_MULTIMODAL_OPTIONAL",
        (
            "Multimodal did not demonstrate "
            "meaningful unseen-test improvement."
        ),
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    start_time = time.time()

    banner(
        "PART 27E V5 - GENUINE LOW-MEMORY MULTIMODAL BENCHMARK"
    )

    # --------------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    # --------------------------------------------------------------
    # Load only required columns.
    #
    # V5 deliberately does NOT load the whole 949k-row wide dataset.
    # --------------------------------------------------------------

    optional_columns = [
        # Calendar/context
        "holiday_active",
        "promotion_active",
        "event_active",
        "temperature",
        "rainfall",
        "weather",
        "tourism",
        "tourist",
        "students",
        "student_population",
        "cultural_context_score",
        "cultural_demand_pressure",
        "religious_context",

        # Potential genuine text
        "holiday_name",
        "event_name",
        "promotion_name",
        "weather_description",
        "context_text",

        # Potential genuine visual/CV features
        "image_embedding",
        "image_feature",
        "vision_embedding",
        "visual_embedding",
        "scene_embedding",
        "cv_embedding",
    ]

    # Read header first.
    header = pd.read_csv(
        DATA_PATH,
        nrows=0,
    )

    available_columns = set(
        header.columns
    )

    load_columns = []

    for column in (
        REQUIRED_COLUMNS
        + optional_columns
    ):
        if column in available_columns:
            load_columns.append(column)

    print(
        f"Loading {len(load_columns)} columns..."
    )

    df = pd.read_csv(
        DATA_PATH,
        usecols=load_columns,
        low_memory=True,
    )

    ensure_columns(
        df,
        REQUIRED_COLUMNS,
    )

    print(
        f"Source rows: {len(df):,}"
    )

    # --------------------------------------------------------------
    # Basic normalization
    # --------------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "date",
            "outlet_id",
            "product_id",
            TARGET,
        ]
    )

    df["outlet_id"] = (
        df["outlet_id"]
        .astype(str)
    )

    df["product_id"] = (
        df["product_id"]
        .astype(str)
    )

    df[TARGET] = (
        pd.to_numeric(
            df[TARGET],
            errors="coerce",
        )
        .fillna(0.0)
        .astype(np.float32)
    )

    # --------------------------------------------------------------
    # Series selection
    # --------------------------------------------------------------

    section(
        "Selecting low-memory benchmark series..."
    )

    selected_series = select_series(
        df,
        max_series=MAX_SERIES,
    )

    print(
        f"Selected series: "
        f"{len(selected_series)}"
    )

    df = compact_to_series(
        df,
        selected_series,
    )

    print(
        f"Compact rows: "
        f"{len(df):,}"
    )

    # Release source references where possible.
    gc.collect()

    # --------------------------------------------------------------
    # Historical features
    # --------------------------------------------------------------

    section(
        "Building historical features..."
    )

    df = build_historical_features(
        df
    )

    # --------------------------------------------------------------
    # Modality discovery
    # --------------------------------------------------------------

    section(
        "MODALITY DISCOVERY"
    )

    numeric_context = (
        discover_numeric_context_columns(
            df
        )
    )

    text_context = (
        discover_text_context_columns(
            df
        )
    )

    visual_columns = (
        discover_visual_columns(
            df
        )
    )

    print(
        f"Numeric context: "
        f"{numeric_context}"
    )

    print(
        f"Text context: "
        f"{text_context}"
    )

    print(
        f"Genuine visual features: "
        f"{visual_columns}"
    )

    if not visual_columns:
        print(
            "NOTE: No genuine visual features "
            "exist in the current dataset."
        )

    if not text_context:
        print(
            "NOTE: No genuine text context "
            "exists in the current dataset."
        )

    # --------------------------------------------------------------
    # Firewall
    # --------------------------------------------------------------

    firewall_errors = []

    for column in (
        BASE_FEATURES
        + numeric_context
        + text_context
        + visual_columns
    ):
        if is_forbidden_column(column):
            firewall_errors.append(column)

    if firewall_errors:
        raise RuntimeError(
            "FORECAST-TIME FIREWALL FAILED:\n"
            + "\n".join(
                f"  - {x}"
                for x in firewall_errors
            )
        )

    print(
        "\nFORECAST-TIME FIREWALL: PASS"
    )

    # --------------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------------

    section(
        "CHRONOLOGICAL SPLIT"
    )

    train, validation, test = (
        build_temporal_splits(
            df,
            validation_days=VALIDATION_DAYS,
            test_days=TEST_DAYS,
        )
    )

    print(
        f"Train:      {len(train):,}"
    )

    print(
        f"Validation: {len(validation):,}"
    )

    print(
        f"Test:       {len(test):,}"
    )

    print(
        f"Train end: "
        f"{train['date'].max().date()}"
    )

    print(
        f"Validation: "
        f"{validation['date'].min().date()} "
        f"→ "
        f"{validation['date'].max().date()}"
    )

    print(
        f"Test:       "
        f"{test['date'].min().date()} "
        f"→ "
        f"{test['date'].max().date()}"
    )

    # --------------------------------------------------------------
    # Baseline
    # --------------------------------------------------------------

    results = []

    validation_history, test_history = (
        history_prediction(
            train,
            validation,
            test,
        )
    )

    validation_actual = (
        validation[TARGET]
        .to_numpy(dtype=np.float32)
    )

    test_actual = (
        test[TARGET]
        .to_numpy(dtype=np.float32)
    )

    history_validation_metrics = (
        evaluate_predictions(
            validation_actual,
            validation_history,
        )
    )

    history_test_metrics = (
        evaluate_predictions(
            test_actual,
            test_history,
        )
    )

    section(
        "RUNNING: history"
    )

    print(
        f"Validation MAE: "
        f"{history_validation_metrics['mae']:.6f}"
    )

    print(
        f"Test MAE: "
        f"{history_test_metrics['mae']:.6f}"
    )

    print(
        f"Test RMSE: "
        f"{history_test_metrics['rmse']:.6f}"
    )

    results.append(
        {
            "mode": "history",

            "validation_mae":
                history_validation_metrics["mae"],

            "validation_rmse":
                history_validation_metrics["rmse"],

            "validation_smape":
                history_validation_metrics["smape"],

            "validation_bias":
                history_validation_metrics["bias"],

            "test_mae":
                history_test_metrics["mae"],

            "test_rmse":
                history_test_metrics["rmse"],

            "test_smape":
                history_test_metrics["smape"],

            "test_bias":
                history_test_metrics["bias"],

            "high_demand_mae":
                history_test_metrics[
                    "high_demand_mae"
                ],

            "spike_recall":
                history_test_metrics[
                    "spike_recall"
                ],

            "feature_count": 0,
        }
    )

    del (
        validation_history,
        test_history,
    )

    gc.collect()

    # --------------------------------------------------------------
    # ML modes
    # --------------------------------------------------------------

    modes = [
        "numeric",
        "numeric_text",
        "numeric_visual",
        "multimodal",
    ]

    for mode in modes:
        result = run_model_mode(
            mode,
            train,
            validation,
            test,
            numeric_context,
            text_context,
            visual_columns,
        )

        results.append(result)

        gc.collect()

    # --------------------------------------------------------------
    # Results
    # --------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------------
    # Validation winner
    # --------------------------------------------------------------

    validation_winner = (
        results_df.loc[
            results_df["validation_mae"].idxmin(),
            "mode",
        ]
    )

    test_winner = (
        results_df.loc[
            results_df["test_mae"].idxmin(),
            "mode",
        ]
    )

    # --------------------------------------------------------------
    # Multimodal improvement vs history
    # --------------------------------------------------------------

    history_mae = float(
        results_df.loc[
            results_df["mode"] == "history",
            "test_mae",
        ].iloc[0]
    )

    multimodal_mae = float(
        results_df.loc[
            results_df["mode"] == "multimodal",
            "test_mae",
        ].iloc[0]
    )

    multimodal_improvement = (
        (
            history_mae
            - multimodal_mae
        )
        / max(history_mae, 1e-8)
        * 100.0
    )

    # --------------------------------------------------------------
    # Validation/test stability
    # --------------------------------------------------------------

    validation_rank = (
        results_df
        .sort_values(
            "validation_mae"
        )["mode"]
        .tolist()
    )

    test_rank = (
        results_df
        .sort_values(
            "test_mae"
        )["mode"]
        .tolist()
    )

    top1_stable = (
        validation_rank[0]
        == test_rank[0]
    )

    # --------------------------------------------------------------
    # Final validation
    # --------------------------------------------------------------

    validation_result = validate_results(
        results_df,
        numeric_context,
        text_context,
        visual_columns,
    )

    # --------------------------------------------------------------
    # Decision
    # --------------------------------------------------------------

    decision, reason = (
        determine_decision(
            results_df
        )
    )

    # --------------------------------------------------------------
    # Save compact results
    # --------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    runtime = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------------
    # Final report
    # --------------------------------------------------------------

    banner(
        "PART 27E V5 RESULTS"
    )

    print(
        results_df[
            [
                "mode",
                "validation_mae",
                "test_mae",
                "test_rmse",
                "test_smape",
                "test_bias",
                "high_demand_mae",
                "spike_recall",
                "feature_count",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nVALIDATION WINNER: "
        f"{validation_winner}"
    )

    print(
        "UNSEEN TEST WINNER: "
        f"{test_winner}"
    )

    print(
        "Multimodal vs history: "
        f"{multimodal_improvement:.4f}%"
    )

    print(
        "Validation/test top-1 stable: "
        f"{top1_stable}"
    )

    print(
        "Selected series: "
        f"{len(selected_series)}"
    )

    print(
        "Test rows: "
        f"{len(test):,}"
    )

    print(
        "Safe numeric context: "
        f"{len(numeric_context)}"
    )

    print(
        "Safe text context: "
        f"{len(text_context)}"
    )

    print(
        "Genuine visual features: "
        f"{len(visual_columns)}"
    )

    print(
        f"Runtime: {runtime:.2f}s"
    )

    print(
        "\nDECISION: "
        f"{decision}"
    )

    print(
        "REASON: "
        f"{reason}"
    )

    print(
        "\nACCEPTANCE"
    )

    print(
        "1. Forecast-time firewall: "
        + (
            "PASS"
            if not firewall_errors
            else "FAIL"
        )
    )

    print(
        "2. Context discovery: "
        + (
            "PASS"
            if validation_result["passed"]
            else "FAIL"
        )
    )

    print(
        "3. Five benchmark modes: "
        + (
            "PASS"
            if set(results_df["mode"])
            == {
                "history",
                "numeric",
                "numeric_text",
                "numeric_visual",
                "multimodal",
            }
            else "FAIL"
        )
    )

    print(
        "4. Finite metrics: "
        + (
            "PASS"
            if validation_result["passed"]
            else "FAIL"
        )
    )

    print(
        "5. Aggregate benchmark validation: "
        + (
            "PASS"
            if validation_result["passed"]
            else "FAIL"
        )
    )

    if validation_result["errors"]:
        print(
            "\nVALIDATION ERRORS:"
        )

        for error in validation_result["errors"]:
            print(
                f"  - {error}"
            )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_PATH
    )

    if validation_result["passed"]:
        print(
            "\nPART 27E V5 ACCEPTANCE: PASS"
        )
    else:
        print(
            "\nPART 27E V5 ACCEPTANCE: FAIL"
        )

    # --------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------

    del (
        df,
        train,
        validation,
        test,
        results_df,
    )

    gc.collect()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()