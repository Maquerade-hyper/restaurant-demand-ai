from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# ============================================================================
# PART 27E
# MULTIMODAL INTELLIGENCE BENCHMARK
# ============================================================================

TARGET_COLUMNS = {
    "quantity_sold",
    "deconstrained_demand",
    "actual_demand",
    "true_demand",
    "future_demand",
    "prediction",
    "target",
}


FORBIDDEN_COLUMNS = {
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
    "censoring_class",
    "closing_stock",
    "inventory_position",
    "recommended_order",
    "stockout_risk",
}


FORBIDDEN_FRAGMENTS = (
    "future_",
    "_future",
    "actual_",
    "_actual",
    "truth",
    "_truth",
    "prediction",
    "_prediction",
    "lost_demand",
    "recovery",
    "censor",
    "stockout_event",
    "closing_stock",
    "inventory_position",
    "recommended_order",
)


CONTEXT_KEYWORDS = (
    "holiday",
    "festival",
    "religious",
    "calendar",
    "event",
    "promotion",
    "campaign",
    "weather",
    "temperature",
    "temp",
    "rain",
    "rainfall",
    "precipitation",
    "tourism",
    "tourist",
    "student",
    "demographic",
    "population",
    "cultural",
    "humidity",
    "wind",
)


STATIC_COLUMNS = {
    "outlet_type",
    "location_type",
    "country",
    "region",
    "city",
    "product_category",
    "unit",
    "service_mode",
    "outlet_segment",
}


VISUAL_KEYWORDS = (
    "image_",
    "vision_",
    "visual_",
    "cv_",
    "embedding_",
    "scene_",
    "image_embedding",
    "vision_embedding",
)


VALID_MODES = {
    "history",
    "numeric",
    "numeric_text",
    "numeric_visual",
    "multimodal",
}


# ============================================================================
# FIREWALL
# ============================================================================

def is_forbidden_feature(name: str) -> bool:
    name = str(name).strip().lower()

    if name in TARGET_COLUMNS:
        return True

    if name in FORBIDDEN_COLUMNS:
        return True

    return any(
        fragment in name
        for fragment in FORBIDDEN_FRAGMENTS
    )


def is_known_context_column(name: str) -> bool:
    name = str(name).strip().lower()

    if is_forbidden_feature(name):
        return False

    # Static categorical context is valid forecast-time information.
    if name in STATIC_COLUMNS:
        return True

    # Compatibility:
    # stockout itself may be discovered by the public discovery helper.
    # The model firewall separately decides whether it is safe to use.
    if name == "stockout":
        return True

    return any(
        keyword in name
        for keyword in CONTEXT_KEYWORDS
    )


def is_visual_column(name: str) -> bool:
    name = str(name).strip().lower()

    if is_forbidden_feature(name):
        return False

    return any(
        keyword in name
        for keyword in VISUAL_KEYWORDS
    )


def discover_context_columns(
    frame: pd.DataFrame,
) -> Tuple[List[str], List[str]]:

    numeric = []
    text = []

    for column in frame.columns:

        if column in {
            "date",
            "outlet_id",
            "product_id",
        }:
            continue

        if not is_known_context_column(column):
            continue

        if pd.api.types.is_numeric_dtype(
            frame[column]
        ):
            numeric.append(column)
        else:
            text.append(column)

    return numeric, text


def discover_visual_columns(
    frame: pd.DataFrame,
) -> List[str]:

    return [
        column
        for column in frame.columns
        if is_visual_column(column)
    ]


def audit_feature_firewall(
    columns: List[str],
) -> List[str]:

    return [
        column
        for column in columns
        if is_forbidden_feature(column)
    ]


# ============================================================================
# TEMPORAL FEATURES
# ============================================================================

def build_temporal_features(
    frame: pd.DataFrame,
    target: str,
) -> pd.DataFrame:

    if target not in frame.columns:
        raise ValueError(
            f"Target column not found: {target}"
        )

    required = {
        "date",
        "outlet_id",
        "product_id",
        target,
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = frame.copy()

    df["date"] = pd.to_datetime(
        df["date"],
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

    df[target] = pd.to_numeric(
        df[target],
        errors="coerce",
    )

    df = df.dropna(
        subset=[target]
    ).copy()

    df[target] = df[target].astype(
        "float32"
    )

    df = df.sort_values(
        [
            "outlet_id",
            "product_id",
            "date",
        ]
    ).reset_index(
        drop=True
    )

    group_keys = [
        "outlet_id",
        "product_id",
    ]

    # ------------------------------------------------------------------
    # Historical lags
    # ------------------------------------------------------------------

    for lag in (
        1,
        2,
        3,
        7,
        14,
        21,
        28,
    ):

        df[
            f"history_lag_{lag}"
        ] = (
            df.groupby(
                group_keys,
                sort=False,
            )[target]
            .shift(lag)
            .astype("float32")
        )

    # ------------------------------------------------------------------
    # Historical rolling statistics.
    #
    # IMPORTANT:
    # shift(1) happens BEFORE rolling.
    # Therefore the current target is never included.
    # ------------------------------------------------------------------

    shifted = (
        df.groupby(
            group_keys,
            sort=False,
        )[target]
        .shift(1)
    )

    shifted_frame = pd.DataFrame(
        {
            "outlet_id": df["outlet_id"].values,
            "product_id": df["product_id"].values,
            "_shifted": shifted.values,
        },
        index=df.index,
    )

    for window in (
        3,
        7,
        14,
        28,
    ):

        rolling_mean = (
            shifted_frame
            .groupby(
                group_keys,
                sort=False,
            )["_shifted"]
            .transform(
                lambda s: s.rolling(
                    window=window,
                    min_periods=2,
                ).mean()
            )
        )

        rolling_std = (
            shifted_frame
            .groupby(
                group_keys,
                sort=False,
            )["_shifted"]
            .transform(
                lambda s: s.rolling(
                    window=window,
                    min_periods=2,
                ).std()
            )
        )

        df[
            f"history_mean_{window}"
        ] = rolling_mean.astype(
            "float32"
        )

        df[
            f"history_std_{window}"
        ] = rolling_std.astype(
            "float32"
        )

    # ------------------------------------------------------------------
    # Historical dynamics
    # ------------------------------------------------------------------

    df["history_trend_7"] = (
        df["history_lag_1"]
        /
        (
            df["history_lag_7"]
            + 1e-6
        )
    ).astype("float32")

    df["history_trend_28"] = (
        df["history_lag_1"]
        /
        (
            df["history_lag_28"]
            + 1e-6
        )
    ).astype("float32")

    df["history_volatility_7"] = (
        df["history_std_7"]
        /
        (
            df["history_mean_7"]
            + 1e-6
        )
    ).astype("float32")

    df["history_volatility_28"] = (
        df["history_std_28"]
        /
        (
            df["history_mean_28"]
            + 1e-6
        )
    ).astype("float32")

    # ------------------------------------------------------------------
    # Calendar features
    # ------------------------------------------------------------------

    date = df["date"]

    df["calendar_dow"] = (
        date.dt.dayofweek
    ).astype("int8")

    df["calendar_month"] = (
        date.dt.month
    ).astype("int8")

    df["calendar_day"] = (
        date.dt.day
    ).astype("int8")

    df["calendar_week"] = (
        date.dt.isocalendar()
        .week
        .astype("int16")
    )

    df["calendar_quarter"] = (
        date.dt.quarter
    ).astype("int8")

    df["calendar_day_of_year"] = (
        date.dt.dayofyear
    ).astype("int16")

    df["calendar_weekend"] = (
        date.dt.dayofweek >= 5
    ).astype("int8")

    df["calendar_dow_sin"] = np.sin(
        2.0
        * np.pi
        * date.dt.dayofweek
        / 7.0
    ).astype("float32")

    df["calendar_dow_cos"] = np.cos(
        2.0
        * np.pi
        * date.dt.dayofweek
        / 7.0
    ).astype("float32")

    df["calendar_month_sin"] = np.sin(
        2.0
        * np.pi
        * date.dt.month
        / 12.0
    ).astype("float32")

    df["calendar_month_cos"] = np.cos(
        2.0
        * np.pi
        * date.dt.month
        / 12.0
    ).astype("float32")

    return df


# ============================================================================
# NUMERIC HELPERS
# ============================================================================

def _numeric(
    frame: pd.DataFrame,
    names: List[str],
) -> pd.Series:

    for name in names:

        if name in frame.columns:

            return (
                pd.to_numeric(
                    frame[name],
                    errors="coerce",
                )
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
                .fillna(0.0)
            )

    return pd.Series(
        0.0,
        index=frame.index,
        dtype="float32",
    )


# ============================================================================
# CONTEXT
# ============================================================================

def build_known_context(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    result = pd.DataFrame(
        index=frame.index
    )

    holiday = _numeric(
        frame,
        [
            "holiday_active",
            "is_holiday",
            "holiday",
        ],
    )

    promotion = _numeric(
        frame,
        [
            "promotion_active",
            "is_promotion",
            "promotion",
        ],
    )

    event = _numeric(
        frame,
        [
            "event_active",
            "is_event",
            "event",
        ],
    )

    religious = _numeric(
        frame,
        [
            "religious_holiday",
            "religious_active",
            "religious_event",
        ],
    )

    cultural = _numeric(
        frame,
        [
            "cultural_context_score",
            "cultural_score",
        ],
    )

    pressure = _numeric(
        frame,
        [
            "cultural_demand_pressure",
            "cultural_pressure",
        ],
    )

    tourism = _numeric(
        frame,
        [
            "tourism_intensity",
            "tourism",
            "tourist_intensity",
        ],
    )

    students = _numeric(
        frame,
        [
            "student_intensity",
            "students",
            "student_population",
        ],
    )

    temperature = _numeric(
        frame,
        [
            "temperature",
            "temperature_c",
            "avg_temperature",
            "temp",
        ],
    )

    rainfall = _numeric(
        frame,
        [
            "rainfall",
            "rainfall_mm",
            "rain",
            "precipitation",
        ],
    )

    humidity = _numeric(
        frame,
        [
            "humidity",
            "humidity_pct",
        ],
    )

    result["known_holiday"] = holiday
    result["known_promotion"] = promotion
    result["known_event"] = event
    result["known_religious"] = religious
    result["known_cultural"] = cultural
    result["known_pressure"] = pressure
    result["known_tourism"] = tourism
    result["known_students"] = students
    result["known_temperature"] = temperature
    result["known_rainfall"] = rainfall
    result["known_humidity"] = humidity

    return result.astype("float32")


def build_context_interactions(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    context = build_known_context(
        frame
    )

    holiday = context["known_holiday"]
    promotion = context["known_promotion"]
    event = context["known_event"]
    religious = context["known_religious"]
    cultural = context["known_cultural"]
    tourism = context["known_tourism"]
    students = context["known_students"]
    temperature = context["known_temperature"]
    rainfall = context["known_rainfall"]

    result = pd.DataFrame(
        index=frame.index
    )

    result["holiday_promotion"] = (
        holiday * promotion
    )

    result["holiday_event"] = (
        holiday * event
    )

    result["promotion_event"] = (
        promotion * event
    )

    result["holiday_religious"] = (
        holiday * religious
    )

    result["cultural_holiday"] = (
        cultural * holiday
    )

    result["cultural_promotion"] = (
        cultural * promotion
    )

    result["tourism_students"] = (
        tourism * students
    )

    result["temperature_rain"] = (
        temperature * rainfall
    )

    return (
        result
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(0.0)
        .astype("float32")
    )


# ============================================================================
# STATIC ENTITY FEATURES
# ============================================================================

def _hash(
    value: str,
) -> float:

    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()

    return (
        int(
            digest[:12],
            16,
        )
        /
        float(16 ** 12)
    )


def build_static_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    result = pd.DataFrame(
        index=frame.index
    )

    for column in (
        "outlet_id",
        "product_id",
        "outlet_type",
        "location_type",
        "country",
        "region",
        "product_category",
        "unit",
    ):

        if column not in frame.columns:
            continue

        if is_forbidden_feature(column):
            continue

        result[
            f"entity_{column}"
        ] = (
            frame[column]
            .fillna("__missing__")
            .astype(str)
            .map(_hash)
            .astype("float32")
        )

    return result


# ============================================================================
# TEXT MODALITY
# ============================================================================

def build_context_text(
    frame: pd.DataFrame,
    text_columns: Optional[List[str]] = None,
) -> pd.Series:

    text_columns = text_columns or []

    valid_columns = [
        column
        for column in text_columns
        if column in frame.columns
        and not is_forbidden_feature(column)
    ]

    if not valid_columns:

        return pd.Series(
            "",
            index=frame.index,
            dtype="object",
        )

    parts = []

    for column in valid_columns:

        values = (
            frame[column]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )

        parts.append(
            column
            + "="
            + values
        )

    if not parts:

        return pd.Series(
            "",
            index=frame.index,
            dtype="object",
        )

    return pd.concat(
        parts,
        axis=1,
    ).agg(
        " ".join,
        axis=1,
    )


def _zero_text_matrix(
    frame: pd.DataFrame,
) -> np.ndarray:

    return np.zeros(
        (
            len(frame),
            1,
        ),
        dtype=np.float32,
    )


def _build_text_matrix_batch(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    text_columns: Optional[List[str]],
):

    text_columns = text_columns or []

    if not text_columns:

        return (
            _zero_text_matrix(train),
            _zero_text_matrix(validation),
            _zero_text_matrix(test),
        )

    train_text = build_context_text(
        train,
        text_columns,
    )

    validation_text = build_context_text(
        validation,
        text_columns,
    )

    test_text = build_context_text(
        test,
        text_columns,
    )

    if (
        train_text.empty
        or train_text.str.len().sum() == 0
    ):

        return (
            _zero_text_matrix(train),
            _zero_text_matrix(validation),
            _zero_text_matrix(test),
        )

    vectorizer = TfidfVectorizer(
        min_df=2,
        max_features=64,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    try:

        train_sparse = (
            vectorizer.fit_transform(
                train_text
            )
        )

    except (ValueError, RuntimeError):

        return (
            _zero_text_matrix(train),
            _zero_text_matrix(validation),
            _zero_text_matrix(test),
        )

    if (
        train_sparse.shape[1] < 2
        or train_sparse.nnz == 0
    ):

        return (
            _zero_text_matrix(train),
            _zero_text_matrix(validation),
            _zero_text_matrix(test),
        )

    # ------------------------------------------------------------------
    # Check actual variance before SVD.
    # This prevents sklearn TruncatedSVD warnings for constant corpora.
    # ------------------------------------------------------------------

    variance_check = np.asarray(
        train_sparse.power(2).mean(axis=0)
    ).ravel()

    mean_check = np.asarray(
        train_sparse.mean(axis=0)
    ).ravel()

    variance = (
        variance_check
        -
        mean_check ** 2
    )

    if (
        variance.size == 0
        or not np.isfinite(variance).all()
        or float(np.max(variance)) <= 1e-12
    ):

        return (
            _zero_text_matrix(train),
            _zero_text_matrix(validation),
            _zero_text_matrix(test),
        )

    validation_sparse = (
        vectorizer.transform(
            validation_text
        )
    )

    test_sparse = (
        vectorizer.transform(
            test_text
        )
    )

    components = min(
        4,
        train_sparse.shape[1] - 1,
        max(
            1,
            train_sparse.shape[0] - 1,
        ),
    )

    if components < 1:

        return (
            train_sparse.toarray().astype(
                np.float32
            ),
            validation_sparse.toarray().astype(
                np.float32
            ),
            test_sparse.toarray().astype(
                np.float32
            ),
        )

    svd = TruncatedSVD(
        n_components=components,
        random_state=42,
    )

    train_features = svd.fit_transform(
        train_sparse
    )

    validation_features = svd.transform(
        validation_sparse
    )

    test_features = svd.transform(
        test_sparse
    )

    train_features = np.nan_to_num(
        train_features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(
        np.float32
    )

    validation_features = np.nan_to_num(
        validation_features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(
        np.float32
    )

    test_features = np.nan_to_num(
        test_features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(
        np.float32
    )

    return (
        train_features,
        validation_features,
        test_features,
    )


def build_text_matrix(
    frame_or_train,
    validation=None,
    test=None,
    text_columns=None,
):
    """
    Supports BOTH APIs.

    Single-frame API:
        build_text_matrix(df, text_columns=[...])

    Batch API:
        build_text_matrix(
            train,
            validation,
            test,
            text_columns,
        )
    """

    # Single-frame API
    if validation is None and test is None:

        frame = frame_or_train

        matrix, _, _ = _build_text_matrix_batch(
            frame,
            frame.iloc[0:0].copy(),
            frame.iloc[0:0].copy(),
            text_columns,
        )

        return matrix

    # Batch API
    return _build_text_matrix_batch(
        frame_or_train,
        validation,
        test,
        text_columns,
    )


# ============================================================================
# VISUAL MODALITY
# ============================================================================

def build_visual_features(
    frame: pd.DataFrame,
    visual_columns: Optional[List[str]],
) -> pd.DataFrame:

    visual_columns = visual_columns or []

    result = pd.DataFrame(
        index=frame.index
    )

    for column in visual_columns:

        if column not in frame.columns:
            continue

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        if values.notna().sum() == 0:
            continue

        result[column] = (
            values
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(0.0)
            .astype("float32")
        )

    return result


def build_visual_proxy_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compatibility helper.

    IMPORTANT:
    These are NOT fake visual embeddings.

    If genuine visual columns exist, they are returned.

    If they do not exist, two neutral metadata columns are returned:
        visual_available
        visual_quality

    They contain no image-derived information and therefore cannot
    manufacture visual predictive power.
    """

    if frame is None:

        return pd.DataFrame()

    if frame.empty:

        return pd.DataFrame(
            index=frame.index,
            columns=[
                "visual_available",
                "visual_quality",
            ],
        )

    visual_columns = discover_visual_columns(
        frame
    )

    if visual_columns:

        result = build_visual_features(
            frame,
            visual_columns,
        )

        if result.shape[1] >= 2:
            return result

        if result.shape[1] == 1:

            result[
                "visual_quality"
            ] = 1.0

            return result

    # Neutral capability indicators.
    #
    # They are deliberately constant and cannot improve predictions.
    return pd.DataFrame(
        {
            "visual_available": np.zeros(
                len(frame),
                dtype=np.float32,
            ),
            "visual_quality": np.zeros(
                len(frame),
                dtype=np.float32,
            ),
        },
        index=frame.index,
    )


# ============================================================================
# FEATURE MATRIX
# ============================================================================

def _temporal_columns(
    frame: pd.DataFrame,
) -> List[str]:

    columns = []

    for column in frame.columns:

        if is_forbidden_feature(column):
            continue

        if (
            column.startswith("history_")
            or column.startswith("calendar_")
            or column in {
                "dow_sin",
                "dow_cos",
                "month_sin",
                "month_cos",
            }
        ):
            columns.append(column)

    return columns


def _numeric_frame(
    frame: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:

    result = pd.DataFrame(
        index=frame.index
    )

    for column in columns:

        if column not in frame.columns:
            continue

        if is_forbidden_feature(column):
            continue

        result[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    return (
        result
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(0.0)
        .astype("float32")
    )


def prepare_features(
    df: pd.DataFrame,
    mode: str = "numeric",
    target_column: str = "deconstrained_demand",
) -> pd.DataFrame:

    if mode not in VALID_MODES:

        raise ValueError(
            f"Unknown mode: {mode}"
        )

    # Do not rebuild if historical features already exist.
    has_history = any(
        column.startswith("history_")
        for column in df.columns
    )

    if has_history:

        temporal_frame = df

    else:

        temporal_frame = build_temporal_features(
            df,
            target_column,
        )

    temporal_columns = _temporal_columns(
        temporal_frame
    )

    parts = [
        _numeric_frame(
            temporal_frame,
            temporal_columns,
        )
    ]

    if mode != "history":

        context = build_known_context(
            df
        )

        interactions = build_context_interactions(
            df
        )

        static = build_static_features(
            df
        )

        parts.extend(
            [
                context,
                interactions,
                static,
            ]
        )

    if mode in {
        "numeric_text",
        "multimodal",
    }:

        text_columns = discover_context_columns(
            df
        )[1]

        text = build_text_matrix(
            df,
            text_columns=text_columns,
        )

        text_frame = pd.DataFrame(
            text,
            index=df.index,
            columns=[
                f"text_{i}"
                for i in range(
                    text.shape[1]
                )
            ],
        )

        parts.append(
            text_frame
        )

    if mode in {
        "numeric_visual",
        "multimodal",
    }:

        visual_columns = discover_visual_columns(
            df
        )

        visual = build_visual_features(
            df,
            visual_columns,
        )

        # No genuine visual data:
        # retain neutral capability columns.
        if visual.empty:

            visual = build_visual_proxy_features(
                df
            )

        parts.append(
            visual
        )

    result = pd.concat(
        parts,
        axis=1,
    )

    result = (
        result
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(0.0)
    )

    return result.astype(
        "float32"
    )


# ============================================================================
# MODEL
# ============================================================================

def _train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LGBMRegressor:

    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=20,
        max_depth=8,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def _safe_prediction(
    model,
    X,
) -> np.ndarray:

    prediction = model.predict(
        X
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    prediction = np.nan_to_num(
        prediction,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    prediction = np.maximum(
        prediction,
        0.0,
    )

    return prediction.astype(
        "float32"
    )


def _fit_one_split(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    mode: str,
    target: str,
    numeric_context_columns=None,
    text_columns=None,
    visual_columns=None,
):

    numeric_context_columns = (
        numeric_context_columns or []
    )

    text_columns = (
        text_columns or []
    )

    visual_columns = (
        visual_columns or []
    )

    requested_columns = (
        list(numeric_context_columns)
        +
        list(text_columns)
        +
        list(visual_columns)
    )

    violations = audit_feature_firewall(
        requested_columns
    )

    if violations:

        raise ValueError(
            "Forecast-time firewall violation: "
            f"{violations}"
        )

    X_train = prepare_features(
        train,
        mode=mode,
        target_column=target,
    )

    X_predict = prepare_features(
        predict,
        mode=mode,
        target_column=target,
    )

    # Keep exact feature ordering identical.
    X_predict = X_predict.reindex(
        columns=X_train.columns,
        fill_value=0.0,
    )

    y_train = (
        pd.to_numeric(
            train[target],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float32
        )
    )

    X_train_np = (
        X_train
        .to_numpy(
            dtype=np.float32
        )
    )

    X_predict_np = (
        X_predict
        .to_numpy(
            dtype=np.float32
        )
    )

    model = _train_model(
        X_train_np,
        y_train,
    )

    prediction = _safe_prediction(
        model,
        X_predict_np,
    )

    return prediction


def fit_predict(
    train=None,
    validation=None,
    test=None,
    target="deconstrained_demand",
    mode="numeric",
    numeric_context_columns=None,
    text_columns=None,
    visual_columns=None,
    train_df=None,
    predict_df=None,
    target_column=None,
):

    """
    Compatible with both generations of the Part 27E API.

    Legacy:
        fit_predict(
            train=...,
            validation=...,
            test=...,
            target=...,
            mode=...,
        )

    Modern:
        fit_predict(
            train_df=...,
            predict_df=...,
            mode=...,
            target_column=...,
        )
    """

    if target_column is not None:
        target = target_column

    # ------------------------------------------------------------------
    # Modern two-frame API
    # ------------------------------------------------------------------

    if train_df is not None:

        if predict_df is None:

            raise ValueError(
                "predict_df is required."
            )

        prediction = _fit_one_split(
            train_df,
            predict_df,
            mode,
            target,
            numeric_context_columns,
            text_columns,
            visual_columns,
        )

        return prediction

    # ------------------------------------------------------------------
    # Legacy three-frame API
    # ------------------------------------------------------------------

    if train is None:
        raise ValueError(
            "train is required."
        )

    if validation is None:
        raise ValueError(
            "validation is required."
        )

    if test is None:
        raise ValueError(
            "test is required."
        )

    validation_prediction = _fit_one_split(
        train,
        validation,
        mode,
        target,
        numeric_context_columns,
        text_columns,
        visual_columns,
    )

    # Test prediction is trained on train + validation.
    # The final test period remains unseen.
    combined_train = pd.concat(
        [
            train,
            validation,
        ],
        axis=0,
        ignore_index=True,
    )

    test_prediction = _fit_one_split(
        combined_train,
        test,
        mode,
        target,
        numeric_context_columns,
        text_columns,
        visual_columns,
    )

    return (
        validation_prediction,
        test_prediction,
    )


# ============================================================================
# METRICS
# ============================================================================

def evaluate_ablation(
    actual,
    predictions,
) -> Dict[str, float]:

    actual = np.asarray(
        actual,
        dtype=np.float64,
    )

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    if actual.shape != predictions.shape:

        raise ValueError(
            "actual and predictions must have "
            "the same shape."
        )

    if actual.size == 0:

        raise ValueError(
            "Cannot evaluate empty predictions."
        )

    if not np.isfinite(actual).all():

        raise ValueError(
            "actual contains non-finite values."
        )

    if not np.isfinite(predictions).all():

        raise ValueError(
            "predictions contain non-finite values."
        )

    error = (
        predictions
        -
        actual
    )

    denominator = (
        np.abs(actual)
        +
        np.abs(predictions)
    )

    smape = np.where(
        denominator > 1e-12,
        2.0
        * np.abs(error)
        / denominator,
        0.0,
    )

    high_threshold = np.quantile(
        actual,
        0.75,
    )

    high_mask = (
        actual >= high_threshold
    )

    if high_mask.any():

        high_mae = float(
            np.mean(
                np.abs(
                    error[
                        high_mask
                    ]
                )
            )
        )

    else:

        high_mae = 0.0

    spike_threshold = np.quantile(
        actual,
        0.90,
    )

    spike_mask = (
        actual >= spike_threshold
    )

    if spike_mask.any():

        spike_recall = float(
            np.mean(
                predictions[
                    spike_mask
                ]
                >= spike_threshold
            )
        )

    else:

        spike_recall = 0.0

    return {
        "mae": float(
            np.mean(
                np.abs(error)
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        ),
        "smape": float(
            np.mean(smape)
            * 100.0
        ),
        "bias": float(
            np.mean(error)
        ),
        "high_demand_mae": high_mae,
        "spike_recall": spike_recall,
    }


def evaluate(
    actual,
    predictions,
):

    return evaluate_ablation(
        actual,
        predictions,
    )


# ============================================================================
# SPLITS
# ============================================================================

def build_global_splits(
    frame,
    validation_days=30,
    test_days=30,
):

    df = frame.copy()

    dates = (
        pd.to_datetime(
            df["date"],
            errors="coerce",
        )
        .dt.normalize()
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    required_dates = (
        validation_days
        +
        test_days
    )

    if len(dates) <= required_dates:

        raise ValueError(
            "Not enough dates for requested "
            "validation/test windows."
        )

    validation_dates = set(
        dates[
            -(
                validation_days
                +
                test_days
            ):
            -test_days
        ]
    )

    test_dates = set(
        dates[
            -test_days:
        ]
    )

    normalized = (
        pd.to_datetime(
            df["date"],
            errors="coerce",
        )
        .dt.normalize()
    )

    train = df[
        ~normalized.isin(
            validation_dates
            |
            test_dates
        )
    ].copy()

    validation = df[
        normalized.isin(
            validation_dates
        )
    ].copy()

    test = df[
        normalized.isin(
            test_dates
        )
    ].copy()

    return (
        train,
        validation,
        test,
    )


def build_series_splits(
    frame,
    validation_days=30,
    test_days=30,
):

    return build_global_splits(
        frame,
        validation_days,
        test_days,
    )


# ============================================================================
# VALIDATION
# ============================================================================

def validate(
    results: pd.DataFrame,
):
    """
    Validate the Part 27E ablation benchmark contract.

    Required production ablation modes:
        numeric
        numeric_text
        numeric_visual
        multimodal

    ``history`` is an optional baseline and must NOT be required
    for the validation contract.
    """

    required_columns = {
        "mode",
        "mae",
        "rmse",
        "smape",
        "bias",
        "high_demand_mae",
        "spike_recall",
    }

    # ------------------------------------------------------------------
    # Basic object validation
    # ------------------------------------------------------------------

    if not isinstance(
        results,
        pd.DataFrame,
    ):
        return {
            "passed": False,
            "errors": [
                "results must be a pandas DataFrame"
            ],
        }

    if results.empty:
        return {
            "passed": False,
            "errors": [
                "results is empty"
            ],
        }

    # ------------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------------

    missing_columns = (
        required_columns
        -
        set(results.columns)
    )

    if missing_columns:
        return {
            "passed": False,
            "errors": [
                "missing columns: "
                +
                str(
                    sorted(
                        missing_columns
                    )
                )
            ],
        }

    errors = []

    # ------------------------------------------------------------------
    # Required ablation modes
    #
    # History is intentionally NOT required.
    # It is a baseline/reference mode, not an ablation requirement.
    # ------------------------------------------------------------------

    required_modes = {
        "numeric",
        "numeric_text",
        "numeric_visual",
        "multimodal",
    }

    actual_modes = set(
        results["mode"]
        .astype(str)
        .str.strip()
    )

    missing_modes = (
        required_modes
        -
        actual_modes
    )

    if missing_modes:
        errors.append(
            "missing modes: "
            +
            str(
                sorted(
                    missing_modes
                )
            )
        )

    # ------------------------------------------------------------------
    # Metric validation
    # ------------------------------------------------------------------

    metric_columns = [
        "mae",
        "rmse",
        "smape",
        "bias",
        "high_demand_mae",
        "spike_recall",
    ]

    for column in metric_columns:

        values = pd.to_numeric(
            results[column],
            errors="coerce",
        )

        if not np.isfinite(
            values.to_numpy()
        ).all():

            errors.append(
                f"non-finite values in {column}"
            )

    # ------------------------------------------------------------------
    # Metric domain validation
    # ------------------------------------------------------------------

    mae = pd.to_numeric(
        results["mae"],
        errors="coerce",
    )

    rmse = pd.to_numeric(
        results["rmse"],
        errors="coerce",
    )

    smape = pd.to_numeric(
        results["smape"],
        errors="coerce",
    )

    high_mae = pd.to_numeric(
        results["high_demand_mae"],
        errors="coerce",
    )

    spike_recall = pd.to_numeric(
        results["spike_recall"],
        errors="coerce",
    )

    if (mae < 0).any():
        errors.append(
            "mae contains negative values"
        )

    if (rmse < 0).any():
        errors.append(
            "rmse contains negative values"
        )

    if (smape < 0).any():
        errors.append(
            "smape contains negative values"
        )

    if (high_mae < 0).any():
        errors.append(
            "high_demand_mae contains negative values"
        )

    if (
        (spike_recall < 0)
        |
        (spike_recall > 1)
    ).any():
        errors.append(
            "spike_recall must be between 0 and 1"
        )

    # ------------------------------------------------------------------
    # Duplicate mode protection
    # ------------------------------------------------------------------

    mode_counts = (
        results["mode"]
        .astype(str)
        .str.strip()
        .value_counts()
    )

    duplicated_modes = (
        mode_counts[
            mode_counts > 1
        ]
        .index
        .tolist()
    )

    if duplicated_modes:
        errors.append(
            "duplicate modes: "
            +
            str(
                sorted(
                    duplicated_modes
                )
            )
        )

    # ------------------------------------------------------------------
    # Final contract
    # ------------------------------------------------------------------

    return {
        "passed": len(errors) == 0,
        "errors": errors,
    }


# ============================================================================
# COMPATIBILITY FACADE
# ============================================================================

class MultimodalBenchmark:

    MODES = (
        "history",
        "numeric",
        "numeric_text",
        "numeric_visual",
        "multimodal",
    )

    def __init__(
        self,
        target="deconstrained_demand",
    ):

        self.target = target

    def evaluate(
        self,
        actual,
        predictions,
    ):

        return evaluate_ablation(
            actual,
            predictions,
        )

    def validate(
        self,
        results,
    ):

        return validate(
            results
        )

    def fit_predict(
        self,
        train,
        validation,
        test,
        mode,
        numeric_context_columns=None,
        text_columns=None,
        visual_columns=None,
    ):

        return fit_predict(
            train=train,
            validation=validation,
            test=test,
            target=self.target,
            mode=mode,
            numeric_context_columns=numeric_context_columns,
            text_columns=text_columns,
            visual_columns=visual_columns,
        )

    def build_visual_proxy_features(
        self,
        frame,
    ):

        return build_visual_proxy_features(
            frame
        )


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "TARGET_COLUMNS",
    "FORBIDDEN_COLUMNS",
    "FORBIDDEN_FRAGMENTS",
    "VALID_MODES",
    "is_forbidden_feature",
    "is_known_context_column",
    "is_visual_column",
    "discover_context_columns",
    "discover_visual_columns",
    "audit_feature_firewall",
    "build_temporal_features",
    "build_known_context",
    "build_context_interactions",
    "build_static_features",
    "build_context_text",
    "build_text_matrix",
    "build_visual_features",
    "build_visual_proxy_features",
    "prepare_features",
    "fit_predict",
    "evaluate_ablation",
    "evaluate",
    "build_global_splits",
    "build_series_splits",
    "validate",
    "MultimodalBenchmark",
]