
import pandas as pd

from app.features.time_features import (
    create_time_features,
)

from app.features.lag_features import (
    create_lag_features,
)

from app.features.rolling_features import (
    create_rolling_features,
)

from app.features.context_features import (
    create_context_features,
)

from app.features.trend_features import (
    create_trend_features,
)

from app.features.volatility_features import (
    create_volatility_features,
)

from app.features.order_velocity import (
    create_order_velocity_features,
)

from app.features.seasonality_features import (
    create_seasonality_features,
)


def build_features(
    df: pd.DataFrame,
    date_column: str = "date",
    target_column: str = "quantity_sold",
    group_columns: list[str] | None = None,
) -> pd.DataFrame:

    if group_columns is None:
        group_columns = [
            "outlet_id",
            "product_id",
        ]

    result = df.copy()

    # ==========================================
    # 1. TIME
    # ==========================================

    result = create_time_features(
        result,
        date_column=date_column,
    )

    # ==========================================
    # 2. HISTORICAL LAGS
    # ==========================================

    if (
        target_column in result.columns
        and all(
            column in result.columns
            for column in group_columns
        )
    ):

        result = create_lag_features(
            result,
            target_column=target_column,
            group_columns=group_columns,
        )

        # ======================================
        # 3. HISTORICAL ROLLING FEATURES
        # ======================================

        result = create_rolling_features(
            result,
            target_column=target_column,
            group_columns=group_columns,
        )

        # ======================================
        # 4. ADVANCED TRENDS
        # ======================================

        result = create_trend_features(
            result,
            target_column=target_column,
            group_columns=group_columns,
        )

        # ======================================
        # 5. VOLATILITY
        # ======================================

        result = create_volatility_features(
            result,
            target_column=target_column,
            group_columns=group_columns,
        )

        # ======================================
        # 6. ORDER VELOCITY
        # ======================================

        result = create_order_velocity_features(
            result,
            target_column=target_column,
            group_columns=group_columns,
        )

    # ==========================================
    # 7. SEASONALITY
    # ==========================================

    result = create_seasonality_features(
        result
    )

    # ==========================================
    # 8. CONTEXT
    # ==========================================

    result = create_context_features(
        result
    )

    return result

