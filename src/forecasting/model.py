from typing import Any

import pandas as pd

from src.forecasting.baseline import (
    forecast_demand,
)


DEFAULT_REVIEW_PERIOD_DAYS = 7


def get_demand_forecast(
    sku: str,
    lead_time_days: int,
    review_period_days: int = DEFAULT_REVIEW_PERIOD_DAYS,
    data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Public forecasting interface for the rest of the project.

    Contract:
        forecast = total expected demand during

        lead_time_days + review_period_days

    Inventory logic must NOT multiply forecast
    by lead time again.
    """

    if lead_time_days < 0:
        raise ValueError(
            "lead_time_days cannot be negative."
        )

    if review_period_days < 0:
        raise ValueError(
            "review_period_days cannot be negative."
        )

    horizon_days = (
        lead_time_days
        + review_period_days
    )

    if horizon_days <= 0:
        raise ValueError(
            "Forecast horizon must be positive."
        )

    result = forecast_demand(
        sku=sku,
        horizon_days=horizon_days,
        data=data,
    )

    # Return only clean values needed by
    # Inventory / Backend.
    return {
        "sku": result["sku"],
        "forecast": result["forecast"],
        "horizon_days": result["horizon_days"],
        "lead_time_days": lead_time_days,
        "review_period_days": review_period_days,
        "anomalies_found": result[
            "anomalies_found"
        ],
        "stockout_days": result[
            "stockout_days"
        ],
        "recovered_lost_demand": result[
            "recovered_lost_demand"
        ],
    }