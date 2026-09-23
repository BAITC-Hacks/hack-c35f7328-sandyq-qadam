from __future__ import annotations

from math import sqrt


def calculate_safety_stock(
    forecast: float,
    lead_time_days: int,
    *,
    forecast_error_std: float = 0.0,
    service_level_z: float = 1.65,
    fallback_rate: float = 0.20,
) -> float:
    """Return explainable safety stock for one SKU.

    When Data/ML provides forecast error, the statistical formula is used.
    Until then, the MVP fallback is a fixed share of the horizon forecast.
    """
    forecast = max(float(forecast), 0.0)
    lead_time_days = max(int(lead_time_days), 1)
    forecast_error_std = max(float(forecast_error_std), 0.0)
    if forecast_error_std > 0:
        return max(float(service_level_z), 0.0) * forecast_error_std * sqrt(lead_time_days)
    return forecast * max(float(fallback_rate), 0.0)

