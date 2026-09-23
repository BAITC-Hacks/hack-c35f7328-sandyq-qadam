from __future__ import annotations

from typing import TypedDict


REVIEW_PERIOD_DAYS = 7


class ForecastInput(TypedDict):
    sku: str
    forecast: float
    forecast_error_std: float
    horizon_days: int


class OrderDecision(TypedDict):
    sku: str
    supplier: str
    forecast: float
    stock: float
    incoming: float
    lead_time_days: int
    review_period_days: int
    horizon_days: int
    safety_stock: float
    minimum_order_qty: int
    recommended_order: int
    unit_cost: float
    order_value: float
    risk: str
    explanation: str


def expected_horizon_days(lead_time_days: int) -> int:
    return max(int(lead_time_days), 1) + REVIEW_PERIOD_DAYS

