from __future__ import annotations

from math import ceil
from typing import Any

from .contracts import REVIEW_PERIOD_DAYS, expected_horizon_days
from .risk import calculate_risk
from .explanation import build_explanation


def _round_to_moq(value: float, minimum_order_qty: int) -> int:
    if value <= 0:
        return 0
    minimum_order_qty = max(int(minimum_order_qty), 1)
    return int(ceil(value / minimum_order_qty) * minimum_order_qty)


def calculate_order(
    sku: str,
    forecast: float,
    stock: float,
    incoming: float,
    lead_time_days: int,
    supplier: str,
    *,
    safety_stock: float,
    minimum_order_qty: int = 1,
    unit_cost: float = 0.0,
    review_period_days: int = REVIEW_PERIOD_DAYS,
) -> dict[str, Any]:
    """Convert a horizon forecast into an explainable supplier order.

    Contract: `forecast` already covers `lead_time_days + review_period_days`.
    Lead time is retained for traceability and is never applied twice.
    """
    forecast = max(float(forecast), 0.0)
    stock = max(float(stock), 0.0)
    incoming = max(float(incoming), 0.0)
    safety_stock = max(float(safety_stock), 0.0)
    lead_time_days = max(int(lead_time_days), 1)
    unit_cost = max(float(unit_cost), 0.0)

    inventory_position = stock + incoming
    target_stock = forecast + safety_stock
    raw_order = target_stock - inventory_position
    recommended_order = _round_to_moq(raw_order, minimum_order_qty)

    risk = calculate_risk(forecast, stock, incoming, safety_stock)
    explanation = build_explanation(
        forecast=forecast, horizon_days=expected_horizon_days(lead_time_days, review_period_days),
        stock=stock, incoming=incoming, safety_stock=safety_stock,
        recommended_order=recommended_order, risk=risk,
        minimum_order_qty=max(int(minimum_order_qty), 1),
    )
    return {
        "sku": str(sku),
        "supplier": str(supplier),
        "forecast": forecast,
        "stock": stock,
        "incoming": incoming,
        "lead_time_days": lead_time_days,
        "review_period_days": review_period_days,
        "horizon_days": expected_horizon_days(lead_time_days, review_period_days),
        "safety_stock": safety_stock,
        "minimum_order_qty": max(int(minimum_order_qty), 1),
        "recommended_order": recommended_order,
        "unit_cost": unit_cost,
        "order_value": recommended_order * unit_cost,
        "risk": risk,
        "explanation": explanation,
    }
