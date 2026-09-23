"""Inventory decision module: forecast in, supplier order out."""

from .order_calculator import calculate_order
from .pipeline import run_inventory_pipeline
from .contracts import ForecastInput, OrderDecision, REVIEW_PERIOD_DAYS, expected_horizon_days
from .safety_stock import calculate_safety_stock
from .supplier_grouping import group_orders_by_supplier
from .synthetic import build_synthetic_inventory

__all__ = [
    "build_synthetic_inventory",
    "expected_horizon_days",
    "ForecastInput",
    "OrderDecision",
    "REVIEW_PERIOD_DAYS",
    "calculate_order",
    "calculate_safety_stock",
    "group_orders_by_supplier",
    "run_inventory_pipeline",
]
