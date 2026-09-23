from __future__ import annotations

from typing import Any

import pandas as pd

from .contracts import expected_horizon_days
from .order_calculator import calculate_order
from .safety_stock import calculate_safety_stock
from .supplier_grouping import group_orders_by_supplier


INVENTORY_COLUMNS = {
    "sku",
    "stock",
    "incoming",
    "lead_time_days",
    "supplier",
    "minimum_order_qty",
    "unit_cost",
}
FORECAST_COLUMNS = {"sku", "forecast", "forecast_error_std"}


def _validate(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"В таблице {name} отсутствуют поля: {', '.join(sorted(missing))}")
    if frame["sku"].duplicated().any():
        duplicates = sorted(frame.loc[frame["sku"].duplicated(), "sku"].astype(str).unique())
        raise ValueError(f"В таблице {name} повторяются SKU: {', '.join(duplicates)}")


def run_inventory_pipeline(
    inventory: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    service_level_z: float = 1.65,
    fallback_rate: float = 0.20,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    """Join Data/ML forecasts with warehouse state and build supplier orders."""
    _validate(inventory, INVENTORY_COLUMNS, "inventory")
    _validate(forecasts, FORECAST_COLUMNS, "forecasts")
    merged = inventory.merge(forecasts, on="sku", how="left", validate="one_to_one")
    missing_forecasts = merged.loc[merged["forecast"].isna(), "sku"].astype(str).tolist()
    if missing_forecasts:
        raise ValueError("Нет прогноза для SKU: " + ", ".join(missing_forecasts))
    if "horizon_days" in forecasts.columns:
        expected = merged["lead_time_days"].map(expected_horizon_days)
        mismatch = merged.loc[merged["horizon_days"].astype(int) != expected.astype(int), ["sku", "horizon_days"]]
        if not mismatch.empty:
            details = ", ".join(f"{row.sku}: {int(row.horizon_days)}" for row in mismatch.itertuples())
            raise ValueError(f"Неверный горизонт прогноза (ожидается lead_time + 7): {details}")

    orders: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        safety_stock = calculate_safety_stock(
            row.forecast,
            row.lead_time_days,
            forecast_error_std=row.forecast_error_std,
            service_level_z=service_level_z,
            fallback_rate=fallback_rate,
        )
        decision = calculate_order(
            sku=row.sku,
            forecast=row.forecast,
            stock=row.stock,
            incoming=row.incoming,
            lead_time_days=row.lead_time_days,
            supplier=row.supplier,
            safety_stock=safety_stock,
            minimum_order_qty=row.minimum_order_qty,
            unit_cost=row.unit_cost,
        )
        if hasattr(row, "category"):
            decision["category"] = row.category
        orders.append(decision)

    orders_frame = pd.DataFrame(orders).sort_values(
        ["risk", "supplier", "sku"],
        key=lambda column: column.map({"HIGH": 0, "MEDIUM": 1, "LOW": 2}).fillna(column)
        if column.name == "risk"
        else column,
    ).reset_index(drop=True)
    return orders_frame, group_orders_by_supplier(orders)
