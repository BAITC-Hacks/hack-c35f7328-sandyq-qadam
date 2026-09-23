"""Public frontend entry point: real demand forecast -> an order dictionary.

No demo-forecast fallback: missing/invalid data raises an actionable exception.
Warehouse inputs belong to the caller and must not be attributed to UCI sales.
"""
from __future__ import annotations

import math
from numbers import Integral
from pathlib import Path

import pandas as pd

from .contracts import REVIEW_PERIOD_DAYS
from .explanation import build_explanation
from .order_calculator import calculate_order
from .safety_stock import calculate_safety_stock
from .supplier_grouping import group_orders_by_supplier

DEFAULT_DEMAND_PATH = Path(__file__).resolve().parents[2] / "data/processed/final_demand.csv"


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name}: требуется неотрицательное конечное число")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name}: требуется число") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name}: требуется неотрицательное конечное число")
    return number


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name}: требуется целое число >= {minimum}")
    return int(value)


def _text(value, name):
    if value is None or pd.isna(value) or not str(value).strip():
        raise ValueError(f"{name}: пустое значение")
    return str(value).strip()


def load_demand_history(data_path=None):
    path = Path(data_path) if data_path is not None else DEFAULT_DEMAND_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Не найден {path}. Подготовьте данные модулями src.data.preprocess, "
            "src.data.anomaly_detection и src.data.stockout_correction. Mock-прогноз не используется."
        )
    return pd.read_csv(path, dtype={"sku": str})


def _validate_history(data):
    required = {"sku", "date", "final_demand", "anomalies_found", "stockout_flag", "stockout_adjustment"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"В истории спроса отсутствуют поля: {', '.join(sorted(missing))}")
    result = data.copy()
    result["sku"] = result["sku"].map(lambda value: _text(value, "sku"))
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    if result["date"].isna().any() or result.duplicated(["sku", "date"]).any():
        raise ValueError("История содержит пустые даты или повторяющиеся SKU/date")
    for column in ("final_demand", "stockout_adjustment", "anomalies_found"):
        result[column] = result[column].map(lambda value: _number(value, column))
    if (result["anomalies_found"] % 1 != 0).any():
        raise ValueError("anomalies_found должен содержать целые числа")
    flags = result["stockout_flag"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if flags.isna().any():
        raise ValueError("stockout_flag: допустимы только True/False или 1/0")
    result["stockout_flag"] = flags.astype(bool)
    return result.sort_values(["sku", "date"]).reset_index(drop=True)


def get_order_recommendation(
    sku, stock, incoming, lead_time_days, review_period_days=REVIEW_PERIOD_DAYS,
    supplier=None, safety_factor=.20, *, data=None, data_path=None,
    product_name=None, minimum_order_qty=1, unit_cost=0.0,
):
    """Return a JSON-serializable recommendation using Artem's public forecast.

    `data` is an optional prepared demand DataFrame (not a mocked forecast).
    Defaults: read data/processed/final_demand.csv relative to the repository.
    `minimum_order_qty` retains the existing MVP meaning: rounding multiple.
    Incoming is assumed usable during the planning horizon, with no ETA model.
    """
    sku = _text(sku, "sku")
    stock = _number(stock, "stock")
    incoming = _number(incoming, "incoming")
    safety_factor = _number(safety_factor, "safety_factor")
    unit_cost = _number(unit_cost, "unit_cost")
    lead_time_days = _integer(lead_time_days, "lead_time_days", 1)
    review_period_days = _integer(review_period_days, "review_period_days")
    minimum_order_qty = _integer(minimum_order_qty, "minimum_order_qty", 1)
    supplier = _text(supplier, "supplier") if supplier is not None else "Не назначен"
    if data is not None and data_path is not None:
        raise ValueError("Передайте только data или data_path, не оба сразу")
    history = _validate_history(data if data is not None else load_demand_history(data_path))
    selected = history[history["sku"] == sku]
    if selected.empty:
        raise ValueError(f"Нет истории спроса для SKU {sku}")

    # Lazy import keeps legacy inventory-only usage independent of forecasting.
    from src.forecasting.model import get_demand_forecast

    prediction = get_demand_forecast(sku, lead_time_days, review_period_days, data=history)
    horizon = lead_time_days + review_period_days
    if prediction["horizon_days"] != horizon or str(prediction["sku"]) != sku:
        raise ValueError("Forecasting вернул другой SKU или горизонт")
    forecast = _number(prediction["forecast"], "forecast")
    safety = calculate_safety_stock(forecast, lead_time_days, fallback_rate=safety_factor)
    decision = calculate_order(
        sku, forecast, stock, incoming, lead_time_days, supplier,
        safety_stock=safety, minimum_order_qty=minimum_order_qty, unit_cost=unit_cost,
        review_period_days=review_period_days,
    )
    if product_name is None and "product_name" in selected:
        names = selected["product_name"].dropna().astype(str)
        names = names[names.str.strip() != ""]
        product_name = names.iloc[-1] if not names.empty else None
    decision.update({
        "product_name": str(product_name) if product_name is not None else sku,
        "anomalies_found": int(prediction["anomalies_found"]),
        "stockout_days": int(prediction["stockout_days"]),
        "recovered_lost_demand": float(prediction["recovered_lost_demand"]),
        "safety_factor": safety_factor,
        "safety_stock_method": "fraction_of_horizon_forecast",
        "forecast_source": "src.forecasting.model.get_demand_forecast",
    })
    decision["explanation"] = build_explanation(
        **{key: decision[key] for key in (
            "forecast", "horizon_days", "stock", "incoming", "safety_stock",
            "recommended_order", "risk", "minimum_order_qty", "anomalies_found",
            "stockout_days", "recovered_lost_demand",
        )}
    )
    return decision


def get_order_recommendations(warehouse_rows, *, data=None, data_path=None, safety_factor=.20):
    """Batch helper; returns rows plus positive orders grouped by supplier."""
    if data is not None and data_path is not None:
        raise ValueError("Передайте только data или data_path, не оба сразу")
    history = data if data is not None else load_demand_history(data_path)
    decisions, seen = [], set()
    for row in warehouse_rows:
        sku = _text(row["sku"], "sku")
        if sku in seen:
            raise ValueError(f"Повторяющийся SKU в складских данных: {sku}")
        seen.add(sku)
        options = dict(row)
        options.setdefault("safety_factor", safety_factor)
        decisions.append(get_order_recommendation(**options, data=history))
    return decisions, group_orders_by_supplier(decisions)
