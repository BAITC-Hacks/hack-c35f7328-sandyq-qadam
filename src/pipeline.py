"""UI boundary: explicit demo or real Forecasting -> Inventory integration."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _numbers(frame, columns, *, integers=()):
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column]
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{column}: нужны конечные неотрицательные числа.")
        if column in integers and (values % 1 != 0).any():
            raise ValueError(f"{column}: нужны целые числа.")


def get_recommendations(mode="demo", *, inventory=None, demand_data=None):
    """Return UI-ready records. Real failures NEVER silently become demo data.

    Real mode accepts DataFrames or reads repository-root data files.
    Forecast is total demand over lead_time_days + 7, never a daily rate.
    """
    if mode == "demo":
        from app.mock_data import MOCK_DATA
        return [dict(item, data_source="demo") for item in deepcopy(MOCK_DATA)]
    if mode != "live":
        raise ValueError("Неизвестный режим: используйте demo или live.")

    try:
        from src.forecasting.model import get_demand_forecast
        from src.inventory import run_inventory_pipeline
    except ImportError as exc:
        raise ValueError(
            "Для расчёта подключите Data/ML из dev и модуль feature/inventory. "
            "Демо работает отдельно."
        ) from exc

    if inventory is None:
        path = ROOT / "data/synthetic/inventory.csv"
        if not path.is_file():
            raise ValueError(f"Нет складских данных: {path}")
        inventory = pd.read_csv(path, dtype={"sku": str})
    if demand_data is None:
        path = ROOT / "data/processed/final_demand.csv"
        if not path.is_file():
            raise ValueError(
                "Нет data/processed/final_demand.csv. Сначала запустите подготовку "
                "данных Артёма; настоящие данные не заменяются демо автоматически."
            )
        demand_data = pd.read_csv(path, dtype={"sku": str}, parse_dates=["date"])

    inventory = inventory.copy(deep=True)
    demand_data = demand_data.copy(deep=True)
    required_stock = {"sku", "stock", "incoming", "lead_time_days", "supplier"}
    required_history = {"sku", "date", "final_demand", "anomalies_found",
                        "stockout_flag", "stockout_adjustment"}
    for name, frame, required in [("склад", inventory, required_stock),
                                  ("история", demand_data, required_history)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: отсутствуют поля {', '.join(sorted(missing))}")
        if frame["sku"].isna().any():
            raise ValueError(f"{name}: SKU не может быть пустым.")
        frame["sku"] = frame["sku"].astype(str).str.strip()
        if frame["sku"].eq("").any():
            raise ValueError(f"{name}: SKU не может быть пустым.")
    if inventory.empty:
        return []
    if inventory["sku"].duplicated().any():
        raise ValueError("В складе повторяются SKU.")
    if inventory["supplier"].isna().any():
        raise ValueError("Не указан поставщик.")
    for column, default in [("minimum_order_qty", 1), ("unit_cost", 0)]:
        if column not in inventory:
            inventory[column] = default
    _numbers(inventory, ["stock", "incoming", "lead_time_days", "minimum_order_qty", "unit_cost"],
             integers=("lead_time_days", "minimum_order_qty"))
    if (inventory["lead_time_days"] < 1).any() or (inventory["minimum_order_qty"] < 1).any():
        raise ValueError("Срок поставки и кратность партии должны быть не меньше 1.")
    demand_data["date"] = pd.to_datetime(demand_data["date"], errors="raise")
    if demand_data["date"].isna().any():
        raise ValueError("В истории есть пустые даты.")
    if demand_data.duplicated(["sku", "date"]).any():
        raise ValueError("В истории повторяется сочетание SKU и даты.")
    _numbers(demand_data, ["final_demand", "anomalies_found", "stockout_flag", "stockout_adjustment"])
    missing_skus = sorted(set(inventory["sku"]) - set(demand_data["sku"]))
    if missing_skus:
        raise ValueError("Нет истории для SKU: " + ", ".join(missing_skus))

    results = []
    for row in inventory.itertuples(index=False):
        result = get_demand_forecast(
            sku=row.sku, lead_time_days=int(row.lead_time_days),
            review_period_days=7, data=demand_data,
        )
        if result["horizon_days"] != int(row.lead_time_days) + 7:
            raise ValueError(f"Неверный горизонт прогноза для {row.sku}")
        results.append(result)

    # Select fields explicitly: the two lead_time columns must not collide.
    forecasts = pd.DataFrame(results)[["sku", "forecast", "horizon_days"]].copy()
    # Zero selects the documented 20% fallback, NOT a measured zero error.
    forecasts["forecast_error_std"] = 0.0
    orders, _ = run_inventory_pipeline(inventory, forecasts)
    details = {result["sku"]: result for result in results}
    warehouse = inventory.set_index("sku")
    records = []
    for order in orders.to_dict("records"):
        sku = order["sku"]
        result = details[sku]
        order.update({key: result[key] for key in
                      ("anomalies_found", "stockout_days", "recovered_lost_demand")})
        name = warehouse.loc[sku].get("product_name", sku)
        order["product_name"] = sku if pd.isna(name) else str(name)
        order["data_source"] = "live"
        order["safety_stock_method"] = "20% прогноза; ошибка модели не предоставлена"
        order["explanation"] += (
            f" Горизонт: {order['horizon_days']} дней. Страховой запас — 20% прогноза; "
            "оценка ошибки модели отсутствует. Риск оценивает покрытие всего горизонта; "
            "товар в пути считается прибывающим вовремя."
        )
        history = demand_data.loc[demand_data["sku"].eq(sku)].sort_values("date").tail(56)
        order["history"] = [
            {"date": row.date.strftime("%Y-%m-%d"), "demand": float(row.final_demand),
             "anomaly": bool(row.anomalies_found > 0)}
            for row in history.itertuples(index=False)
        ]
        # Public API returns an aggregate. Do not invent a daily forecast chart.
        records.append(order)
    return records
