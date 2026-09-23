import pytest

from src.inventory import (
    build_synthetic_inventory,
    calculate_order,
    calculate_safety_stock,
    group_orders_by_supplier,
    run_inventory_pipeline,
)


def order(**overrides):
    values = {
        "sku": "85123A",
        "forecast": 120,
        "stock": 40,
        "incoming": 30,
        "lead_time_days": 14,
        "supplier": "Supplier_A",
        "safety_stock": 20,
        "minimum_order_qty": 1,
    }
    values.update(overrides)
    return calculate_order(**values)


def test_acceptance_example_returns_70():
    assert order()["recommended_order"] == 70


def test_order_never_becomes_negative():
    assert order(stock=500)["recommended_order"] == 0


def test_incoming_reduces_order():
    assert order(incoming=0)["recommended_order"] > order(incoming=50)["recommended_order"]


def test_moq_rounds_up():
    assert order(forecast=101, stock=40, incoming=30, safety_stock=0, minimum_order_qty=20)["recommended_order"] == 40


def test_high_risk_when_inventory_does_not_cover_forecast():
    assert order(stock=40, incoming=30)["risk"] == "HIGH"


def test_medium_risk_when_forecast_covered_but_safety_not_covered():
    assert order(stock=110, incoming=10, safety_stock=20)["risk"] == "MEDIUM"


def test_low_risk_when_target_stock_is_covered():
    assert order(stock=140, incoming=0, safety_stock=20)["risk"] == "LOW"


def test_percentage_fallback_is_twenty_percent():
    assert calculate_safety_stock(120, 14, fallback_rate=0.20) == 24


def test_statistical_safety_stock_uses_forecast_error():
    result = calculate_safety_stock(120, 4, forecast_error_std=5, service_level_z=1.65)
    assert result == pytest.approx(16.5)


def test_supplier_grouping_excludes_zero_orders():
    groups = group_orders_by_supplier(
        [
            order(sku="SKU1", supplier="Supplier_A"),
            order(sku="SKU2", supplier="Supplier_B", stock=500),
            order(sku="SKU3", supplier="Supplier_A"),
        ]
    )
    assert list(groups) == ["Supplier_A"]
    assert [item["sku"] for item in groups["Supplier_A"]] == ["SKU1", "SKU3"]


def test_synthetic_inventory_is_reproducible():
    first = build_synthetic_inventory(seed=42)
    second = build_synthetic_inventory(seed=42)
    assert first.equals(second)


def test_pipeline_joins_mock_forecasts_and_inventory():
    inventory = build_synthetic_inventory(seed=42, skus=["85123A"])
    forecasts = __import__("pandas").DataFrame(
        [{"sku": "85123A", "forecast": 120, "forecast_error_std": 6}]
    )
    orders, groups = run_inventory_pipeline(inventory, forecasts)
    assert len(orders) == 1
    assert orders.iloc[0]["sku"] == "85123A"
    assert orders.iloc[0]["supplier"] in groups


def test_pipeline_rejects_missing_forecast():
    inventory = build_synthetic_inventory(seed=42, skus=["85123A"])
    forecasts = __import__("pandas").DataFrame(
        [{"sku": "OTHER", "forecast": 120, "forecast_error_std": 6}]
    )
    with pytest.raises(ValueError, match="Нет прогноза"):
        run_inventory_pipeline(inventory, forecasts)


def test_pipeline_rejects_wrong_forecast_horizon():
    inventory = build_synthetic_inventory(seed=42, skus=["85123A"])
    forecasts = __import__("pandas").DataFrame(
        [{"sku": "85123A", "forecast": 120, "forecast_error_std": 6, "horizon_days": 99}]
    )
    with pytest.raises(ValueError, match="Неверный горизонт"):
        run_inventory_pipeline(inventory, forecasts)
