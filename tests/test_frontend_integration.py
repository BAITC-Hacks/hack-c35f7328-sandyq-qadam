"""Runs against Artem's unmodified dev code plus Ilya's Inventory module.

Standalone frontend checkout: these tests explicitly skip until both are merged.
"""
import json
import importlib

import pandas as pd
import pytest

from src.pipeline import get_recommendations


@pytest.fixture(autouse=True)
def require_team_modules():
    try:
        model = importlib.import_module("src.forecasting.model")
        inventory_module = importlib.import_module("src.inventory")
    except ImportError:
        pytest.skip("Integration requires dev forecasting and feature/inventory")
    if not hasattr(model, "get_demand_forecast") or not hasattr(inventory_module, "run_inventory_pipeline"):
        pytest.skip("Integration requires implemented team modules, not empty scaffold")


@pytest.fixture
def inputs():
    inventory = pd.DataFrame([
        dict(sku="20685", supplier="Supplier_A", stock=80, incoming=40, lead_time_days=14),
        dict(sku="00123", supplier="Supplier_B", stock=1000, incoming=0, lead_time_days=7),
    ])
    rows = []
    for sku in inventory["sku"]:
        for index, day in enumerate(pd.date_range("2026-06-01", periods=84)):
            rows.append(dict(sku=sku, date=day, final_demand=10.0,
                             anomalies_found=int(index == 0), stockout_flag=int(index < 2),
                             stockout_adjustment=10.0 if index < 2 else 0.0))
    return inventory, pd.DataFrame(rows)


def test_actual_forecaster_to_inventory_to_ui_records(inputs):
    inventory, history = inputs
    rows = get_recommendations("live", inventory=inventory, demand_data=history)
    by_sku = {row["sku"]: row for row in rows}
    first = by_sku["20685"]
    assert first["forecast"] == 210
    assert first["horizon_days"] == 21
    assert first["safety_stock"] == 42
    assert first["recommended_order"] == 132
    assert first["risk"] == "HIGH"
    assert first["anomalies_found"] == 1
    assert first["stockout_days"] == 2
    assert first["recovered_lost_demand"] == 20
    assert first["data_source"] == "live"
    assert by_sku["00123"]["horizon_days"] == 14
    assert by_sku["00123"]["forecast"] == 140
    assert by_sku["00123"]["recommended_order"] == 0
    assert len(first["history"]) == 56
    json.dumps(rows, allow_nan=False)


def test_incoming_changes_real_recommendation(inputs):
    inventory, history = inputs
    before = get_recommendations("live", inventory=inventory, demand_data=history)[0]
    inventory.loc[0, "incoming"] += 20
    after = get_recommendations("live", inventory=inventory, demand_data=history)[0]
    assert before["recommended_order"] - after["recommended_order"] == 20


def test_history_changes_real_forecast(inputs):
    inventory, history = inputs
    history["final_demand"] *= 2
    rows = get_recommendations("live", inventory=inventory, demand_data=history)
    assert rows[0]["forecast"] == 420


def test_inputs_are_not_mutated(inputs):
    inventory, history = inputs
    original_stock, original_history = inventory.copy(deep=True), history.copy(deep=True)
    get_recommendations("live", inventory=inventory, demand_data=history)
    pd.testing.assert_frame_equal(inventory, original_stock)
    pd.testing.assert_frame_equal(history, original_history)


@pytest.mark.parametrize("column,value", [("stock", -1), ("incoming", float("nan")),
                                        ("lead_time_days", 1.5), ("lead_time_days", 0)])
def test_invalid_stock_inputs_rejected(inputs, column, value):
    inventory, history = inputs
    inventory[column] = inventory[column].astype(float)
    inventory.loc[0, column] = value
    with pytest.raises(ValueError):
        get_recommendations("live", inventory=inventory, demand_data=history)


def test_missing_sku_is_not_silently_dropped(inputs):
    inventory, history = inputs
    history = history.loc[history["sku"].eq("20685")]
    with pytest.raises(ValueError, match="00123"):
        get_recommendations("live", inventory=inventory, demand_data=history)


def test_duplicate_sku_is_rejected(inputs):
    inventory, history = inputs
    with pytest.raises(ValueError, match="повторяются SKU"):
        get_recommendations("live", inventory=pd.concat([inventory, inventory]), demand_data=history)


def test_bad_horizon_is_rejected(inputs, monkeypatch):
    from src.forecasting import model
    real = model.get_demand_forecast
    def wrong(**kwargs):
        result = real(**kwargs)
        result["horizon_days"] = 99
        return result
    monkeypatch.setattr(model, "get_demand_forecast", wrong)
    with pytest.raises(ValueError, match="горизонт"):
        get_recommendations("live", inventory=inputs[0], demand_data=inputs[1])


def test_real_recommendations_render_in_dashboard(inputs, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import src.pipeline as pipeline
    real = pipeline.get_recommendations
    def with_test_inputs(mode="demo"):
        if mode == "live":
            return real("live", inventory=inputs[0], demand_data=inputs[1])
        return real(mode)
    monkeypatch.setattr(pipeline, "get_recommendations", with_test_inputs)
    path = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"
    screen = AppTest.from_file(str(path), default_timeout=15).run()
    screen.radio(key="mode").set_value("Расчёт по данным").run()
    assert not screen.exception
    assert not screen.error
    assert screen.metric[0].value == "2"
    assert screen.dataframe[0].value.iloc[0]["К заказу"] == 132
