import json
from math import ceil

import pandas as pd
import pytest

from src.inventory import get_order_recommendation, get_order_recommendations, calculate_risk
from src.forecasting.model import get_demand_forecast


@pytest.fixture
def history():
    dates = pd.date_range("2024-01-01", periods=84)
    return pd.DataFrame({
        "date": dates, "sku": "00123", "product_name": "Test product",
        "final_demand": 10., "anomalies_found": 0,
        "stockout_flag": False, "stockout_adjustment": 0.,
    })


def recommend(history, **kwargs):
    options = dict(sku="00123", stock=80, incoming=40, lead_time_days=14, supplier="Supplier_A", data=history)
    options.update(kwargs)
    return get_order_recommendation(**options)


def test_real_forecasting_function_is_connected(history):
    result = recommend(history)
    assert result["forecast"] == get_demand_forecast("00123", 14, 7, data=history)["forecast"] == 210
    assert result["safety_stock"] == 42
    assert result["recommended_order"] == 132
    assert result["horizon_days"] == 21
    assert result["risk"] == "HIGH"
    assert result["product_name"] == "Test product"
    json.dumps(result, allow_nan=False)


def test_fractional_forecast_rounds_up(history, monkeypatch):
    import src.forecasting.model as model
    real = model.get_demand_forecast
    calls = []
    def fractional(sku, lead, review, data):
        calls.append((sku, lead, review))
        return {**real(sku, lead, review, data=data), "forecast": 205.12}
    monkeypatch.setattr(model, "get_demand_forecast", fractional)
    result = recommend(history)
    assert calls == [("00123", 14, 7)]
    assert result["safety_stock"] == pytest.approx(41.024)
    assert result["recommended_order"] == 127
    assert "126.144" in result["explanation"]
    assert "127" in result["explanation"]


@pytest.mark.parametrize("review", [0, 3, 14])
def test_custom_review_period_propagates(history, review):
    result = recommend(history, review_period_days=review)
    assert result["horizon_days"] == 14 + review
    assert result["review_period_days"] == review
    assert result["forecast"] == (14 + review) * 10
    assert result["recommended_order"] == ceil(max(0, result["forecast"] * 1.2 - 120))


def test_enough_stock_zeroes_order(history):
    assert recommend(history, stock=1000)["recommended_order"] == 0


def test_incoming_reduces_order(history):
    assert recommend(history, incoming=0)["recommended_order"] > recommend(history, incoming=50)["recommended_order"]


def test_safety_factor_is_a_parameter(history):
    result = recommend(history, safety_factor=.1)
    assert result["safety_stock"] == 21
    assert result["recommended_order"] == 111


def test_history_metadata_survives(history):
    history.loc[0, "anomalies_found"] = 2
    history.loc[0, "stockout_flag"] = True
    history.loc[0, "stockout_adjustment"] = 15
    result = recommend(history)
    assert result["anomalies_found"] == 2
    assert result["stockout_days"] == 1
    assert result["recovered_lost_demand"] == 15
    assert "15.000" in result["explanation"]


@pytest.mark.parametrize("position,risk", [(119, "HIGH"), (120, "MEDIUM"), (139, "MEDIUM"), (140, "LOW")])
def test_risk_boundaries(position, risk):
    assert calculate_risk(120, position, 0, 20) == risk


@pytest.mark.parametrize("field,value", [
    ("stock", -1), ("incoming", float("nan")), ("safety_factor", float("inf")),
    ("lead_time_days", 0), ("lead_time_days", 1.5), ("review_period_days", -1),
    ("minimum_order_qty", 0), ("unit_cost", -1), ("stock", True),
])
def test_invalid_warehouse_inputs_rejected(history, field, value):
    with pytest.raises(ValueError, match=field):
        recommend(history, **{field: value})


def test_unknown_sku_raises_no_mock_fallback(history):
    with pytest.raises(ValueError, match="Нет истории"):
        recommend(history, sku="MISSING")


def test_missing_file_gives_preparation_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.data.preprocess"):
        get_order_recommendation("00123", 0, 0, 14, data_path=tmp_path / "missing.csv")


def test_leading_zero_sku_survives_csv_and_changed_cwd(history, tmp_path, monkeypatch):
    path = tmp_path / "demand.csv"
    history.to_csv(path, index=False)
    monkeypatch.chdir(tmp_path)
    result = get_order_recommendation("00123", 80, 40, 14, data_path=path)
    assert result["sku"] == "00123"
    assert result["recommended_order"] == 132


def test_duplicate_daily_rows_rejected(history):
    with pytest.raises(ValueError, match="повторяющиеся"):
        recommend(pd.concat([history, history.iloc[:1]]))


@pytest.mark.parametrize("column,value", [("final_demand", float("nan")), ("stockout_flag", "maybe"), ("anomalies_found", .5)])
def test_invalid_history_rejected(history, column, value):
    history[column] = history[column].astype(object)
    history.loc[0, column] = value
    with pytest.raises(ValueError):
        recommend(history)


def test_input_is_not_mutated(history):
    original = history.copy(deep=True)
    recommend(history)
    pd.testing.assert_frame_equal(history, original)


def test_batch_groups_positive_orders(history):
    combined = pd.concat([history, history.assign(sku="SECOND")], ignore_index=True)
    rows, groups = get_order_recommendations([
        dict(sku="00123", stock=0, incoming=0, lead_time_days=14, supplier="A"),
        dict(sku="SECOND", stock=999, incoming=0, lead_time_days=14, supplier="B"),
    ], data=combined)
    assert len(rows) == 2
    assert list(groups) == ["A"]
    assert groups["A"][0]["sku"] == "00123"


def test_batch_rejects_duplicate_skus(history):
    row = dict(sku="00123", stock=0, incoming=0, lead_time_days=14)
    with pytest.raises(ValueError, match="Повторяющийся SKU"):
        get_order_recommendations([row, row], data=history)


def test_no_supplier_is_not_invented(history):
    assert recommend(history, supplier=None)["supplier"] == "Не назначен"
