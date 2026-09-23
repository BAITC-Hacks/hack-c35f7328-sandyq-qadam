from pathlib import Path
from math import ceil

import pytest
from streamlit.testing.v1 import AppTest

from app.components import create_mock_history
from src.pipeline import get_recommendations

ROOT = Path(__file__).resolve().parents[1]


def test_mock_contract_and_arithmetic():
    rows = get_recommendations()
    assert len(rows) == 10
    for row in rows:
        assert row["data_source"] == "demo"
        assert row["risk"] in {"HIGH", "MEDIUM", "LOW"}
        assert row["horizon_days"] == row["lead_time_days"] + row["review_period_days"]
        target = row["forecast"] + row["safety_stock"]
        position = row["stock"] + row["incoming"]
        assert row["recommended_order"] == max(0, ceil(target - position))
        assert row["risk"] == ("HIGH" if position < row["forecast"] else "MEDIUM" if position < target else "LOW")


def test_demo_does_not_share_mutable_records():
    rows = get_recommendations()
    rows[0]["stock"] = -100
    assert get_recommendations()[0]["stock"] == 40


@pytest.mark.parametrize("index", [0, 1])
def test_demo_chart_matches_full_horizon(index):
    item = get_recommendations()[index]
    history, forecast = create_mock_history(item)
    assert len(forecast) == item["horizon_days"]
    assert forecast["Прогноз"].sum() == pytest.approx(item["forecast"])
    assert history["Дата"].max() < forecast["Дата"].min()


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="режим"):
        get_recommendations("invalid")


def app():
    return AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=15).run()


def test_dashboard_starts_and_high_risk_is_not_zero():
    screen = app()
    assert not screen.exception
    assert screen.metric[0].value == "10"
    assert screen.metric[1].value == "5"
    assert len(screen.tabs) == 2


def test_risk_and_supplier_filters():
    screen = app()
    screen.selectbox(key="risk").select("HIGH").run()
    assert not screen.exception
    table = screen.dataframe[0].value
    assert len(table) == 5
    assert set(table["Риск"]) == {"Высокий"}
    screen.selectbox(key="supplier").select("Supplier_B").run()
    assert len(screen.dataframe[0].value) == 1
    assert screen.dataframe[0].value.iloc[0]["SKU"] == "84997D"


def test_empty_filter_is_handled():
    screen = app()
    screen.selectbox(key="supplier").select("Supplier_A").run()
    screen.selectbox(key="risk").select("MEDIUM").run()
    assert not screen.exception
    assert screen.dataframe[0].value.empty
    assert any("не найдены" in warning.value for warning in screen.warning)


def test_live_missing_dependencies_or_data_shows_error_not_demo(monkeypatch, tmp_path):
    monkeypatch.setattr("src.pipeline.ROOT", tmp_path)
    screen = app()
    screen.radio(key="mode").set_value("Расчёт по данным").run()
    assert not screen.exception
    assert len(screen.error) == 1
    assert len(screen.metric) == 0
