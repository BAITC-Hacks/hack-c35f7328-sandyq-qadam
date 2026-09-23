import pandas as pd
import pytest

from src.forecasting.baseline import (
    forecast_demand,
    seasonal_weekday_forecast,
)


def create_test_data() -> pd.DataFrame:
    """
    Create synthetic daily history for one SKU.

    We use this instead of real CSV files so tests
    stay fast and reproducible.
    """

    dates = pd.date_range(
        start="2024-01-01",
        periods=120,
        freq="D",
    )

    rows = []

    for i, date in enumerate(dates):
        # Simple weekly pattern:
        # weekdays have demand,
        # weekends have lower demand.
        if date.dayofweek < 5:
            demand = 10 + (i % 3)
        else:
            demand = 3

        rows.append(
            {
                "date": date,
                "sku": "TEST_SKU",
                "product_name": "Test Product",
                "daily_sales": demand,
                "adjusted_demand": demand,
                "anomalies_found": 0,
                "stock": 100,
                "stockout_flag": False,
                "stockout_adjustment": 0.0,
                "final_demand": float(demand),
            }
        )

    return pd.DataFrame(rows)


def test_forecast_is_non_negative():
    data = create_test_data()

    result = forecast_demand(
        sku="TEST_SKU",
        horizon_days=21,
        data=data,
    )

    assert result["forecast"] >= 0


def test_forecast_horizon_is_correct():
    data = create_test_data()

    result = forecast_demand(
        sku="TEST_SKU",
        horizon_days=21,
        data=data,
    )

    assert result["horizon_days"] == 21
    assert len(result["daily_forecast"]) == 21


def test_unknown_sku_raises_error():
    data = create_test_data()

    with pytest.raises(ValueError):
        forecast_demand(
            sku="UNKNOWN_SKU",
            horizon_days=21,
            data=data,
        )


def test_invalid_horizon_raises_error():
    data = create_test_data()

    with pytest.raises(ValueError):
        forecast_demand(
            sku="TEST_SKU",
            horizon_days=0,
            data=data,
        )


def test_stockout_information_is_returned():
    data = create_test_data()

    # Simulate two historical stockouts.
    data.loc[10, "stockout_flag"] = True
    data.loc[10, "stockout_adjustment"] = 8.0
    data.loc[10, "final_demand"] += 8.0

    data.loc[20, "stockout_flag"] = True
    data.loc[20, "stockout_adjustment"] = 6.0
    data.loc[20, "final_demand"] += 6.0

    result = forecast_demand(
        sku="TEST_SKU",
        horizon_days=21,
        data=data,
    )

    assert result["stockout_days"] == 2
    assert result["recovered_lost_demand"] == 14.0


def test_anomaly_count_is_returned():
    data = create_test_data()

    data.loc[5, "anomalies_found"] = 1
    data.loc[30, "anomalies_found"] = 2

    result = forecast_demand(
        sku="TEST_SKU",
        horizon_days=21,
        data=data,
    )

    assert result["anomalies_found"] == 3


def test_seasonal_forecast_has_correct_length():
    data = create_test_data()

    history = data[
        data["sku"] == "TEST_SKU"
    ].copy()

    forecast = seasonal_weekday_forecast(
        history=history,
        horizon_days=14,
        weeks=8,
    )

    assert len(forecast) == 14


def test_seasonal_predictions_are_non_negative():
    data = create_test_data()

    history = data[
        data["sku"] == "TEST_SKU"
    ].copy()

    forecast = seasonal_weekday_forecast(
        history=history,
        horizon_days=14,
        weeks=8,
    )

    assert (
        forecast["predicted_demand"] >= 0
    ).all()