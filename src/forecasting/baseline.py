from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


MODELING_DATA_PATH = Path(
    "data/processed/modeling_daily_sales.csv"
)

ADJUSTED_DEMAND_PATH = Path(
    "data/processed/adjusted_daily_demand.csv"
)


def load_forecasting_data() -> pd.DataFrame:
    """
    Combine the complete daily calendar with
    anomaly-adjusted regular demand.
    """

    modeling = pd.read_csv(
        MODELING_DATA_PATH,
        parse_dates=["date"],
    )

    adjusted = pd.read_csv(
        ADJUSTED_DEMAND_PATH,
        parse_dates=["date"],
    )

    modeling["sku"] = (
        modeling["sku"]
        .astype(str)
        .str.strip()
    )

    adjusted["sku"] = (
        adjusted["sku"]
        .astype(str)
        .str.strip()
    )

    adjusted = adjusted[
        [
            "date",
            "sku",
            "adjusted_demand",
            "anomalies_found",
        ]
    ]

    data = modeling.merge(
        adjusted,
        on=["date", "sku"],
        how="left",
    )

    # Missing date in adjusted data means no sale that day.
    data["adjusted_demand"] = (
        data["adjusted_demand"]
        .fillna(0.0)
    )

    data["anomalies_found"] = (
        data["anomalies_found"]
        .fillna(0)
        .astype(int)
    )

    data = data.sort_values(
        ["sku", "date"]
    ).reset_index(drop=True)

    return data


def seasonal_weekday_forecast(
    history: pd.DataFrame,
    horizon_days: int,
    weeks: int = 8,
) -> pd.DataFrame:
    """
    Forecast future demand using recent demand
    from the same day of week.

    Example:
    future Monday is estimated using previous Mondays.

    This gives us a simple seasonal baseline.
    """

    history = (
        history
        .sort_values("date")
        .copy()
    )

    if history.empty:
        raise ValueError(
            "History is empty."
        )

    history["day_of_week"] = (
        history["date"].dt.dayofweek
    )

    last_date = history["date"].max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )

    predictions = []

    recent_start = (
        last_date
        - pd.Timedelta(
            days=weeks * 7
        )
    )

    recent = history[
        history["date"] > recent_start
    ].copy()

    fallback = float(
        recent["adjusted_demand"].mean()
    )

    for future_date in future_dates:
        weekday = future_date.dayofweek

        same_weekday = recent[
            recent["day_of_week"] == weekday
        ]["adjusted_demand"]

        if len(same_weekday) > 0:
            prediction = float(
                same_weekday.mean()
            )
        else:
            prediction = fallback

        prediction = max(
            0.0,
            prediction,
        )

        predictions.append(
            {
                "date": future_date,
                "predicted_demand": prediction,
            }
        )

    return pd.DataFrame(predictions)


def forecast_demand(
    sku: str,
    horizon_days: int,
    data: pd.DataFrame | None = None,
    weeks: int = 8,
) -> dict:
    """
    Forecast cumulative demand for one SKU.

    Contract:
    horizon_days should equal:
        lead_time_days + review_period_days

    Returns the total forecast over the horizon.
    """

    if horizon_days <= 0:
        raise ValueError(
            "horizon_days must be positive."
        )

    if data is None:
        data = load_forecasting_data()

    sku = str(sku)

    history = data[
        data["sku"] == sku
    ].copy()

    if history.empty:
        raise ValueError(
            f"SKU not found: {sku}"
        )

    future = seasonal_weekday_forecast(
        history=history,
        horizon_days=horizon_days,
        weeks=weeks,
    )

    total_forecast = float(
        future[
            "predicted_demand"
        ].sum()
    )

    anomaly_count = int(
        history[
            "anomalies_found"
        ].sum()
    )

    return {
        "sku": sku,
        "forecast": round(
            total_forecast,
            2,
        ),
        "horizon_days": horizon_days,
        "anomalies_found": anomaly_count,
        "stockout_days": 0,
        "daily_forecast": future,
    }


def evaluate_baseline(
    data: pd.DataFrame,
    test_days: int = 28,
    weeks: int = 8,
) -> dict:
    """
    Time-based backtest.

    For each SKU:
    - last test_days are test data;
    - all earlier observations are training history;
    - forecast the test period;
    - calculate MAE.
    """

    actual_values = []
    predicted_values = []

    sku_results = []

    for sku, group in data.groupby("sku"):
        group = (
            group
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(group) <= test_days + 56:
            continue

        train = group.iloc[
            :-test_days
        ].copy()

        test = group.iloc[
            -test_days:
        ].copy()

        forecast = (
            seasonal_weekday_forecast(
                history=train,
                horizon_days=test_days,
                weeks=weeks,
            )
        )

        actual = (
            test["adjusted_demand"]
            .to_numpy()
        )

        predicted = (
            forecast["predicted_demand"]
            .to_numpy()
        )

        mae = mean_absolute_error(
            actual,
            predicted,
        )

        sku_results.append(
            {
                "sku": sku,
                "mae": mae,
            }
        )

        actual_values.extend(actual)
        predicted_values.extend(
            predicted
        )

    overall_mae = mean_absolute_error(
        actual_values,
        predicted_values,
    )

    results = pd.DataFrame(
        sku_results
    )

    return {
        "overall_mae": overall_mae,
        "sku_results": results,
    }


def main():
    print(
        "Loading forecasting data..."
    )

    data = load_forecasting_data()

    print(
        "Data:",
        data.shape,
    )

    print(
        "SKU:",
        data["sku"].nunique(),
    )

    print()

    example_sku = (
        data["sku"]
        .value_counts()
        .index[0]
    )

    print(
        "Example SKU:",
        example_sku,
    )

    result = forecast_demand(
        sku=example_sku,
        horizon_days=21,
        data=data,
    )

    print()
    print(
        "========== FORECAST =========="
    )

    print(
        "SKU:",
        result["sku"],
    )

    print(
        "Horizon:",
        result["horizon_days"],
    )

    print(
        "Forecast:",
        result["forecast"],
    )

    print(
        "Historical anomalies:",
        result["anomalies_found"],
    )

    print()
    print(
        "Daily forecast:"
    )

    print(
        result["daily_forecast"].head(10)
    )

    print()
    print(
        "Running baseline backtest..."
    )

    evaluation = evaluate_baseline(
        data=data,
        test_days=28,
        weeks=8,
    )

    print()
    print(
        "========== EVALUATION =========="
    )

    print(
        "Overall MAE:",
        round(
            evaluation[
                "overall_mae"
            ],
            2,
        )
    )

    print()
    print(
        "Best SKU by MAE:"
    )

    print(
        evaluation[
            "sku_results"
        ]
        .sort_values("mae")
        .head(10)
    )


if __name__ == "__main__":
    main()