from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error


FINAL_DEMAND_PATH = Path(
    "data/processed/final_demand.csv"
)


def load_forecasting_data() -> pd.DataFrame:
    """
    Load final demand history after:
    - one-off order correction
    - stockout correction
    """

    data = pd.read_csv(
        FINAL_DEMAND_PATH,
        parse_dates=["date"],
    )

    data["sku"] = (
        data["sku"]
        .astype(str)
        .str.strip()
    )

    data = (
        data.sort_values(
            ["sku", "date"]
        )
        .reset_index(drop=True)
    )

    return data


def simple_mean_forecast(
    history: pd.DataFrame,
    horizon_days: int,
    lookback_days: int = 56,
) -> pd.DataFrame:
    """
    Very simple baseline:
    mean final demand over recent days.
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

    recent = history.tail(
        lookback_days
    )

    mean_demand = float(
        recent["final_demand"].mean()
    )

    mean_demand = max(
        0.0,
        mean_demand,
    )

    last_date = history["date"].max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )

    return pd.DataFrame(
        {
            "date": future_dates,
            "predicted_demand": mean_demand,
        }
    )


def seasonal_weekday_forecast(
    history: pd.DataFrame,
    horizon_days: int,
    weeks: int = 8,
) -> pd.DataFrame:
    """
    Seasonal baseline.

    Each future weekday is predicted from recent
    same weekdays using final demand.
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
        recent["final_demand"].mean()
    )

    predictions = []

    for future_date in future_dates:
        weekday = future_date.dayofweek

        same_weekday = recent[
            recent["day_of_week"] == weekday
        ]["final_demand"]

        if not same_weekday.empty:
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

    return pd.DataFrame(
        predictions
    )


def forecast_demand(
    sku: str,
    horizon_days: int,
    data: pd.DataFrame | None = None,
    weeks: int = 8,
) -> dict:
    """
    Main forecasting contract.

    horizon_days =
    lead_time_days + review_period_days

    Returns total forecast over the full horizon.
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

    stockout_days = int(
        history[
            "stockout_flag"
        ].sum()
    )

    recovered_demand = float(
        history[
            "stockout_adjustment"
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
        "stockout_days": stockout_days,
        "recovered_lost_demand": round(
            recovered_demand,
            2,
        ),
        "daily_forecast": future,
    }


def evaluate_baselines(
    data: pd.DataFrame,
    test_days: int = 28,
    lookback_days: int = 56,
    weeks: int = 8,
) -> dict:
    """
    Compare:
    1. simple recent mean
    2. seasonal weekday baseline

    Uses time-based backtesting.
    """

    all_actual = []
    all_simple_predictions = []
    all_seasonal_predictions = []

    sku_results = []

    for sku, group in data.groupby(
        "sku"
    ):
        group = (
            group.sort_values("date")
            .reset_index(drop=True)
        )

        minimum_history = (
            test_days
            + max(
                lookback_days,
                weeks * 7,
            )
        )

        if len(group) <= minimum_history:
            continue

        train = group.iloc[
            :-test_days
        ].copy()

        test = group.iloc[
            -test_days:
        ].copy()

        simple = simple_mean_forecast(
            history=train,
            horizon_days=test_days,
            lookback_days=lookback_days,
        )

        seasonal = (
            seasonal_weekday_forecast(
                history=train,
                horizon_days=test_days,
                weeks=weeks,
            )
        )

        actual = (
            test["final_demand"]
            .to_numpy()
        )

        simple_pred = (
            simple[
                "predicted_demand"
            ]
            .to_numpy()
        )

        seasonal_pred = (
            seasonal[
                "predicted_demand"
            ]
            .to_numpy()
        )

        simple_mae = (
            mean_absolute_error(
                actual,
                simple_pred,
            )
        )

        seasonal_mae = (
            mean_absolute_error(
                actual,
                seasonal_pred,
            )
        )

        sku_results.append(
            {
                "sku": sku,
                "simple_mae": simple_mae,
                "seasonal_mae": seasonal_mae,
                "seasonal_better": (
                    seasonal_mae
                    < simple_mae
                ),
            }
        )

        all_actual.extend(
            actual
        )

        all_simple_predictions.extend(
            simple_pred
        )

        all_seasonal_predictions.extend(
            seasonal_pred
        )

    simple_overall_mae = (
        mean_absolute_error(
            all_actual,
            all_simple_predictions,
        )
    )

    seasonal_overall_mae = (
        mean_absolute_error(
            all_actual,
            all_seasonal_predictions,
        )
    )

    improvement_percent = (
        (
            simple_overall_mae
            - seasonal_overall_mae
        )
        / simple_overall_mae
        * 100
        if simple_overall_mae > 0
        else 0.0
    )

    sku_results = pd.DataFrame(
        sku_results
    )

    seasonal_wins = int(
        sku_results[
            "seasonal_better"
        ].sum()
    )

    return {
        "simple_mae": simple_overall_mae,
        "seasonal_mae": seasonal_overall_mae,
        "improvement_percent": (
            improvement_percent
        ),
        "seasonal_wins": seasonal_wins,
        "total_skus": len(
            sku_results
        ),
        "sku_results": sku_results,
    }


def main():
    print(
        "Loading final demand..."
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

    example_sku = (
        data["sku"]
        .value_counts()
        .index[0]
    )

    print()
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
        result[
            "anomalies_found"
        ],
    )

    print(
        "Historical stockout days:",
        result[
            "stockout_days"
        ],
    )

    print(
        "Recovered lost demand:",
        result[
            "recovered_lost_demand"
        ],
    )

    print()
    print(
        "Daily forecast:"
    )

    print(
        result[
            "daily_forecast"
        ].head(10)
    )

    print()
    print(
        "Comparing baselines..."
    )

    evaluation = (
        evaluate_baselines(
            data=data,
            test_days=28,
            lookback_days=56,
            weeks=8,
        )
    )

    print()
    print(
        "========== EVALUATION =========="
    )

    print(
        "Simple Mean MAE:",
        round(
            evaluation[
                "simple_mae"
            ],
            2,
        ),
    )

    print(
        "Seasonal Weekday MAE:",
        round(
            evaluation[
                "seasonal_mae"
            ],
            2,
        ),
    )

    print(
        "Improvement:",
        round(
            evaluation[
                "improvement_percent"
            ],
            2,
        ),
        "%",
    )

    print(
        "Seasonal better on:",
        evaluation[
            "seasonal_wins"
        ],
        "/",
        evaluation[
            "total_skus"
        ],
        "SKU",
    )

    print()
    print(
        "Per-SKU comparison:"
    )

    print(
        evaluation[
            "sku_results"
        ]
        .sort_values(
            "seasonal_mae"
        )
        .head(10)
    )


if __name__ == "__main__":
    main()