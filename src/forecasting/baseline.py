from pathlib import Path

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
    Combine complete daily calendar with
    anomaly-adjusted demand.
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

    # If there was no transaction on that day,
    # regular demand is currently treated as zero.
    data["adjusted_demand"] = (
        data["adjusted_demand"]
        .fillna(0.0)
    )

    data["anomalies_found"] = (
        data["anomalies_found"]
        .fillna(0)
        .astype(int)
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
    Very simple baseline.

    Uses the mean adjusted demand from the
    last lookback_days and predicts that same
    value for every future day.
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
        recent["adjusted_demand"].mean()
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

    Forecasts each future weekday using demand
    from the same weekday during recent weeks.

    Example:
    future Monday -> average of recent Mondays.
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
        recent["adjusted_demand"].mean()
    )

    predictions = []

    for future_date in future_dates:
        weekday = future_date.dayofweek

        same_weekday = recent[
            recent["day_of_week"] == weekday
        ]["adjusted_demand"]

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
    Main forecasting function used by the project.

    Contract:
        horizon_days =
        lead_time_days + review_period_days

    Uses seasonal weekday forecasting.
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


def evaluate_baselines(
    data: pd.DataFrame,
    test_days: int = 28,
    lookback_days: int = 56,
    weeks: int = 8,
) -> dict:
    """
    Compare two forecasting approaches:

    1. Simple mean baseline
    2. Seasonal weekday baseline

    Uses time-based backtesting:
    last test_days are hidden as test data.
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
            test["adjusted_demand"]
            .to_numpy()
        )

        simple_pred = (
            simple["predicted_demand"]
            .to_numpy()
        )

        seasonal_pred = (
            seasonal[
                "predicted_demand"
            ]
            .to_numpy()
        )

        simple_mae = mean_absolute_error(
            actual,
            simple_pred,
        )

        seasonal_mae = mean_absolute_error(
            actual,
            seasonal_pred,
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

    comparison = (
        evaluation[
            "sku_results"
        ]
        .sort_values(
            "seasonal_mae"
        )
        .head(10)
    )

    print(
        comparison
    )


if __name__ == "__main__":
    main()