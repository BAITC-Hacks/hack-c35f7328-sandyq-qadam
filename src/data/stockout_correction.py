from pathlib import Path

import numpy as np
import pandas as pd


MODELING_DATA_PATH = Path(
    "data/processed/modeling_daily_sales.csv"
)

ADJUSTED_DEMAND_PATH = Path(
    "data/processed/adjusted_daily_demand.csv"
)

MOCK_STOCK_PATH = Path(
    "data/processed/mock_stock_history.csv"
)

FINAL_DEMAND_PATH = Path(
    "data/processed/final_demand.csv"
)


def load_base_demand() -> pd.DataFrame:
    """
    Build complete daily demand history.

    modeling_daily_sales gives us the full calendar.
    adjusted_daily_demand contains demand after
    one-off large order correction.
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

    # No transaction on a day currently means observed demand = 0
    data["adjusted_demand"] = (
        data["adjusted_demand"]
        .fillna(0.0)
    )

    data["anomalies_found"] = (
        data["anomalies_found"]
        .fillna(0)
        .astype(int)
    )

    return (
        data.sort_values(
            ["sku", "date"]
        )
        .reset_index(drop=True)
    )


def generate_mock_stock_history(
    demand: pd.DataFrame,
    stockout_rate: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate reproducible synthetic historical stock data.

    For MVP purposes we simulate stockouts only
    among zero-sale days.

    This lets us demonstrate that zero sales during
    stockout should not be treated as zero demand.

    Later this DataFrame can be replaced by real
    or externally generated stock history.
    """

    rng = np.random.default_rng(seed)

    stock = demand[
        ["date", "sku", "adjusted_demand"]
    ].copy()

    # Normal positive stock level.
    stock["stock"] = 100.0

    # We only simulate stockout among zero-demand days.
    candidates = stock.index[
        stock["adjusted_demand"] == 0
    ].to_numpy()

    number_of_stockouts = int(
        len(candidates) * stockout_rate
    )

    if number_of_stockouts > 0:
        selected = rng.choice(
            candidates,
            size=number_of_stockouts,
            replace=False,
        )

        stock.loc[
            selected,
            "stock",
        ] = 0.0

    return stock[
        [
            "date",
            "sku",
            "stock",
        ]
    ]


def estimate_lost_demand(
    sku_history: pd.DataFrame,
    row_index: int,
    same_weekday_weeks: int = 8,
    fallback_days: int = 28,
) -> float:
    """
    Estimate demand for a stockout day.

    First preference:
        median demand from previous same weekdays.

    Example:
        stockout Monday -> previous Mondays.

    Fallback:
        median positive demand from recent days.
    """

    current_row = sku_history.loc[
        row_index
    ]

    current_date = current_row["date"]
    weekday = current_date.dayofweek

    previous = sku_history[
        sku_history["date"] < current_date
    ].copy()

    # Do not learn from previous known stockout days.
    previous = previous[
        ~previous["stockout_flag"]
    ]

    if previous.empty:
        return 0.0

    same_weekday_start = (
        current_date
        - pd.Timedelta(
            days=same_weekday_weeks * 7
        )
    )

    same_weekday = previous[
        (
            previous["date"]
            >= same_weekday_start
        )
        &
        (
            previous["date"].dt.dayofweek
            == weekday
        )
        &
        (
            previous["adjusted_demand"]
            > 0
        )
    ]

    if not same_weekday.empty:
        estimate = float(
            same_weekday[
                "adjusted_demand"
            ].median()
        )

        return max(
            0.0,
            estimate,
        )

    # Fallback: recent positive-demand days.
    fallback_start = (
        current_date
        - pd.Timedelta(
            days=fallback_days
        )
    )

    recent = previous[
        (
            previous["date"]
            >= fallback_start
        )
        &
        (
            previous["adjusted_demand"]
            > 0
        )
    ]

    if not recent.empty:
        estimate = float(
            recent[
                "adjusted_demand"
            ].median()
        )

        return max(
            0.0,
            estimate,
        )

    return 0.0


def correct_stockouts(
    demand: pd.DataFrame,
    stock_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Correct zero observed demand on known stockout days.

    Important:
    We preserve adjusted_demand as the observed value.

    final_demand is the demand used for forecasting
    after one-off order and stockout corrections.
    """

    data = demand.merge(
        stock_history,
        on=["date", "sku"],
        how="left",
    )

    data["stock"] = (
        data["stock"]
        .fillna(100.0)
    )

    data["stockout_flag"] = (
        data["stock"] <= 0
    )

    data["stockout_adjustment"] = 0.0

    corrected_groups = []

    for sku, group in data.groupby("sku"):
        group = (
            group.sort_values("date")
            .reset_index(drop=True)
            .copy()
        )

        for index in group.index:
            is_stockout = bool(
                group.loc[
                    index,
                    "stockout_flag",
                ]
            )

            observed_demand = float(
                group.loc[
                    index,
                    "adjusted_demand",
                ]
            )

            # For the MVP we correct zero-sales stockout days.
            if (
                is_stockout
                and observed_demand == 0
            ):
                estimate = estimate_lost_demand(
                    sku_history=group,
                    row_index=index,
                )

                group.loc[
                    index,
                    "stockout_adjustment",
                ] = estimate

        group["final_demand"] = (
            group["adjusted_demand"]
            + group["stockout_adjustment"]
        )

        corrected_groups.append(
            group
        )

    result = pd.concat(
        corrected_groups,
        ignore_index=True,
    )

    return (
        result.sort_values(
            ["sku", "date"]
        )
        .reset_index(drop=True)
    )


def main():
    print(
        "Loading anomaly-adjusted demand..."
    )

    demand = load_base_demand()

    print(
        "Demand rows:",
        demand.shape,
    )

    print(
        "SKU:",
        demand["sku"].nunique(),
    )

    print()
    print(
        "Generating mock stock history..."
    )

    stock_history = (
        generate_mock_stock_history(
            demand=demand,
            stockout_rate=0.03,
            seed=42,
        )
    )

    stock_history.to_csv(
        MOCK_STOCK_PATH,
        index=False,
    )

    print()
    print(
        "Applying stockout correction..."
    )

    corrected = correct_stockouts(
        demand=demand,
        stock_history=stock_history,
    )

    corrected.to_csv(
        FINAL_DEMAND_PATH,
        index=False,
    )

    stockout_days = int(
        corrected[
            "stockout_flag"
        ].sum()
    )

    corrected_days = int(
        (
            corrected[
                "stockout_adjustment"
            ] > 0
        ).sum()
    )

    raw_total = float(
        corrected[
            "adjusted_demand"
        ].sum()
    )

    final_total = float(
        corrected[
            "final_demand"
        ].sum()
    )

    recovered_demand = (
        final_total
        - raw_total
    )

    print()
    print(
        "========== SUMMARY =========="
    )

    print(
        "Rows:",
        len(corrected),
    )

    print(
        "SKU:",
        corrected["sku"].nunique(),
    )

    print(
        "Stockout days:",
        stockout_days,
    )

    print(
        "Corrected stockout days:",
        corrected_days,
    )

    print(
        "Demand before stockout correction:",
        round(
            raw_total,
            2,
        ),
    )

    print(
        "Demand after stockout correction:",
        round(
            final_total,
            2,
        ),
    )

    print(
        "Recovered lost demand:",
        round(
            recovered_demand,
            2,
        ),
    )

    print()
    print(
        "Example corrected stockouts:"
    )

    examples = corrected[
        corrected[
            "stockout_adjustment"
        ] > 0
    ][
        [
            "date",
            "sku",
            "adjusted_demand",
            "stock",
            "stockout_adjustment",
            "final_demand",
        ]
    ].head(10)

    print(
        examples
    )

    print()
    print(
        "Saved:",
        MOCK_STOCK_PATH,
    )

    print(
        "Saved:",
        FINAL_DEMAND_PATH,
    )


if __name__ == "__main__":
    main()