from pathlib import Path

import numpy as np
import pandas as pd


TRANSACTIONS_PATH = Path(
    "data/processed/transactions_clean.csv"
)

MODELING_DATA_PATH = Path(
    "data/processed/modeling_daily_sales.csv"
)

CUSTOMER_ANOMALIES_PATH = Path(
    "data/processed/customer_order_anomalies.csv"
)

ADJUSTED_DAILY_PATH = Path(
    "data/processed/adjusted_daily_demand.csv"
)


def load_modeling_skus() -> set[str]:
    modeling_data = pd.read_csv(MODELING_DATA_PATH)

    modeling_data["sku"] = (
        modeling_data["sku"]
        .astype(str)
        .str.strip()
    )

    return set(modeling_data["sku"].unique())


def filter_transactions_to_modeling_skus(
    transactions: pd.DataFrame,
    modeling_skus: set[str],
) -> pd.DataFrame:

    df = transactions.copy()

    df["sku"] = (
        df["sku"]
        .astype(str)
        .str.strip()
    )

    return df[
        df["sku"].isin(modeling_skus)
    ].copy()


def create_customer_daily_orders(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    sales = transactions[
        transactions["is_valid_sale"] == True
    ].copy()

    sales["invoice_date"] = pd.to_datetime(
        sales["invoice_date"],
        errors="coerce",
    )

    sales = sales.dropna(
        subset=["invoice_date", "sku"]
    )

    sales["date"] = (
        sales["invoice_date"]
        .dt.floor("D")
    )

    sales["customer_id"] = (
        sales["customer_id"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    customer_daily = (
        sales.groupby(
            ["date", "sku", "customer_id"],
            as_index=False,
        )
        .agg(
            quantity=("quantity", "sum"),
            product_name=("product_name", "first"),
        )
    )

    return customer_daily


def add_sku_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    sku_stats = (
        df.groupby("sku")["quantity"]
        .agg(
            q1=lambda x: x.quantile(0.25),
            median="median",
            q3=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )

    sku_stats["iqr"] = (
        sku_stats["q3"] - sku_stats["q1"]
    )

    return df.merge(
        sku_stats,
        on="sku",
        how="left",
    )


def add_customer_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate historical behaviour of each
    customer for each SKU.
    """

    customer_stats = (
        df.groupby(
            ["sku", "customer_id"]
        )["quantity"]
        .agg(
            customer_order_days="count",
            customer_median="median",
        )
        .reset_index()
    )

    return df.merge(
        customer_stats,
        on=["sku", "customer_id"],
        how="left",
    )


def detect_large_orders(
    customer_daily: pd.DataFrame,
    iqr_multiplier: float = 5.0,
    median_multiplier: float = 8.0,
    customer_multiplier: float = 5.0,
) -> pd.DataFrame:
    """
    Detect strong one-off customer purchases.

    Conditions:
    - known customer;
    - strong SKU-level statistical anomaly;
    - much larger than normal SKU order;
    - unusual for this customer's own history.
    """

    df = add_sku_statistics(
        customer_daily
    )

    df = add_customer_statistics(
        df
    )

    df["iqr_threshold"] = (
        df["q3"]
        + iqr_multiplier * df["iqr"]
    )

    df["median_threshold"] = (
        df["median"]
        * median_multiplier
    )

    df["customer_threshold"] = (
        df["customer_median"]
        * customer_multiplier
    )

    known_customer = (
        df["customer_id"] != "UNKNOWN"
    )

    sku_anomaly = (
        (df["quantity"] > df["iqr_threshold"])
        &
        (
            df["quantity"]
            >= df["median_threshold"]
        )
    )

    # If customer purchased this SKU only once,
    # it is naturally a one-off relationship.
    rare_customer_order = (
        df["customer_order_days"] <= 2
    )

    # For recurring customers require this order
    # to also be extreme relative to their own history.
    unusual_for_customer = (
        df["quantity"]
        >= df["customer_threshold"]
    )

    df["is_large_order"] = (
        known_customer
        & sku_anomaly
        & (
            rare_customer_order
            | unusual_for_customer
        )
    )

    return df


def adjust_large_orders(
    detected: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preserve actual quantity but cap anomaly
    contribution for regular-demand forecasting.
    """

    df = detected.copy()

    # Conservative replacement:
    # use SKU-level upper threshold instead
    # of deleting the sale completely.
    replacement = np.maximum(
        df["q3"],
        df["median"],
    )

    df["adjusted_quantity"] = np.where(
        df["is_large_order"],
        replacement,
        df["quantity"],
    )

    df["adjusted_quantity"] = (
        df["adjusted_quantity"]
        .clip(lower=0)
    )

    return df


def create_adjusted_daily_demand(
    adjusted_orders: pd.DataFrame,
) -> pd.DataFrame:

    return (
        adjusted_orders
        .groupby(
            ["date", "sku", "product_name"],
            as_index=False,
            dropna=False,
        )
        .agg(
            raw_sales=("quantity", "sum"),
            adjusted_demand=(
                "adjusted_quantity",
                "sum",
            ),
            anomalies_found=(
                "is_large_order",
                "sum",
            ),
        )
    )


def main():

    print("Loading cleaned transactions...")

    transactions = pd.read_csv(
        TRANSACTIONS_PATH
    )

    print(
        "All transactions:",
        transactions.shape,
    )

    modeling_skus = load_modeling_skus()

    print(
        "Modeling SKU count:",
        len(modeling_skus),
    )

    transactions = (
        filter_transactions_to_modeling_skus(
            transactions,
            modeling_skus,
        )
    )

    print(
        "Transactions for modeling SKU:",
        transactions.shape,
    )

    print()
    print(
        "Creating customer daily orders..."
    )

    customer_daily = (
        create_customer_daily_orders(
            transactions
        )
    )

    print(
        "Customer daily orders:",
        customer_daily.shape,
    )

    print()
    print(
        "Detecting one-off large orders..."
    )

    detected = detect_large_orders(
        customer_daily,
        iqr_multiplier=5.0,
        median_multiplier=8.0,
        customer_multiplier=5.0,
    )

    adjusted = adjust_large_orders(
        detected
    )

    adjusted_daily = (
        create_adjusted_daily_demand(
            adjusted
        )
    )

    detected.to_csv(
        CUSTOMER_ANOMALIES_PATH,
        index=False,
    )

    adjusted_daily.to_csv(
        ADJUSTED_DAILY_PATH,
        index=False,
    )

    anomalies = detected[
        detected["is_large_order"]
    ].copy()

    anomaly_count = len(anomalies)

    affected_skus = (
        anomalies["sku"].nunique()
    )

    anomaly_rate = (
        anomaly_count
        / len(detected)
        * 100
        if len(detected)
        else 0
    )

    print()
    print(
        "========== SUMMARY =========="
    )

    print(
        "Customer daily orders:",
        len(detected),
    )

    print(
        "Large one-off orders detected:",
        anomaly_count,
    )

    print(
        "Affected SKU:",
        affected_skus,
    )

    print(
        "Anomaly rate:",
        round(anomaly_rate, 2),
        "%",
    )

    print()
    print(
        "Largest detected anomalies:"
    )

    if not anomalies.empty:

        largest = (
            anomalies
            .sort_values(
                "quantity",
                ascending=False,
            )
        )

        print(
            largest[
                [
                    "date",
                    "sku",
                    "customer_id",
                    "quantity",
                    "median",
                    "customer_median",
                    "customer_order_days",
                    "iqr_threshold",
                ]
            ].head(10)
        )

    else:
        print(
            "No anomalies detected."
        )

    print()
    print(
        "Saved:",
        CUSTOMER_ANOMALIES_PATH,
    )

    print(
        "Saved:",
        ADJUSTED_DAILY_PATH,
    )


if __name__ == "__main__":
    main()