from pathlib import Path

import pandas as pd

from src.data.load_data import load_retail_data


RAW_PATH = Path("data/raw/online_retail_II.xlsx")
PROCESSED_DIR = Path("data/processed")


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Online Retail II transactions.

    Keeps cancellations and returns in the dataset as flags,
    but marks only positive normal sales as valid for forecasting.
    """

    df = df.copy()

    # Rename columns to convenient snake_case names
    df = df.rename(
        columns={
            "Invoice": "invoice",
            "StockCode": "sku",
            "Description": "product_name",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "Price": "price",
            "Customer ID": "customer_id",
            "Country": "country",
        }
    )

    # Remove rows without SKU
    df = df.dropna(subset=["sku"])

    # Convert invoice date
    df["invoice_date"] = pd.to_datetime(
        df["invoice_date"],
        errors="coerce",
    )

    # Remove rows with invalid dates
    df = df.dropna(subset=["invoice_date"])

    # Normalize string columns
    df["sku"] = df["sku"].astype(str).str.strip()
    df["invoice"] = df["invoice"].astype(str).str.strip()

    # Cancellation:
    # UCI documentation says invoices starting with C are cancelled
    df["is_cancelled"] = (
        df["invoice"]
        .str.upper()
        .str.startswith("C")
    )

    # Negative quantity means return
    df["is_return"] = df["quantity"] < 0

    # Valid sale for demand forecasting
    df["is_valid_sale"] = (
        (~df["is_cancelled"])
        & (~df["is_return"])
        & (df["quantity"] > 0)
        & (df["price"] >= 0)
    )

    return df


def create_daily_sales(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate valid transactions into daily sales by SKU.
    """

    sales = transactions[
        transactions["is_valid_sale"]
    ].copy()

    # Remove time component
    sales["date"] = (
        sales["invoice_date"]
        .dt.floor("D")
    )

    daily = (
        sales.groupby(
            ["date", "sku", "product_name"],
            as_index=False,
            dropna=False,
        )["quantity"]
        .sum()
        .rename(
            columns={
                "quantity": "daily_sales"
            }
        )
    )

    return daily


def select_active_skus(
    daily_sales: pd.DataFrame,
    top_n: int = 50,
    min_active_days: int = 60,
) -> pd.DataFrame:
    """
    Select SKUs that have enough history
    to be useful for forecasting.
    """

    sku_stats = (
        daily_sales.groupby("sku")
        .agg(
            active_days=(
                "daily_sales",
                lambda x: (x > 0).sum(),
            ),
            total_sales=(
                "daily_sales",
                "sum",
            ),
            first_date=(
                "date",
                "min",
            ),
            last_date=(
                "date",
                "max",
            ),
        )
        .reset_index()
    )

    # Keep SKUs with enough active days
    eligible = sku_stats[
        sku_stats["active_days"] >= min_active_days
    ].copy()

    # Prefer SKUs with more history and sales
    eligible = (
        eligible.sort_values(
            [
                "active_days",
                "total_sales",
            ],
            ascending=False,
        )
        .head(top_n)
    )

    selected_skus = eligible["sku"]

    selected = daily_sales[
        daily_sales["sku"].isin(
            selected_skus
        )
    ].copy()

    print()
    print("Selected SKU:", len(selected_skus))

    print()
    print("Top selected SKU:")
    print(
        eligible[
            [
                "sku",
                "active_days",
                "total_sales",
                "first_date",
                "last_date",
            ]
        ].head(10)
    )

    return selected


def create_complete_daily_series(
    daily_sales: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build continuous daily time series for every SKU.

    Missing dates are filled with zero sales.
    """

    result = []

    for sku, group in daily_sales.groupby("sku"):
        group = group.copy()
        group = group.sort_values("date")

        start_date = group["date"].min()
        end_date = group["date"].max()

        # Full calendar for this SKU
        full_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq="D",
        )

        # Choose most common product name
        product_names = (
            group["product_name"]
            .dropna()
        )

        if not product_names.empty:
            modes = product_names.mode()

            if not modes.empty:
                product_name = modes.iloc[0]
            else:
                product_name = product_names.iloc[0]

        else:
            product_name = "Unknown"

        # Sum sales per day again just in case
        sku_daily = (
            group.groupby("date")[
                "daily_sales"
            ]
            .sum()
        )

        # Add missing dates as zero
        sku_daily = sku_daily.reindex(
            full_dates,
            fill_value=0,
        )

        sku_daily = (
            sku_daily
            .rename_axis("date")
            .reset_index()
        )

        sku_daily["sku"] = sku
        sku_daily["product_name"] = product_name

        result.append(sku_daily)

    complete = pd.concat(
        result,
        ignore_index=True,
    )

    complete = complete[
        [
            "date",
            "sku",
            "product_name",
            "daily_sales",
        ]
    ]

    return complete


def main():
    print("Loading raw dataset...")

    raw = load_retail_data(
        RAW_PATH
    )

    print(
        "Raw dataset:",
        raw.shape,
    )

    print()
    print("Preprocessing transactions...")

    transactions = preprocess_transactions(
        raw
    )

    print()
    print("Creating daily sales...")

    daily_sales = create_daily_sales(
        transactions
    )

    print()
    print("Selecting active SKU...")

    selected_sales = select_active_skus(
        daily_sales,
        top_n=50,
        min_active_days=60,
    )

    print()
    print(
        "Creating complete daily series..."
    )

    modeling_data = (
        create_complete_daily_series(
            selected_sales
        )
    )

    # Create output directory
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    transactions_path = (
        PROCESSED_DIR
        / "transactions_clean.csv"
    )

    daily_sales_path = (
        PROCESSED_DIR
        / "daily_sales.csv"
    )

    modeling_path = (
        PROCESSED_DIR
        / "modeling_daily_sales.csv"
    )

    print()
    print("Saving processed files...")

    transactions.to_csv(
        transactions_path,
        index=False,
    )

    daily_sales.to_csv(
        daily_sales_path,
        index=False,
    )

    modeling_data.to_csv(
        modeling_path,
        index=False,
    )

    # Statistics
    print()
    print("========== SUMMARY ==========")

    print(
        "Transactions:",
        transactions.shape,
    )

    print(
        "Daily sales:",
        daily_sales.shape,
    )

    print(
        "Modeling data:",
        modeling_data.shape,
    )

    print()

    print(
        "Valid sales:",
        int(
            transactions[
                "is_valid_sale"
            ].sum()
        ),
    )

    print(
        "Cancelled:",
        int(
            transactions[
                "is_cancelled"
            ].sum()
        ),
    )

    print(
        "Returns:",
        int(
            transactions[
                "is_return"
            ].sum()
        ),
    )

    print()

    print(
        "Unique SKU in all daily sales:",
        daily_sales["sku"].nunique(),
    )

    print(
        "Modeling SKU:",
        modeling_data["sku"].nunique(),
    )

    print(
        "Zero-sale days:",
        int(
            (
                modeling_data[
                    "daily_sales"
                ] == 0
            ).sum()
        ),
    )

    print()

    print(
        "Date range:",
        modeling_data["date"].min(),
        "->",
        modeling_data["date"].max(),
    )

    print()
    print("Example:")
    print(
        modeling_data.head(10)
    )

    print()
    print(
        "Saved:",
        transactions_path,
    )

    print(
        "Saved:",
        daily_sales_path,
    )

    print(
        "Saved:",
        modeling_path,
    )


if __name__ == "__main__":
    main()