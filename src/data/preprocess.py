from pathlib import Path

import pandas as pd

from src.data.load_data import load_retail_data


RAW_PATH = Path("data/raw/online_retail_II.xlsx")
PROCESSED_DIR = Path("data/processed")


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Приводим названия колонок к удобному виду
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

    # Убираем строки без SKU
    df = df.dropna(subset=["sku"])

    # Дата
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df = df.dropna(subset=["invoice_date"])

    # Строковые поля
    df["sku"] = df["sku"].astype(str).str.strip()
    df["invoice"] = df["invoice"].astype(str).str.strip()

    # Отмена заказа: invoice начинается с C
    df["is_cancelled"] = df["invoice"].str.upper().str.startswith("C")

    # Возврат / отрицательное количество
    df["is_return"] = df["quantity"] < 0

    # Для forecasting оставляем только реальные положительные продажи
    df["is_valid_sale"] = (
        (~df["is_cancelled"])
        & (~df["is_return"])
        & (df["quantity"] > 0)
        & (df["price"] >= 0)
    )

    return df


def create_daily_sales(transactions: pd.DataFrame) -> pd.DataFrame:
    sales = transactions[transactions["is_valid_sale"]].copy()

    # Нам для прогноза нужна дата без времени
    sales["date"] = sales["invoice_date"].dt.floor("D")

    daily = (
        sales.groupby(
            ["date", "sku", "product_name"],
            as_index=False,
            dropna=False,
        )["quantity"]
        .sum()
        .rename(columns={"quantity": "daily_sales"})
    )

    return daily


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_retail_data(RAW_PATH)

    transactions = preprocess_transactions(raw)
    daily_sales = create_daily_sales(transactions)

    transactions_path = PROCESSED_DIR / "transactions_clean.csv"
    daily_sales_path = PROCESSED_DIR / "daily_sales.csv"

    transactions.to_csv(transactions_path, index=False)
    daily_sales.to_csv(daily_sales_path, index=False)

    print("Transactions:", transactions.shape)
    print("Daily sales:", daily_sales.shape)

    print()
    print("Valid sales:", transactions["is_valid_sale"].sum())
    print("Cancelled:", transactions["is_cancelled"].sum())
    print("Returns:", transactions["is_return"].sum())

    print()
    print("Unique SKU:", daily_sales["sku"].nunique())
    print("Date range:")
    print(daily_sales["date"].min(), "->", daily_sales["date"].max())

    print()
    print(daily_sales.head())


if __name__ == "__main__":
    main()