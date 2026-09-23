from pathlib import Path
import pandas as pd


def load_retail_data(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    sheets = pd.read_excel(
        file_path,
        sheet_name=None,
        engine="openpyxl",
    )

    dataframes = []

    for sheet_name, df in sheets.items():
        df = df.copy()
        df["source_sheet"] = sheet_name
        dataframes.append(df)

    combined = pd.concat(dataframes, ignore_index=True)

    return combined


if __name__ == "__main__":
    path = "data/raw/online_retail_II.xlsx"

    df = load_retail_data(path)

    print(df.head())
    print()
    print("Shape:", df.shape)
    print()
    print("Columns:")
    print(df.columns.tolist())