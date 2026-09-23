"""Run from repository root: python demo_inventory.py."""
import json
from pathlib import Path

import pandas as pd

from src.inventory import run_inventory_pipeline


def main():
    root = Path(__file__).resolve().parent
    inventory = pd.read_csv(root / "data/synthetic/inventory.csv", dtype={"sku": str})
    forecasts = pd.read_csv(root / "data/synthetic/mock_forecasts.csv", dtype={"sku": str})
    orders, groups = run_inventory_pipeline(inventory, forecasts)
    print(orders[["sku", "supplier", "forecast", "stock", "incoming", "recommended_order", "risk"]].to_string(index=False))
    print(json.dumps(groups, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
