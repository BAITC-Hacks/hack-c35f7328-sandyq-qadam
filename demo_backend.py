"""Run from repository root after preparing UCI demand; warehouse values are demo inputs."""
import json

from src.inventory import get_order_recommendation


if __name__ == "__main__":
    result = get_order_recommendation(
        sku="20685", stock=80, incoming=40, lead_time_days=14,
        review_period_days=7, supplier="Demo_Supplier", safety_factor=.20,
    )
    print("UCI historical sales + SYNTHETIC/demo warehouse inputs; not a current company forecast.")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
