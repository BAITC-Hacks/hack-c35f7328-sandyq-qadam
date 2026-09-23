from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_SKUS = ["85123A", "71053", "84406B", "84029G", "22752", "21730", "22423", "85099B"]
SUPPLIERS = ["Supplier_A", "Supplier_B", "Supplier_C"]
CATEGORIES = ["Home", "Lighting", "Electrical", "Decor"]


def build_synthetic_inventory(seed: int = 42, skus: list[str] | None = None) -> pd.DataFrame:
    """Create deterministic warehouse inputs while Data/ML is developed in parallel."""
    rng = np.random.default_rng(seed)
    skus = skus or DEFAULT_SKUS
    rows = []
    for index, sku in enumerate(skus):
        rows.append(
            {
                "sku": sku,
                "stock": int(rng.integers(20, 180)),
                "incoming": int(rng.choice([0, 0, 20, 30, 50])),
                "lead_time_days": int(rng.choice([7, 14, 21, 30])),
                "supplier": SUPPLIERS[index % len(SUPPLIERS)],
                "category": CATEGORIES[index % len(CATEGORIES)],
                "minimum_order_qty": int(rng.choice([1, 5, 10, 20])),
                "unit_cost": float(rng.integers(500, 15000)),
            }
        )
    return pd.DataFrame(rows)

