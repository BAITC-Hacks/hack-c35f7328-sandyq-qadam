from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def group_orders_by_supplier(orders: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group positive recommendations into separate supplier order batches."""
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for order in orders:
        if int(order.get("recommended_order", 0)) > 0:
            grouped[str(order["supplier"])].append(dict(order))
    return {
        supplier: sorted(items, key=lambda item: str(item["sku"]))
        for supplier, items in sorted(grouped.items())
    }

