"""Deterministic coverage categories, not calibrated stockout probabilities."""


def calculate_risk(forecast: float, stock: float, incoming: float, safety_stock: float) -> str:
    position = stock + incoming
    if position < forecast:
        return "HIGH"
    if position < forecast + safety_stock:
        return "MEDIUM"
    return "LOW"
