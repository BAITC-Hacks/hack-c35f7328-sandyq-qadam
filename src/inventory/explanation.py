"""Human-readable arithmetic without an LLM or invented reasons."""


def build_explanation(*, forecast, horizon_days, stock, incoming, safety_stock,
                      recommended_order, risk, minimum_order_qty=1,
                      anomalies_found=None, stockout_days=None, recovered_lost_demand=None):
    raw = forecast + safety_stock - stock - incoming
    message = (
        f"Прогноз на {horizon_days} дней: {forecast:.3f}; остаток: {stock:.3f}; "
        f"в пути: {incoming:.3f}; страховой запас: {safety_stock:.3f}. "
        f"Расчёт: {forecast:.3f} + {safety_stock:.3f} − {stock:.3f} − {incoming:.3f} "
        f"= {raw:.3f}. После ограничения снизу нулём и округления вверх "
        f"до кратности {minimum_order_qty}: заказать {recommended_order} шт. "
        f"Риск покрытия спроса за весь горизонт: {risk}."
    )
    if anomalies_found is not None:
        message += f" Крупных разовых заказов в истории: {anomalies_found}."
    if stockout_days is not None:
        message += f" Отмеченных stockout-дней: {stockout_days}."
    if recovered_lost_demand is not None:
        message += f" Восстановленный спрос в истории: {recovered_lost_demand:.3f}."
    return message
