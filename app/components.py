import pandas as pd
import streamlit as st

RISK_LABELS = {"HIGH": "Высокий", "MEDIUM": "Средний", "LOW": "Низкий"}


def create_mock_history(item):
    """Fixed synthetic chart, spanning the complete item horizon."""
    dates = pd.date_range(end="2026-09-22", periods=14)
    sales = [8, 10, 7, 12, 11, 14, 9, 13, 15, 12, 16, 18, 14, 17]
    history = pd.DataFrame({"Дата": dates, "Спрос": sales})
    horizon = int(item["horizon_days"])
    forecast = pd.DataFrame({
        "Дата": pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=horizon),
        "Прогноз": [item["forecast"] / horizon] * horizon,
    })
    return history, forecast


def show_demand_chart(item):
    st.subheader("История спроса и прогноз")
    if item.get("data_source") == "demo":
        st.caption("Демонстрационный график: синтетическая история и условный прогноз, не реальные продажи.")
        history, forecast = create_mock_history(item)
        chart = history.set_index("Дата").join(forecast.set_index("Дата"), how="outer")
        st.line_chart(chart, color=["#64748b", "#0d9488"])
    else:
        history = pd.DataFrame(item.get("history", []))
        if not history.empty:
            history["date"] = pd.to_datetime(history["date"])
            chart = history.set_index("date")[["demand"]].rename(columns={"demand": "Скорректированный спрос"})
            st.line_chart(chart, color="#0d9488")
            anomalies = history.loc[history["anomaly"]]
            if not anomalies.empty:
                st.caption("Даты обнаруженных аномалий в показанной истории")
                st.dataframe(anomalies[["date", "demand"]], hide_index=True)
        st.caption(
            f"Суммарный прогноз: {item['forecast']:g} ед. за {int(item['horizon_days'])} дней. "
            "Дневной прогноз не передан публичным API модели; график не подменяется демо."
        )
    if item.get("anomalies_found", 0):
        st.warning(f"Исторических аномалий: {int(item['anomalies_found'])}")
