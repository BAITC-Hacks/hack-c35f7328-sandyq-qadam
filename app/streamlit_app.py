"""Adelya's dashboard: display recommendations, never calculate orders in UI."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from app.components import RISK_LABELS, show_demand_chart
from src.pipeline import get_recommendations

st.set_page_config(page_title="Sandyq · Закупки", page_icon="📦", layout="wide")
st.caption("SANDYQ QADAM / PROCUREMENT ASSISTANT")
st.title("Заказать вовремя. Не хранить лишнее.")
st.caption("Рекомендации по пополнению склада с объяснением по каждой позиции")

with st.sidebar:
    st.header("Источник данных")
    mode_label = st.radio("Режим работы", ["Демонстрация", "Расчёт по данным"], key="mode")
    st.caption("Демо показывает интерфейс. Расчёт требует готовой истории спроса и складских данных.")
    st.button("Обновить расчёт")
    st.divider()
    st.caption("Заказы не отправляются автоматически. Рекомендации проверяет менеджер.")

mode = "demo" if mode_label == "Демонстрация" else "live"
if mode == "demo":
    st.info("ДЕМО · Все позиции и графики демонстрационные. Прогнозная модель здесь не запускается.")
else:
    st.info("Расчёт: модель Артёма → Inventory. Остатки из data/synthetic/inventory.csv синтетические.")
try:
    with st.spinner("Подготавливаем рекомендации…"):
        records = get_recommendations(mode=mode)
except (ValueError, ImportError, OSError, KeyError) as exc:
    st.error(f"Расчёт недоступен: {exc}")
    st.caption("Для просмотра экрана выберите «Демонстрация». Данные не подменяются автоматически.")
    st.stop()

if not records:
    st.info("Нет товаров для расчёта. Добавьте складские позиции.")
    st.stop()
df = pd.DataFrame(records)
st.caption("Сводка по всем загруженным товарам, до применения фильтров")
cols = st.columns(4)
cols[0].metric("Товарных позиций", len(df))
cols[1].metric("Высокий риск", int(df["risk"].eq("HIGH").sum()))
cols[2].metric("К заказу, ед.", f"{df['recommended_order'].sum():g}")
cols[3].metric("Исторических аномалий", int(df["anomalies_found"].sum()))
st.divider()

left, right = st.columns(2)
supplier = left.selectbox("Поставщик", ["Все"] + sorted(df["supplier"].unique()), key="supplier")
risk = right.selectbox("Риск", ["Все", "HIGH", "MEDIUM", "LOW"],
                       format_func=lambda value: RISK_LABELS.get(value, value), key="risk")
filtered = df.copy()
if supplier != "Все":
    filtered = filtered.loc[filtered["supplier"].eq(supplier)]
if risk != "Все":
    filtered = filtered.loc[filtered["risk"].eq(risk)]

labels = {"sku": "SKU", "product_name": "Товар", "supplier": "Поставщик",
          "stock": "На складе", "incoming": "В пути", "forecast": "Прогноз спроса",
          "recommended_order": "К заказу", "risk": "Риск"}
st.subheader("Рекомендации по закупкам")
st.caption(f"Показано {len(filtered)} из {len(df)} позиций. Прогноз указан за индивидуальный горизонт каждого товара.")
display = filtered[list(labels)].copy()
display["risk"] = display["risk"].map(RISK_LABELS)
st.dataframe(display.rename(columns=labels), use_container_width=True, hide_index=True)
if filtered.empty:
    st.warning("По выбранным фильтрам товары не найдены. Измените поставщика или риск.")
    st.stop()

details_tab, suppliers_tab = st.tabs(["Карточка товара", "Заказы по поставщикам"])
with details_tab:
    selected_sku = st.selectbox("Выберите SKU", filtered["sku"].tolist(), key="sku")
    item = filtered.loc[filtered["sku"].eq(selected_sku)].iloc[0].to_dict()
    st.subheader(f"{item['sku']} · {item['product_name']}")
    st.caption(f"{item['supplier']} · Риск: {RISK_LABELS[item['risk']]} · Горизонт: {item['horizon_days']} дней")
    cols = st.columns(4)
    for col, (label, key) in zip(cols, [("Прогноз", "forecast"), ("На складе", "stock"),
                                       ("В пути", "incoming"), ("К заказу", "recommended_order")]):
        col.metric(label, f"{item[key]:g}")
    cols = st.columns(3)
    cols[0].metric("Страховой запас", f"{item['safety_stock']:g}")
    cols[1].metric("Срок поставки, дней", item["lead_time_days"])
    cols[2].metric("Период пересмотра, дней", item["review_period_days"])
    st.markdown("#### Почему такая рекомендация?")
    st.info(item["explanation"])
    st.caption(
        f"Исторических дней без товара: {item.get('stockout_days', 0)} · "
        f"Восстановленный спрос: {item.get('recovered_lost_demand', 0):g} ед."
    )
    show_demand_chart(item)

with suppliers_tab:
    st.caption("Только положительные заказы с учётом выбранных фильтров. Отправка поставщику не выполняется.")
    positive = filtered.loc[filtered["recommended_order"] > 0]
    if positive.empty:
        st.success("Для выбранных товаров пополнение не требуется.")
    for name, group in positive.groupby("supplier", sort=True):
        with st.expander(f"{name} · {len(group)} позиций · {group['recommended_order'].sum():g} ед.", expanded=True):
            st.dataframe(group[["sku", "product_name", "recommended_order"]].rename(columns=labels),
                         hide_index=True, use_container_width=True)
