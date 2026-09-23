# Backend: Forecasting → Inventory

Публичная функция для frontend — `src.inventory.get_order_recommendation`.
Она сама вызывает `src.forecasting.model.get_demand_forecast`, считает страховой запас,
заказ и риск, добавляет объяснение и сведения об аномалиях/stockout из Data/ML.
Frontend не должен повторять формулу заказа или умножать прогноз на срок поставки.

## Установка и запуск

Python 3.11; команды из корня командного репозитория:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-backend.txt
python -m pytest -v
```

Для реального исторического сценария распаковать `online_retail_II.xlsx` из Online Retail II в
`data/raw/`. Затем использовать исходные модули Артёма из `dev`:

```bash
python -m src.data.preprocess
python -m src.data.anomaly_detection
python -m src.data.stockout_correction
python demo_backend.py
```

Большие исходные и подготовленные данные не хранятся в Git. Отсутствие
`data/processed/final_demand.csv` вызывает ошибку с инструкцией, а не незаметную
подмену реального прогноза mock-значением. Старый `demo_inventory.py` оставлен
как отдельное явно синтетическое демо механики Inventory.

## Подключение Адели

```python
from src.inventory import get_order_recommendation

recommendation = get_order_recommendation(
    sku="20685",
    stock=80,
    incoming=40,
    lead_time_days=14,
    review_period_days=7,
    supplier="Demo_Supplier",
    safety_factor=0.20,
)
```

Результат — JSON-совместимый dict с полями:

`sku`, `product_name`, `supplier`, `forecast`, `horizon_days`, `stock`, `incoming`,
`lead_time_days`, `review_period_days`, `safety_stock`, `recommended_order`, `risk`,
`anomalies_found`, `stockout_days`, `recovered_lost_demand`, `explanation`.

Дополнительно возвращаются `minimum_order_qty`, `unit_cost`, `order_value`,
`safety_factor`, `safety_stock_method`, `forecast_source`.
Если поставщик не передан, выводится «Не назначен» — сервис не выдумывает поставщика.
Название товара берётся из переданных данных либо явно переданного `product_name`;
если его нет, выводится SKU.

Файл спроса по умолчанию разрешается относительно корня проекта, а не текущей папки запуска.
Для собственной таблицы можно передать `data_path=...` или уже загруженный DataFrame `data=...`.
Обязательные поля истории: `date,sku,final_demand,anomalies_found,stockout_flag,stockout_adjustment`.
SKU нужно читать строками для сохранения ведущих нулей. Одна строка на пару дата/SKU.

Для нескольких товаров:

```python
from src.inventory import get_order_recommendations

rows, supplier_groups = get_order_recommendations([
    dict(sku="20685", stock=80, incoming=40, lead_time_days=14, supplier="Demo_Supplier"),
])
```

`warehouse_rows` — список словарей с аргументами одиночной функции (не произвольная
таблица со служебными колонками). Для DataFrame сначала выбрать эти поля и вызвать
`.to_dict("records")`. `rows` содержит также нулевые заказы; `supplier_groups` —
словарь поставщик → положительные заказы. Пакетный вызов читает файл только один раз.
Отправка заказов поставщикам не реализована и не выполняется.

## Расчёт и проверенный пример

`forecast` — суммарный спрос за `lead_time_days + review_period_days`.
В публичном сервисе `safety_stock = forecast * safety_factor`.
При стандартной кратности 1: `recommended_order = ceil(max(0, forecast + safety_stock - stock - incoming))`.
В существующей функции Inventory параметр `minimum_order_qty` означает кратность
округления вверх, а не отдельную минимальную сумму заказа поставщику.

Для подготовленного UCI SKU 20685 локально проверено:

- прогноз 205.12 за 21 день;
- страховой запас 41.024 при коэффициенте 20%;
- тестовые остаток 80 и incoming 40;
- до округления 126.144, заказ **127**;
- риск HIGH; 1 аномалия, 2 синтетических stockout-дня, восстановленный спрос 20;
- название из данных: DOORMAT RED RETROSPOT.

Категории риска: HIGH, когда `stock + incoming < forecast`; MEDIUM, когда спрос
покрыт, но не страховой запас; LOW, когда покрыт и страховой запас.
Это оценка покрытия суммарного спроса, не вероятность дефицита до конкретной даты.

## Проверки

В локально собранной копии актуальных модулей проверены 57 тестов:
10 исходных Forecasting, 14 исходных Inventory и 33 новых проверки сервиса.
Проверены округление, границы риска, изменение периода пересмотра, перенос метаданных,
группировка, ведущие нули SKU, отсутствие данных, неверные значения и отсутствие мутации входа.
Ни один исходный тест не удалён. Тесты не требуют Excel/API; исторический smoke-сценарий
`demo_backend.py` требует подготовленных данных. Результат smoke-сценария проверен отдельно.

## Происхождение и ограничения данных

- Продажи — исторический UCI Online Retail II; это не реальные продажи Электрокомплекта.
- Остатки, incoming, lead time и поставщики в демо — **синтетические/тестовые**.
- Подготовка Артёма по умолчанию генерирует **синтетическую** stockout-историю с seed=42.
- Локальный smoke запускался на данных, подготовленных по транскрибированным правилам
  модулей `dev/src/data/`; сами Forecasting и Inventory вызваны напрямую. Для повторения
  в репозитории следует запускать оригинальные команды подготовки выше.
- Прогноз относится к периоду после последней исторической даты UCI (2011-12-09), не к сегодняшнему дню.
- 20% — простое MVP-правило запаса, не статистически оценённый уровень сервиса.
- Incoming считается доступным в пределах горизонта; конкретные даты прибытия не моделируются.
- Существующая статистическая ветка `calculate_safety_stock` сохранена для совместимости,
  но публичный сервис не придумывает отсутствующую оценку ошибки и использует процентный запас.
- Качество прогноза, preprocessing, сезонность и коррекция stockout остаются ответственностью Data/ML;
  их алгоритмы в этой доработке не менялись.
- Нет LLM, авторизации, БД, интеграции с 1С и автоматической отправки заказов.
