# Inventory — участник №2

Рабочая зона: `src/inventory/`, `data/synthetic/`, `tests/test_inventory.py`. Модуль получает готовый прогноз и возвращает рекомендации заказов с объяснениями и группировкой по поставщикам.

## Запуск

Python 3.11. Из корня командного репозитория:

```bash
python -m pip install -r requirements-inventory.txt
python -m pytest tests/test_inventory.py -q
python demo_inventory.py
```

14 unit-тестов проверены локально. Демо использует фиксированные синтетические остатки и mock-прогнозы; не требует API-ключей или UCI-файла.

## Для Артёма и Адели

```python
import pandas as pd
from src.inventory import calculate_order, run_inventory_pipeline

example = calculate_order(
    sku="85123A", supplier="Supplier_A", forecast=120,
    stock=40, incoming=30, lead_time_days=14, safety_stock=20,
)
assert example["recommended_order"] == 70

inventory = pd.read_csv("data/synthetic/inventory.csv", dtype={"sku": str})
forecasts = pd.read_csv("data/synthetic/mock_forecasts.csv", dtype={"sku": str})
orders, supplier_groups = run_inventory_pipeline(inventory, forecasts)
```

`orders` — DataFrame для интерфейса, `supplier_groups` — словарь поставщик → список положительных заказов. Нулевые заказы остаются в таблице, но не отправляются в группы.

Контракт Data/ML: `sku` (строка), `forecast`, `forecast_error_std`; желательно обязательно передавать `horizon_days`, чтобы pipeline проверил горизонт. `forecast` — уже суммарный спрос за `lead_time_days + 7`; повторно на lead time не умножается. Горизонт индивидуален для товара. Отсутствие прогноза и дубликаты SKU вызывают ошибку. В demo-файле горизонты опущены, значения mock предназначены для проверки механики.

Склад: `sku,stock,incoming,lead_time_days,supplier,minimum_order_qty,unit_cost`; `category` передаётся в результат как метаданные. Генератор `build_synthetic_inventory(seed=42, skus=[...])` позволяет согласовать товары с Data/ML.

## Расчёт и ограничения MVP

Заказ = `max(0, forecast + safety_stock - stock - incoming)`, с округлением вверх до кратности партии. В этой версии `minimum_order_qty` интерпретируется как кратность партии, а не только нижний порог заказа.

Safety stock передаётся отдельно в `calculate_order`; pipeline считает его как `1.65 × forecast_error_std × sqrt(lead_time_days)` при положительной ошибке, иначе 20% от прогноза. `forecast_error_std` должен описывать ежедневные ошибки; это MVP-приближение с допущением независимости дней. Уровень сервиса не гарантируется, дисперсия за review period отдельно не учитывается.

HIGH: `stock + incoming < forecast`. MEDIUM: спрос покрыт, страховой запас не покрыт. LOW: покрыт весь целевой запас. Это риск покрытия горизонта, не прогноз даты дефицита. Считается, что incoming поступит вовремя; расписание отдельных поставок отсутствует. Отрицательные числовые входы в прямом калькуляторе ограничиваются нулём; строгая валидация всех некорректных входов ещё не реализована.

Stockout correction и обработка продаж остаются у Data/ML. Существующий пустой scaffold `src/inventory/stockout.py` не используется. Эта часть не реализует ML, frontend, 1С и отправку заказов. Подтверждать реальную закупку должен сотрудник.
