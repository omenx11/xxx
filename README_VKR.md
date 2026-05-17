# ВКР: Прогнозирование ВРП на душу населения регионов РФ

**Тема:** Разработка алгоритмов машинного обучения для прогнозирования показателей социально-экономического развития.

**Объект:** 85 субъектов Российской Федерации.
**Период:** 2001–2023 гг.
**Целевая переменная:** ВРП на душу населения (номинальный + реальный в ценах 2015 года).

---

## Структура проекта (новый код в `src_vkr/`, `scripts/`, `app_vkr/`, `notebooks_vkr/`)

```
grp_vkr/
├── data/
│   ├── raw/
│   │   ├── data_regions_collection_102_v20260313.parquet   (Росстат)
│   │   ├── macro_external.csv                              (макрофакторы)
│   │   ├── region_centroids.json
│   │   ├── region_to_fo.json
│   │   └── federal_districts.json
│   └── processed/
│       ├── panel_preprocessed.parquet                      (чистая wide-панель)
│       ├── panel_features.parquet                          (+ лаги/rolling/yoy)
│       ├── feature_catalog.csv
│       ├── selected_features.csv                           (~20 фичей)
│       ├── feature_importance.csv                          (gain + permutation)
│       ├── feature_drop_audit.csv
│       └── leak_audit.csv                                  (аудит утечек)
├── src_vkr/                                                ПЕРЕСОБРАННЫЕ МОДУЛИ
│   ├── config.py            — единый источник констант
│   ├── typology.py          — экспертная типология регионов
│   ├── preprocess.py        — Росстат long → wide + дефлятор 2015
│   ├── features.py          — лаги, rolling, growth, контроль утечек
│   ├── selection.py         — отбор признаков (corr + VIF + ML)
│   ├── metrics.py           — MAE/RMSE/MAPE/sMAPE/WAPE/R²/RMSLE
│   ├── models.py            — баззлайны + 10 ML-моделей + walk-forward CV
│   ├── grid.py              — GridSearch на TimeSeriesSplit
│   ├── shap_analysis.py     — SHAP global / per_cluster / per_region
│   ├── forecast.py          — рекурсивный сценарный прогноз 1–5 лет
│   └── viz.py               — фигуры + карта РФ (Plotly)
├── scripts/
│   ├── run_pipeline.py      — единая команда: 9 шагов
│   └── make_notebook.py     — генератор Jupyter-ноутбука
├── models/                  — обученные финальные бандлы (pkl)
├── results/
│   ├── model_summary.csv               — walk-forward CV
│   ├── model_summary_nominal.csv       — для номинального ВРП
│   ├── model_summary_real.csv          — для реального ВРП
│   ├── per_year_metrics.csv            — динамика по годам
│   ├── oof_predictions.csv             — out-of-fold предсказания
│   ├── grid_search_results.csv         — лучшие гиперпараметры
│   ├── forecast_all_scenarios.csv      — 85 рег × 5 лет × 3 сценария × ном/реал
│   ├── shap_global.csv / shap_per_cluster.csv / shap_per_region.csv
│   └── log_residuals.npy               — для 90% CI
├── reports/
│   ├── figures/             — PNG-графики для текста ВКР
│   ├── tables/              — CSV-таблицы
│   └── maps/                — интерактивные карты (HTML)
├── notebooks_vkr/
│   └── 01_vkr_main.ipynb    — исследовательский ноутбук на 22 раздела ВКР
└── app_vkr/
    └── streamlit_app.py     — прототип аналитической системы
```

---

## Быстрый старт

### 1. Установка зависимостей

```powershell
py -3.12 -m venv C:\venvs\grp_vkr
C:\venvs\grp_vkr\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Полный пайплайн

```powershell
python scripts/run_pipeline.py
```

Это запускает 9 шагов последовательно (≈ 15–25 минут):
1. **Preprocess** — `panel_preprocessed.parquet` (~10 с)
2. **Feature engineering** — `panel_features.parquet` (~5 с)
3. **Feature selection** — `selected_features.csv`, VIF (~10 с)
4. **GridSearch** — `grid_search_results.csv` (≈ 8 мин)
5. **Walk-forward CV** — `model_summary.csv` (≈ 3 мин на nom + real)
6. **Train final** — `models/*.pkl`
7. **SHAP** — `shap_*.csv`
8. **Forecast** — `forecast_all_scenarios.csv`
9. **Visualize** — `reports/figures/*.png`, `reports/maps/*.html`

Альтернативно — запустить только один шаг:
```powershell
python scripts/run_pipeline.py --step 5
```

### 3. Сгенерировать ноутбук

```powershell
python scripts/make_notebook.py
jupyter notebook notebooks_vkr/01_vkr_main.ipynb
```

### 4. Запустить интерфейс

```powershell
streamlit run app_vkr/streamlit_app.py
```

---

## Методология

### Дефлятор ВРП
Используется **официальный Индекс физического объёма ВРП Росстата (Y477110109)** —
% к предыдущему году в постоянных ценах. Единый базовый год — **2015**.

```
real_factor(i, 2015) = 1.0
real_factor(i, t)    = real_factor(i, t−1) · ИФО(i, t) / 100,  t > 2015
real_factor(i, t)    = real_factor(i, t+1) / (ИФО(i, t+1)/100),  t < 2015

real_GRP(i, t) = nom_GRP(i, 2015) · real_factor(i, t)
```

При отсутствии ИФО для региона используется fallback на CPI (Y477110111).

**Покрытие реального ВРП:** 100% наблюдений (1939 из 1939).

### Контроль утечек
| Где | Что | Реализация |
|------|------|-------------|
| Лаги | данные строго ≤ t−1 | `groupby("object_name").shift(n ≥ 1)` |
| Кластеры | фиксированный mapping | `src_vkr/typology.py`, не из данных |
| Imputer | fit только на train | `SimpleImputer.fit(train_X)` |
| GridSearch | на train ≤ 2019 | TimeSeriesSplit, тест нетронут |
| Walk-forward | train ≤ t−1, test = t | для t ∈ {2020, 2021, 2022, 2023} |
| Аудит | автоматический | `data/processed/leak_audit.csv` |

### Walk-forward CV
```
test_year=2020: train 2001–2019 → test 2020
test_year=2021: train 2001–2020 → test 2021
test_year=2022: train 2001–2021 → test 2022
test_year=2023: train 2001–2022 → test 2023
```

Тестовый период включает **COVID (2020) и санкции (2022)** — самый строгий out-of-sample.

### Модели
| Группа | Модели |
|--------|---------|
| Бенчмарки | Naive, MeanGrowth |
| Эконометрика | LinearRegression, Ridge, Lasso, ElasticNet |
| Деревья | RandomForest |
| Бустинг | GradientBoosting, XGBoost, LightGBM, CatBoost |
| Нейросеть | MLPRegressor (32→16, tanh, early stopping) |

Все ML обучены на `log(target)`, обратное преобразование — `expm1`.

### Сценарии
**Baseline** — продолжение текущих тенденций.
**Optimistic** — рост инвестиций, низкая инфляция, высокая нефть.
**Pessimistic** — стагнация, высокая инфляция, низкая нефть.

| Параметр | Baseline | Optimistic | Pessimistic |
|----------|----------|------------|-------------|
| Нефть Brent, $/барр. | 75 | 95 | 60 |
| Ключ. ставка, % | 12 | 8 | 16 |
| Инфляция, % | 6.5 | 4.5 | 9.5 |
| USD/RUB | 90 | 80 | 100 |
| Рост ВВП РФ, % | +2.5 | +3.5 | +0.8 |
| Номинальный коэффициент с 2025 г. | 0 | +1%/год | −1%/год |
| Реальный коэффициент с 2025 г. | 0 | +1%/год | −1.2%/год |

### Метрики
* **MAE / RMSE** — в рублях/чел.
* **MAPE / sMAPE / WAPE** — в %.
* **R²** — на исходной шкале.
* **RMSLE** — на log-шкале (главная для скошённых распределений).

---

## Структура ноутбука (22 раздела ВКР)

1. Введение — актуальность, цель, задачи, гипотеза.
2. Теоретико-экономическое обоснование (Кобба–Дуглас, AS=AD).
3. Описание данных.
4. Предобработка.
5. **Проверка признаков на наличие утечек данных.**
6. Формирование признаков.
7. Отбор признаков (corr + VIF + ML).
8. Корреляционный анализ.
9. Сравнение классических и современных методов.
10. Разделение данных и скользящее тестирование.
11. Метрики качества.
12. GridSearch / TimeSeriesSplit.
13. Выбор лучшей модели.
14. Feature Importance.
15. SHAP-анализ.
16. Прогноз на 1–5 лет.
17. Сценарный прогноз.
18. Карта РФ с границами регионов.
19. Интерфейс аналитической системы.
20. Практическая ценность.
21. Ограничения исследования.
22. Итоговые выводы.

---

## Воспроизводимость
* `random_state = 42` зафиксирован во всех ML.
* Все пути относительные (`pathlib.Path(__file__).parent.parent`).
* Тестовый период (2020–2023) не используется для отбора признаков, тюнинга.
* Сценарии — фиксированные параметры в `src_vkr/config.py`.
* Версии библиотек закреплены в `requirements.txt`.

---

## Технологии
* **Python ≥ 3.10**, рекомендуется 3.12.
* `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`,
  `statsmodels`, `linearmodels`, `shap`, `matplotlib`, `seaborn`,
  `plotly`, `streamlit`, `geopandas`, `folium`, `joblib`.

---

## Цитирование
Если используете часть кода — приложите ссылку на репозиторий ВКР.
