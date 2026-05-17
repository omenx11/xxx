"""forecast.py — рекурсивный сценарный прогноз ВРП/душа на 1–5 лет.

Алгоритм:
1. На обученной модели (бандл = {model, imputer, features, target_raw}),
   итеративно прогнозируем годы FORECAST_START..FORECAST_END.
2. На каждой итерации обновляем лаги целевой переменной (target_log_lag1/2/3,
   rolling, growth) на основе уже спрогнозированных значений предыдущих лет.
3. Лаги "экспертных" признаков (Y477...) заполняем последним фактическим
   значением (assumption: для прогнозируемых лет известны лагированные
   значения за последний фактический год t-1, t-2).
4. Макрофакторы (key_rate, oil, инфляция и т.д.) задаются ВНЕШНЕ через сценарий.

Сценарии:
- baseline      — умеренные параметры, продолжение текущих тенденций;
- optimistic    — рост инвестиций, снижение безработицы, низкая инфляция,
                  стабильный рубль, высокие цены на нефть;
- pessimistic   — снижение инвестиций, рост безработицы, высокая инфляция,
                  слабый рубль, низкие цены на нефть.

Сценарии действуют ДВУМЯ каналами:
(а) подмена макрофакторов в признаковом векторе;
(б) ежегодный log-shock на реальный темп роста после первого прогнозного года.

Доверительные интервалы (90%) считаются ТОЛЬКО для базового сценария,
используя квантили лог-остатков walk-forward CV.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DATA_PROC, RESULTS, MODELS, RANDOM_STATE,
    TARGET_NOM, GRP_REAL_PC, GRP_IFO, GRP_TOTAL, CPI_REG, BASE_YEAR,
    TEST_END, FORECAST_START, FORECAST_END, HORIZONS,
    MACRO_COLS, MACRO_FORECAST, SCENARIO_LOG_SHOCK,
    SCENARIO_NOMINAL_LOG_SHOCK,
    EXPERT_INDICATORS, ALPHA_CI,
)
from .typology import get_cluster

SCENARIOS = list(SCENARIO_LOG_SHOCK.keys())  # ['baseline','optimistic','pessimistic']


def compute_log_residuals(oof_df: pd.DataFrame) -> np.ndarray:
    """Лог-остатки walk-forward CV (для CI прогноза)."""
    df = oof_df.copy()
    df["log_resid"] = (
        np.log1p(df["y_pred"].clip(lower=1)) - np.log1p(df["y_true"].clip(lower=1))
    )
    return df["log_resid"].dropna().values


def _macro_for_year(year: int, scenario: str) -> dict[str, float]:
    """Линейная интерполяция макропараметров для года прогноза.

    На горизонте 1–5 лет (2024–2028) применяем плавный линейный переход
    от значения текущего года к таргету через 5 лет.
    """
    out = {}
    h = year - (FORECAST_START - 1)  # 1..5
    h = max(1, min(5, h))
    for k, (b, o, p) in MACRO_FORECAST.items():
        if scenario == "baseline":
            tgt = b
        elif scenario == "optimistic":
            tgt = o
        elif scenario == "pessimistic":
            tgt = p
        else:
            tgt = b
        out[k] = float(tgt)
    return out


def _last_known_macro(macro_panel: pd.DataFrame, year_target: int) -> dict[str, float]:
    """Лаг(t-1) макропоказателей для года t."""
    sub = macro_panel[macro_panel["year"] == year_target - 1]
    if len(sub) == 0:
        sub = macro_panel.tail(1)
    return sub.iloc[0][MACRO_COLS].to_dict()


def _last_known_indicator(df: pd.DataFrame, region: str,
                           code: str, year_t: int, lag: int) -> float:
    """Значение индикатора code в году (year_t - lag) для региона."""
    yr = year_t - lag
    rec = df[(df["object_name"] == region) & (df["year"] == yr)]
    if rec.empty or code not in rec.columns:
        return np.nan
    return float(rec[code].iloc[0])


def _actual_region_value(panel: pd.DataFrame, region: str, year: int,
                         column: str) -> float:
    rec = panel[(panel["object_name"] == region) & (panel["year"] == year)]
    if rec.empty or column not in rec.columns:
        return np.nan
    value = rec[column].iloc[0]
    return float(value) if pd.notna(value) else np.nan


def _build_row(panel: pd.DataFrame, region: str, year_t: int,
                features: list[str], pred_log_cache: dict,
                real_log_cache: dict,
                scenario: str) -> dict:
    """Собирает признаковый вектор для региона region и года year_t.

    pred_log_cache : {(region, year): log(nominal_pred)} — кеш ном. прогнозов.
    real_log_cache : {(region, year): log(real_pred_2015)} — кеш реал. прогнозов.
    """
    row = {}
    # 1) Базовые статические признаки
    row["year_norm"] = (year_t - 2001) / 22.0
    row["cluster_id"] = get_cluster(region)

    # one-hot ФО (берём из panel — фиксированный для региона)
    fo_cols = [c for c in features if c.startswith("fo_")]
    last_known = panel[panel["object_name"] == region].sort_values("year").tail(1)
    for c in fo_cols:
        row[c] = float(last_known[c].iloc[0]) if c in last_known.columns else 0.0

    # 2) Лаги target_log: 1, 2, 3 (NOMINAL)
    def get_nom_log(yr):
        if (region, yr) in pred_log_cache:
            return pred_log_cache[(region, yr)]
        rec = panel[(panel["object_name"] == region) & (panel["year"] == yr)]
        if rec.empty:
            return np.nan
        if "nom_log" in rec.columns and pd.notna(rec["nom_log"].iloc[0]):
            return float(rec["nom_log"].iloc[0])
        return np.nan

    nl1 = get_nom_log(year_t - 1)
    nl2 = get_nom_log(year_t - 2)
    nl3 = get_nom_log(year_t - 3)
    row["nom_log_lag1"] = nl1
    row["nom_log_lag2"] = nl2
    row["nom_log_lag3"] = nl3
    row["nom_log_roll3_mean"] = float(np.nanmean([nl1, nl2, nl3]))
    row["nom_log_roll3_std"] = float(np.nanstd([nl1, nl2, nl3]))
    row["nom_growth1"] = (nl1 - nl2) if pd.notna(nl1) and pd.notna(nl2) else np.nan
    nl4 = get_nom_log(year_t - 4)
    row["nom_growth3"] = (nl1 - nl4) if pd.notna(nl1) and pd.notna(nl4) else np.nan

    # 3) Лаги real_log (REAL, в ценах 2015) — также через кеш для будущих лет
    def get_real_log(yr):
        if (region, yr) in real_log_cache:
            return real_log_cache[(region, yr)]
        rec = panel[(panel["object_name"] == region) & (panel["year"] == yr)]
        if rec.empty or "real_log" not in rec.columns:
            return np.nan
        v = rec["real_log"].iloc[0]
        return float(v) if pd.notna(v) else np.nan

    rl1 = get_real_log(year_t - 1)
    rl2 = get_real_log(year_t - 2)
    rl3 = get_real_log(year_t - 3)
    row["real_log_lag1"] = rl1
    row["real_log_lag2"] = rl2
    row["real_log_lag3"] = rl3
    row["real_log_roll3_mean"] = float(np.nanmean([rl1, rl2, rl3]))
    row["real_log_roll3_std"] = float(np.nanstd([rl1, rl2, rl3]))
    row["real_growth1"] = (rl1 - rl2) if pd.notna(rl1) and pd.notna(rl2) else np.nan

    # 4) Относительная позиция: log(grp_lag1) - медиана по году
    if "nom_log_vs_median_lag1" in features:
        # пересчитываем медиану lag1 на основе panel и кеша
        # упрощённо: возьмём медиану по lag1 из panel за year_t-1
        med = panel[panel["year"] == year_t - 1]["nom_log"].median() if "nom_log" in panel.columns else 0
        row["nom_log_vs_median_lag1"] = nl1 - med if pd.notna(nl1) else np.nan

    # 5) Лаги экспертных индикаторов (Y477...) — берём фактические значения t-1, t-2
    for f in features:
        if f.startswith("Y477"):
            # формат вида CODE_lag1, CODE_lag2, CODE_yoy_lag1
            parts = f.split("_")
            code = parts[0]
            if "lag1" in f:
                row[f] = _last_known_indicator(panel, region, code, year_t, lag=1)
            elif "lag2" in f:
                row[f] = _last_known_indicator(panel, region, code, year_t, lag=2)
            elif "yoy_lag1" in f:
                v1 = _last_known_indicator(panel, region, code, year_t, lag=1)
                v2 = _last_known_indicator(panel, region, code, year_t, lag=2)
                if pd.notna(v1) and pd.notna(v2) and v2 > 0:
                    row[f] = float(np.log1p(max(v1, 0)) - np.log1p(max(v2, 0)))
                else:
                    row[f] = np.nan

    # 6) Макрофакторы (lag1) — на основе сценария
    sc_macro = _macro_for_year(year_t, scenario)
    for c in MACRO_COLS:
        key = f"{c}_lag1"
        if key in features:
            # на t-1 берём сценарное значение (применённое в год t-1)
            if year_t - 1 < FORECAST_START:
                # lag1 для года прогноза = фактическое значение за прошлый год
                rec = panel[panel["year"] == year_t - 1]
                if len(rec) and c in rec.columns:
                    row[key] = float(rec[c].mean())  # макро общий по году
                else:
                    row[key] = sc_macro.get(c, np.nan)
            else:
                # уже в горизонте прогноза — сценарное
                row[key] = sc_macro.get(c, np.nan)

    # 7) Остальные признаки, не покрытые выше — заполним NaN (imputer обработает)
    for f in features:
        if f not in row:
            row[f] = np.nan
    return row


def forecast_one_scenario(panel: pd.DataFrame, bundle: dict,
                           scenario: str,
                           horizons: tuple[int, ...] = HORIZONS,
                           log_resid: np.ndarray = None) -> pd.DataFrame:
    """Рекурсивный прогноз одного сценария для всех регионов.

    Возвращает long DF: [object_name, year, scenario, horizon,
                          y_pred_nom, y_pred_real, y_lo_nom, y_hi_nom].
    """
    model = bundle["model"]
    imputer = bundle["imputer"]
    features = bundle["features"]
    last_year = int(panel["year"].max())

    # CI: используем квантили лог-остатков (только для baseline)
    if log_resid is not None and len(log_resid) >= 20:
        q_lo = float(np.quantile(log_resid, ALPHA_CI / 2))
        q_hi = float(np.quantile(log_resid, 1 - ALPHA_CI / 2))
    else:
        q_lo, q_hi = -0.15, 0.15

    pred_cache: dict[tuple[str, int], float] = {}
    real_cache: dict[tuple[str, int], float] = {}
    rows = []
    real_growth_shock = SCENARIO_LOG_SHOCK.get(scenario, 0.0)
    nominal_level_shock = SCENARIO_NOMINAL_LOG_SHOCK.get(scenario, 0.0)

    regions = sorted(panel["object_name"].unique())
    for region in regions:
        prev_nominal = _actual_region_value(panel, region, last_year, TARGET_NOM)
        prev_real = _actual_region_value(panel, region, last_year, GRP_REAL_PC)

        for h in horizons:
            year_t = last_year + h
            row = _build_row(panel, region, year_t, features,
                             pred_cache, real_cache, scenario)
            X = np.array([[row.get(f, np.nan) for f in features]], dtype=float)
            X_i = imputer.transform(X)
            log_pred_raw = float(model.predict(X_i)[0])
            log_pred = log_pred_raw
            if h > 1:
                log_pred += (h - 1) * nominal_level_shock

            # Кеш ном. прогноза (для лагов следующих лет)
            pred_cache[(region, year_t)] = log_pred

            y_pred = float(np.expm1(log_pred))

            # CI только для baseline
            if scenario == "baseline":
                lo = float(np.expm1(log_pred + q_lo * np.sqrt(h)))
                hi = float(np.expm1(log_pred + q_hi * np.sqrt(h)))
            else:
                lo = hi = np.nan

            # Реальный прогноз строится индексно от фактического уровня 2023.
            # Это сохраняет преемственность: 2024 не перескакивает на уровень
            # номинального прогноза, а меняется на номинальный темп минус
            # сценарная инфляция. Структурный shock включается с h=2.
            if pd.isna(prev_nominal) or prev_nominal <= 0:
                prev_nominal = max(y_pred, 1.0)
            if pd.isna(prev_real) or prev_real <= 0:
                prev_real = max(y_pred, 1.0)

            inflation = _macro_for_year(year_t, scenario)["inflation_rf_wb"]
            nominal_growth_log = np.log1p(max(y_pred, 1.0)) - np.log1p(max(prev_nominal, 1.0))
            real_growth_log = nominal_growth_log - np.log1p(inflation / 100)
            if h > 1:
                real_growth_log += real_growth_shock
            y_real = float(max(prev_real * np.exp(real_growth_log), 1.0))

            # Кешируем real_log для лагов следующих лет
            real_cache[(region, year_t)] = float(np.log1p(max(y_real, 1)))
            prev_nominal = y_pred
            prev_real = y_real

            rows.append({
                "object_name": region,
                "year": year_t,
                "scenario": scenario,
                "horizon": h,
                "y_pred_nominal": y_pred,
                "y_pred_real_2015": y_real,
                "y_lo_nominal": lo,
                "y_hi_nominal": hi,
                "model": bundle.get("name", "model"),
            })
    return pd.DataFrame(rows)


def all_scenarios_forecast(panel: pd.DataFrame, bundle: dict,
                            horizons: tuple[int, ...] = HORIZONS,
                            log_resid: np.ndarray = None) -> pd.DataFrame:
    print("=" * 70)
    print(" СЦЕНАРНЫЙ ПРОГНОЗ 2024–2028")
    print("=" * 70)
    parts = []
    for sc in SCENARIOS:
        print(f"\n  → {sc}...")
        df_sc = forecast_one_scenario(panel, bundle, sc, horizons, log_resid)
        parts.append(df_sc)
    out = pd.concat(parts, ignore_index=True)
    out["date_computed"] = pd.Timestamp.now().strftime("%Y-%m-%d")
    out.to_csv(RESULTS / "forecast_all_scenarios.csv", index=False,
                encoding="utf-8-sig")
    print(f"\n  [OK] {RESULTS / 'forecast_all_scenarios.csv'}  shape={out.shape}")
    return out
