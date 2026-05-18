"""forecast.py — рекурсивный сценарный прогноз ВРП/душа на 1–5 лет.

Алгоритм:
1. На обученной модели для номинального ВРП прогнозируются годы FORECAST_START..FORECAST_END.
2. На отдельной модели для реального ВРП прогнозируется ВРП на душу населения в ценах BASE_YEAR.
3. На каждой итерации обновляются лаги целевой переменной на основе уже
   спрогнозированных значений предыдущих лет.
4. Лаги экспертных региональных индикаторов (Y477...) заполняются последними
   фактическими значениями t-1 и t-2.
5. Макрофакторы задаются внешними сценариями и используются только как лаги.

Сценарии:
- baseline      — инерционное продолжение текущей траектории;
- optimistic    — более мягкие макроусловия и умеренный рост;
- pessimistic   — жёсткие макроусловия и отрицательная сценарная корректировка.

Доверительные интервалы (90%) считаются для базового номинального сценария
через квантили лог-остатков walk-forward CV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    DATA_PROC, RESULTS,
    TARGET_NOM, GRP_REAL_PC, BASE_YEAR,
    FORECAST_START, HORIZONS,
    MACRO_COLS, MACRO_FORECAST, SCENARIO_LOG_SHOCK,
    SCENARIO_NOMINAL_LOG_SHOCK, ALPHA_CI,
)
from .typology import get_cluster

SCENARIOS = list(SCENARIO_LOG_SHOCK.keys())


def compute_log_residuals(oof_df: pd.DataFrame) -> np.ndarray:
    """Лог-остатки walk-forward CV для расчёта доверительных интервалов."""
    df = oof_df.copy()
    df["log_resid"] = np.log1p(df["y_pred"].clip(lower=1)) - np.log1p(df["y_true"].clip(lower=1))
    return df["log_resid"].dropna().values


def _macro_for_year(year: int, scenario: str) -> dict[str, float]:
    """Сценарные макропараметры для года прогноза.

    Значения фиксированы по сценариям на горизонте 2024–2028: это не прогноз
    макроэкономики, а управляемые предположения для сценарного анализа.
    """
    out = {}
    for k, (b, o, p) in MACRO_FORECAST.items():
        if scenario == "optimistic":
            tgt = o
        elif scenario == "pessimistic":
            tgt = p
        else:
            tgt = b
        out[k] = float(tgt)
    return out


def _last_known_indicator(df: pd.DataFrame, region: str, code: str, year_t: int, lag: int) -> float:
    yr = year_t - lag
    rec = df[(df["object_name"] == region) & (df["year"] == yr)]
    if rec.empty or code not in rec.columns:
        return np.nan
    value = rec[code].iloc[0]
    return float(value) if pd.notna(value) else np.nan


def _actual_region_value(panel: pd.DataFrame, region: str, year: int, column: str) -> float:
    rec = panel[(panel["object_name"] == region) & (panel["year"] == year)]
    if rec.empty or column not in rec.columns:
        return np.nan
    value = rec[column].iloc[0]
    return float(value) if pd.notna(value) else np.nan


def _value_log(panel: pd.DataFrame, region: str, year: int, log_col: str, cache: dict[tuple[str, int], float]) -> float:
    if (region, year) in cache:
        return cache[(region, year)]
    rec = panel[(panel["object_name"] == region) & (panel["year"] == year)]
    if rec.empty or log_col not in rec.columns:
        return np.nan
    value = rec[log_col].iloc[0]
    return float(value) if pd.notna(value) else np.nan


def _build_row(panel: pd.DataFrame,
               region: str,
               year_t: int,
               features: list[str],
               pred_log_cache: dict[tuple[str, int], float],
               real_log_cache: dict[tuple[str, int], float],
               scenario: str) -> dict:
    """Собирает признаковый вектор для региона и прогнозного года."""
    row: dict[str, float] = {}
    row["year_norm"] = (year_t - 2001) / 22.0
    row["cluster_id"] = get_cluster(region)

    fo_cols = [c for c in features if c.startswith("fo_")]
    last_known = panel[panel["object_name"] == region].sort_values("year").tail(1)
    for c in fo_cols:
        row[c] = float(last_known[c].iloc[0]) if c in last_known.columns else 0.0

    # Номинальные лаги
    nl1 = _value_log(panel, region, year_t - 1, "nom_log", pred_log_cache)
    nl2 = _value_log(panel, region, year_t - 2, "nom_log", pred_log_cache)
    nl3 = _value_log(panel, region, year_t - 3, "nom_log", pred_log_cache)
    row["nom_log_lag1"] = nl1
    row["nom_log_lag2"] = nl2
    row["nom_log_lag3"] = nl3
    row["nom_log_roll3_mean"] = float(np.nanmean([nl1, nl2, nl3]))
    row["nom_log_roll3_std"] = float(np.nanstd([nl1, nl2, nl3]))
    row["nom_growth1"] = (nl1 - nl2) if pd.notna(nl1) and pd.notna(nl2) else np.nan
    nl4 = _value_log(panel, region, year_t - 4, "nom_log", pred_log_cache)
    row["nom_growth3"] = (nl1 - nl4) if pd.notna(nl1) and pd.notna(nl4) else np.nan

    # Реальные лаги
    rl1 = _value_log(panel, region, year_t - 1, "real_log", real_log_cache)
    rl2 = _value_log(panel, region, year_t - 2, "real_log", real_log_cache)
    rl3 = _value_log(panel, region, year_t - 3, "real_log", real_log_cache)
    row["real_log_lag1"] = rl1
    row["real_log_lag2"] = rl2
    row["real_log_lag3"] = rl3
    row["real_log_roll3_mean"] = float(np.nanmean([rl1, rl2, rl3]))
    row["real_log_roll3_std"] = float(np.nanstd([rl1, rl2, rl3]))
    row["real_growth1"] = (rl1 - rl2) if pd.notna(rl1) and pd.notna(rl2) else np.nan

    if "nom_log_vs_median_lag1" in features:
        med = panel[panel["year"] == year_t - 1]["nom_log"].median() if "nom_log" in panel.columns else 0
        row["nom_log_vs_median_lag1"] = nl1 - med if pd.notna(nl1) else np.nan

    # Лаги экспертных индикаторов
    for f in features:
        if not f.startswith("Y477"):
            continue
        code = f.split("_")[0]
        if "yoy_lag1" in f:
            v1 = _last_known_indicator(panel, region, code, year_t, lag=1)
            v2 = _last_known_indicator(panel, region, code, year_t, lag=2)
            row[f] = float(np.log1p(max(v1, 0)) - np.log1p(max(v2, 0))) if pd.notna(v1) and pd.notna(v2) and v2 > 0 else np.nan
        elif "lag1" in f:
            row[f] = _last_known_indicator(panel, region, code, year_t, lag=1)
        elif "lag2" in f:
            row[f] = _last_known_indicator(panel, region, code, year_t, lag=2)

    # Макрофакторы: для первого прогнозного года lag1 = факт прошлого года,
    # далее lag1 = сценарное значение предыдущего прогнозного года.
    sc_macro = _macro_for_year(year_t - 1, scenario)
    for c in MACRO_COLS:
        key = f"{c}_lag1"
        if key not in features:
            continue
        if year_t - 1 < FORECAST_START:
            rec = panel[panel["year"] == year_t - 1]
            row[key] = float(rec[c].mean()) if len(rec) and c in rec.columns else sc_macro.get(c, np.nan)
        else:
            row[key] = sc_macro.get(c, np.nan)

    for f in features:
        row.setdefault(f, np.nan)
    return row


def _predict_log(bundle: dict, row: dict[str, float]) -> float:
    features = bundle["features"]
    X = np.array([[row.get(f, np.nan) for f in features]], dtype=float)
    X_i = bundle["imputer"].transform(X)
    return float(bundle["model"].predict(X_i)[0])


def forecast_one_scenario(panel: pd.DataFrame,
                          nominal_bundle: dict,
                          real_bundle: dict | None,
                          scenario: str,
                          horizons: tuple[int, ...] = HORIZONS,
                          log_resid: np.ndarray | None = None) -> pd.DataFrame:
    """Рекурсивный прогноз одного сценария для всех регионов."""
    features = nominal_bundle["features"]
    last_year = int(panel["year"].max())

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

    for region in sorted(panel["object_name"].unique()):
        for h in horizons:
            year_t = last_year + h
            row = _build_row(panel, region, year_t, features, pred_cache, real_cache, scenario)

            nom_log = _predict_log(nominal_bundle, row)
            if h > 1:
                nom_log += (h - 1) * nominal_level_shock
            pred_cache[(region, year_t)] = nom_log
            y_nom = float(np.expm1(nom_log))

            if scenario == "baseline":
                lo = float(np.expm1(nom_log + q_lo * np.sqrt(h)))
                hi = float(np.expm1(nom_log + q_hi * np.sqrt(h)))
            else:
                lo = hi = np.nan

            if real_bundle is not None:
                real_features = real_bundle["features"]
                real_row = _build_row(panel, region, year_t, real_features, pred_cache, real_cache, scenario)
                real_log = _predict_log(real_bundle, real_row)
                if h > 1:
                    real_log += (h - 1) * real_growth_shock
                y_real = float(np.expm1(real_log))
            else:
                # Fallback: индексная корректировка номинального прогноза.
                prev_nom = _actual_region_value(panel, region, year_t - 1, TARGET_NOM)
                prev_real = _actual_region_value(panel, region, year_t - 1, GRP_REAL_PC)
                if (region, year_t - 1) in pred_cache:
                    prev_nom = float(np.expm1(pred_cache[(region, year_t - 1)]))
                if (region, year_t - 1) in real_cache:
                    prev_real = float(np.expm1(real_cache[(region, year_t - 1)]))
                inflation = _macro_for_year(year_t, scenario)["inflation_rf_wb"]
                nominal_growth_log = np.log1p(max(y_nom, 1.0)) - np.log1p(max(prev_nom, 1.0))
                real_growth_log = nominal_growth_log - np.log1p(inflation / 100)
                if h > 1:
                    real_growth_log += real_growth_shock
                y_real = float(max(prev_real * np.exp(real_growth_log), 1.0))
                real_log = float(np.log1p(max(y_real, 1)))

            real_cache[(region, year_t)] = real_log

            rows.append({
                "object_name": region,
                "year": year_t,
                "scenario": scenario,
                "horizon": h,
                "y_pred_nominal": y_nom,
                "y_pred_real_2015": y_real,
                "y_lo_nominal": lo,
                "y_hi_nominal": hi,
                "model": nominal_bundle.get("name", "model"),
                "real_model": real_bundle.get("name", "fallback_index") if real_bundle else "fallback_index",
            })
    return pd.DataFrame(rows)


def all_scenarios_forecast(panel: pd.DataFrame,
                            nominal_bundle: dict,
                            real_bundle: dict | None = None,
                            horizons: tuple[int, ...] = HORIZONS,
                            log_resid: np.ndarray | None = None) -> pd.DataFrame:
    print("=" * 70)
    print(" СЦЕНАРНЫЙ ПРОГНОЗ 2024–2028")
    print("=" * 70)
    parts = []
    for sc in SCENARIOS:
        print(f"\n  → {sc}...")
        df_sc = forecast_one_scenario(panel, nominal_bundle, real_bundle, sc, horizons, log_resid)
        parts.append(df_sc)
    out = pd.concat(parts, ignore_index=True)
    out["date_computed"] = pd.Timestamp.now().strftime("%Y-%m-%d")
    out.to_csv(RESULTS / "forecast_all_scenarios.csv", index=False, encoding="utf-8-sig")
    print(f"\n  [OK] {RESULTS / 'forecast_all_scenarios.csv'}  shape={out.shape}")
    return out


if __name__ == "__main__":
    import pickle
    panel = pd.read_parquet(DATA_PROC / "panel_features.parquet")
    with open("models/gradientboosting_nominal.pkl", "rb") as f:
        nominal = pickle.load(f)
    with open("models/gradientboosting_real.pkl", "rb") as f:
        real = pickle.load(f)
    all_scenarios_forecast(panel, nominal, real_bundle=real)
