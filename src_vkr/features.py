"""features.py — построение признаков и контроль утечек.

Ключевые принципы:

1. Все лаговые признаки строятся СТРОГО внутри региона
   (`groupby("object_name").shift(k)` с k ≥ 1).

2. Скользящие признаки сначала лагируются, затем агрегируются
   (rolling считается по уже сдвинутым значениям).

3. Темпы роста — это лаг(t-1) / лаг(t-2) - 1 (без значения текущего года).

4. Никаких "будущих" макрофакторов: только лаги.

5. Контроль утечек:
   - см. FORBIDDEN_FEATURES в config: эти признаки в году t недоступны;
   - целевая переменная и её прямые производные исключены из X;
   - кластер региона — экспертный (фиксированный mapping), не подсматривается.

В результате формируется набор признаков, который содержит:
  - target_lag1, target_lag2, target_lag3  — лаги целевой переменной (log);
  - target_roll3_mean, target_roll3_std    — 3-летние агрегаты;
  - target_growth1, target_growth3         — темпы прироста;
  - для ~30 ключевых индикаторов из EXPERT_INDICATORS: lag1, lag2, yoy;
  - макрофакторы (key_rate, oil, инфляция и т.д.) — lag1;
  - временной тренд (year_norm) и кластер региона;
  - федеральный округ как one-hot.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DATA_RAW, DATA_PROC, TARGET_NOM, GRP_REAL_PC, GRP_TOTAL,
    EXPERT_INDICATORS, MACRO_COLS, FORBIDDEN_FEATURES, YEARS_RANGE,
    RANDOM_STATE,
)
from .typology import get_cluster

warnings.filterwarnings("ignore")


def _lag(group: pd.DataFrame, col: str, k: int) -> pd.Series:
    return group[col].shift(k)


def add_target_lags(df: pd.DataFrame, target_col: str, prefix: str) -> pd.DataFrame:
    """Добавляет лаги, rolling и темпы прироста по target.

    target_col   — имя колонки (TARGET_NOM или GRP_REAL_PC).
    prefix       — префикс для имён ('nom' или 'real').

    Колонки:
      - {prefix}_log               — log1p(target) текущего года (используется как y);
      - {prefix}_log_lag1/2/3      — лаги log целевой переменной;
      - {prefix}_log_roll3_mean    — скользящее среднее (3 года, по лагам);
      - {prefix}_log_roll3_std     — скользящее std (3 года, по лагам);
      - {prefix}_growth1           — (lag1 - lag2) / lag2;
      - {prefix}_growth3           — (lag1 - lag4) / lag4 (3-летний прирост);
    """
    df = df.sort_values(["object_name", "year"]).copy()
    log_col = f"{prefix}_log"
    df[log_col] = np.log1p(df[target_col].clip(lower=0))

    gb = df.groupby("object_name", sort=False)
    for k in (1, 2, 3):
        df[f"{prefix}_log_lag{k}"] = gb[log_col].shift(k)

    # rolling по лагам (т.е. строго по прошлому)
    lag1 = df.groupby("object_name", sort=False)[log_col].shift(1)
    df[f"{prefix}_log_roll3_mean"] = (
        lag1.groupby(df["object_name"]).rolling(3, min_periods=2).mean()
        .reset_index(level=0, drop=True)
    )
    df[f"{prefix}_log_roll3_std"] = (
        lag1.groupby(df["object_name"]).rolling(3, min_periods=2).std()
        .reset_index(level=0, drop=True)
    )

    # Темпы прироста по лагам
    df[f"{prefix}_growth1"] = (
        df[f"{prefix}_log_lag1"] - df[f"{prefix}_log_lag2"]
    )
    df[f"{prefix}_growth3"] = (
        df[f"{prefix}_log_lag1"] - df.groupby("object_name", sort=False)[log_col].shift(4)
    )
    return df


def add_feature_lags(df: pd.DataFrame, codes: list[str],
                     lags: tuple[int, ...] = (1, 2)) -> pd.DataFrame:
    """Лаги ключевых индикаторов (без значения t)."""
    df = df.sort_values(["object_name", "year"]).copy()
    gb = df.groupby("object_name", sort=False)
    for code in codes:
        if code not in df.columns:
            continue
        for k in lags:
            df[f"{code}_lag{k}"] = gb[code].shift(k)
        # YoY (по логам, для устойчивости к выбросам)
        log_col = np.log1p(df[code].clip(lower=0))
        df[f"{code}_yoy"] = log_col - log_col.groupby(df["object_name"]).shift(1)
        # И лаг YoY (чтобы не подглядывать в текущий год)
        df[f"{code}_yoy_lag1"] = df.groupby("object_name", sort=False)[
            f"{code}_yoy"
        ].shift(1)
        df.drop(columns=[f"{code}_yoy"], inplace=True)
    return df


def add_macro_lags(df: pd.DataFrame, lags: tuple[int, ...] = (1,)) -> pd.DataFrame:
    """Лаги общероссийских макрофакторов."""
    df = df.copy()
    for c in MACRO_COLS:
        if c not in df.columns:
            continue
        # Лаги общие (по году) — но всё равно через groupby для устойчивости
        for k in lags:
            df[f"{c}_lag{k}"] = df.groupby("object_name", sort=False)[c].shift(k)
    return df


def add_temporal_and_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет временной тренд, кластер региона, федеральный округ."""
    df = df.copy()
    df["year_norm"] = (df["year"] - YEARS_RANGE[0]) / (
        YEARS_RANGE[1] - YEARS_RANGE[0]
    )
    df["cluster_id"] = df["object_name"].apply(get_cluster).astype(int)

    # ФО (one-hot)
    region_to_fo_path = DATA_RAW / "region_to_fo.json"
    if region_to_fo_path.exists():
        with open(region_to_fo_path, encoding="utf-8") as f:
            r2fo = json.load(f)
        df["federal_district"] = df["object_name"].map(r2fo).fillna("Прочее")
    else:
        df["federal_district"] = "Прочее"
    fo_dummies = pd.get_dummies(df["federal_district"], prefix="fo", dtype=int)
    df = pd.concat([df, fo_dummies], axis=1)
    return df


def add_relative_position(df: pd.DataFrame) -> pd.DataFrame:
    """Относительная позиция региона: лог-разница с медианой РФ (по предыдущему году)."""
    df = df.sort_values(["object_name", "year"]).copy()
    log_grp = np.log1p(df[TARGET_NOM].clip(lower=0))
    log_grp_lag1 = log_grp.groupby(df["object_name"]).shift(1)
    df["nom_log_lag1_anchor"] = log_grp_lag1
    median_log_by_year = log_grp_lag1.groupby(df["year"]).transform("median")
    df["nom_log_vs_median_lag1"] = log_grp_lag1 - median_log_by_year
    df = df.drop(columns=["nom_log_lag1_anchor"], errors="ignore")
    return df


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Полный feature engineering на основе preprocessed wide-panel."""
    print("=" * 70)
    print(" FEATURE ENGINEERING")
    print("=" * 70)
    df = panel.copy()

    print("\n  [1/5] Лаги, rolling, темпы прироста для целевой переменной (nom + real)")
    df = add_target_lags(df, target_col=TARGET_NOM, prefix="nom")
    df = add_target_lags(df, target_col=GRP_REAL_PC, prefix="real")

    expert_codes = [c for c in EXPERT_INDICATORS.keys() if c in df.columns]
    print(f"\n  [2/5] Лаги {len(expert_codes)} экспертных индикаторов "
          f"(из {len(EXPERT_INDICATORS)} в каталоге)")
    df = add_feature_lags(df, expert_codes, lags=(1, 2))

    print(f"\n  [3/5] Лаги {len(MACRO_COLS)} макрофакторов")
    df = add_macro_lags(df, lags=(1,))

    print("\n  [4/5] Временной тренд, кластер, федеральный округ")
    df = add_temporal_and_cluster(df)

    print("\n  [5/5] Относительная позиция (лог-разница с медианой РФ)")
    df = add_relative_position(df)

    out = DATA_PROC / "panel_features.parquet"
    df.to_parquet(out, index=False)
    print(f"\n  [OK] {out}  shape={df.shape}")
    return df


def get_candidate_features(df: pd.DataFrame) -> list[str]:
    """Возвращает список кандидатных признаков для отбора.

    Отбрасывает явные утечки и нерелевантные служебные колонки.
    """
    drop = {
        "object_name", "year", "federal_district", "grp_deflator_src",
    } | set(FORBIDDEN_FEATURES) | set(MACRO_COLS)
    # MACRO_COLS в их чистом виде (без лагов) тоже исключаем — заменены на _lag1
    drop |= {TARGET_NOM, GRP_TOTAL, GRP_REAL_PC, "real_factor",
             "grp_nom_growth", "grp_real_growth"}
    # nominal/real log без лага — это таргеты
    drop |= {"nom_log", "real_log"}
    # Сырые индикаторы (Y477...) — только через лаги; чистые коды исключаем
    cands = [
        c for c in df.columns
        if c not in drop
        and not (c.startswith("Y477") and "_lag" not in c
                 and "_yoy" not in c and "_roll" not in c)
    ]
    return cands


def write_leak_audit(df: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    """Аудит признаков на утечки: корреляция с target в один и тот же год.

    Высокая корреляция (>0.99) при отсутствии лага в имени — подозрение.
    """
    audit_rows = []
    y = df["nom_log"]
    for c in candidates:
        if df[c].dtype == object:
            continue
        x = df[c]
        mask = x.notna() & y.notna()
        if mask.sum() < 50:
            corr = np.nan
        else:
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
        suspicious = (
            abs(corr) > 0.97 and
            not any(tag in c for tag in ["lag", "roll", "growth", "yoy"]) and
            c not in {"year_norm", "cluster_id", "nom_log_vs_median_lag1"}
        )
        audit_rows.append({
            "feature": c,
            "corr_with_target": round(corr, 4) if pd.notna(corr) else None,
            "has_lag_in_name": any(tag in c for tag in ["lag", "roll", "growth", "yoy"]),
            "suspect_leak": suspicious,
        })
    audit = pd.DataFrame(audit_rows).sort_values(
        "corr_with_target", key=lambda s: s.abs(), ascending=False
    )
    out = DATA_PROC / "leak_audit.csv"
    audit.to_csv(out, index=False, encoding="utf-8-sig")
    return audit


if __name__ == "__main__":
    panel = pd.read_parquet(DATA_PROC / "panel_preprocessed.parquet")
    df = build_features(panel)
    cands = get_candidate_features(df)
    print(f"\n  Кандидатных признаков: {len(cands)}")
    audit = write_leak_audit(df, cands)
    print(f"  Подозрения на утечки: {audit['suspect_leak'].sum()}")
