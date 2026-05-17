"""preprocess.py — препроцессинг сырых данных Росстата → чистая wide-панель.

Этапы:
1. Загрузка long-формата (parquet), фильтрация регионов и периода 2001–2023.
2. Замена спецзначений (-99999999) на NaN.
3. Дедупликация по свежести источника.
4. Нормализация единиц измерения (млн руб., тыс чел., руб.).
5. Pivot из long в wide (object_name × year × indicator_code).
6. Подключение макрофакторов (общих по году).
7. Расчёт реального ВРП на душу населения в ценах 2015 года через ИФО ВРП.
8. Дополнительные базовые признаки: реальный ВРП всего региона, темпы роста.
9. Дроп очень разреженных колонок (>60% пропусков).
10. Сохранение panel_preprocessed.parquet + feature_catalog.csv.

Дефлятор: используется официальный Индекс физического объёма ВРП Росстата
(Y477110109, % к предыдущему году) с единой базой 2015 года для всех регионов.

Метод (рекурсивная процедура):
  real_factor(2015) = 1.0
  real_factor(t) = real_factor(t-1) * (ИФО(t) / 100)      при t > 2015
  real_factor(t) = real_factor(t+1) / (ИФО(t+1) / 100)    при t < 2015

  GRP_real(i, t) = GRP_nom(i, 2015) * real_factor(i, t)
  GRP_real/pc(i, t) = GRP_real(i, t) / Population(i, t)

Если данных по ИФО нет — fallback на накопленный CPI (Y477110111).
"""
from __future__ import annotations
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DATA_RAW, DATA_PROC, YEARS_RANGE, TARGET_NOM, GRP_TOTAL, GRP_IFO, CPI_REG,
    GRP_REAL_PC, BASE_YEAR, MACRO_COLS,
)

warnings.filterwarnings("ignore")

# Спецзначения, обозначающие пропуски в данных Росстата
SPECIAL_VAL = [-99_999_999, -77_777_777, -88_888_888]

# Агрегаты, которые нужно убрать (есть как родительские регионы + детальные)
AGG_REMOVE = [
    "Архангельская область (с автономным округом)",
    "Тюменская область (с автономными округами)",
]

# Канонизация единиц: ключ — текст единицы в данных,
# значение — (множитель, краткое название канонической единицы)
UNIT_NORM = {
    "Миллиардов рублей":                                  (1000.0, "Млн руб"),
    "В фактически действовавших ценах, миллиардов рублей":(1000.0, "Млн руб"),
    "Миллионов рублей":                                   (1.0,    "Млн руб"),
    "В фактически действовавших ценах, миллионов рублей": (1.0,    "Млн руб"),
    "В текущих рыночных ценах, миллионов рублей":         (1.0,    "Млн руб"),
    "Тысяч рублей":                                       (0.001,  "Млн руб"),
    "Рублей":                                             (1.0,    "Руб"),
    "В фактически действовавших ценах, рублей":           (1.0,    "Руб"),
    "Рублей в месяц":                                     (1.0,    "Руб/мес"),
    "В месяц, рублей":                                    (1.0,    "Руб/мес"),
    "Тысяч человек":                                      (1.0,    "Тыс чел"),
    "На конец года, тысяч человек":                       (1.0,    "Тыс чел"),
    "Оценка на конец года, тысяч человек":                (1.0,    "Тыс чел"),
    "Оценка, тысяч человек":                              (1.0,    "Тыс чел"),
    "Человек":                                            (0.001,  "Тыс чел"),
    "На конец года, человек":                             (0.001,  "Тыс чел"),
    "На 1 января, человек":                               (0.001,  "Тыс чел"),
    "В среднем за год, человек":                          (0.001,  "Тыс чел"),
    "На конец года,тысяч человек":                        (1.0,    "Тыс чел"),
}


def _extract_source_year(s: str) -> int:
    """Извлекает год из строки источника (для выбора самой свежей версии)."""
    m = re.search(r"(\d{4})$", str(s).strip())
    return int(m.group(1)) if m else 0


def _drop_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["object_name"].isin(AGG_REMOVE)].copy()


def load_long_panel(parquet_path: Path | str) -> pd.DataFrame:
    """Загружает long-панель и применяет базовую фильтрацию.

    Возвращает DataFrame с фильтрацией:
      - только object_level == "Регион";
      - только year ∈ YEARS_RANGE;
      - без агрегатов (Тюмень с АО, Архангельск с АО).
    """
    df = pd.read_parquet(parquet_path)
    df = df[
        (df["object_level"] == "Регион") &
        (df["year"].between(*YEARS_RANGE))
    ].copy()
    df = _drop_aggregates(df)
    # Спецзначения → NaN
    for sv in SPECIAL_VAL:
        df.loc[df["indicator_value"] == sv, "indicator_value"] = np.nan
    # Дедупликация по свежести источника
    df["_src_year"] = df["source"].apply(_extract_source_year)
    df = df.sort_values("_src_year").drop_duplicates(
        subset=["indicator_code", "subsection", "object_name", "year"], keep="last"
    )
    # Нормализация единиц
    mult = df["indicator_unit"].map({u: v[0] for u, v in UNIT_NORM.items()}).fillna(1.0)
    df["indicator_value_norm"] = df["indicator_value"] * mult
    df["unit_canonical"] = df["indicator_unit"].map(
        {u: v[1] for u, v in UNIT_NORM.items()}
    ).fillna(df["indicator_unit"])
    return df


def pivot_to_wide(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot long → wide (object_name × year × indicator_code).

    Возвращает (wide, catalog), где catalog — справочник кодов с именами.
    """
    # Выбираем "основные" значения (subsection == 'CD' или единственное)
    subs_per = long_df.groupby("indicator_code")["subsection"].nunique()
    single = subs_per[subs_per == 1].index
    has_cd = long_df[long_df["subsection"] == "CD"]["indicator_code"].unique()
    has_blank = long_df[
        long_df["subsection"].isin(["", "nan", "None"]) | long_df["subsection"].isna()
    ]["indicator_code"].unique()

    mask = (
        long_df["indicator_code"].isin(single) |
        (long_df["indicator_code"].isin(has_cd) & (long_df["subsection"] == "CD")) |
        (~long_df["indicator_code"].isin(has_cd) &
         long_df["indicator_code"].isin(has_blank) &
         (long_df["subsection"].isin(["", "nan", "None"]) | long_df["subsection"].isna()))
    )
    src = long_df[mask].drop_duplicates(
        ["indicator_code", "object_name", "year"], keep="last"
    )
    wide = src.pivot_table(
        index=["object_name", "year"],
        columns="indicator_code",
        values="indicator_value_norm",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # Каталог
    catalog = (
        src[["indicator_code", "indicator_name", "unit_canonical"]]
        .drop_duplicates("indicator_code")
        .reset_index(drop=True)
    )
    return wide, catalog


def attach_macro(wide: pd.DataFrame, macro_path: Path | str) -> pd.DataFrame:
    """Добавляет общероссийские макрофакторы (общие по году)."""
    macro = pd.read_csv(macro_path)
    use = ["year"] + [c for c in MACRO_COLS if c in macro.columns]
    macro = macro[use].query(f"{YEARS_RANGE[0]} <= year <= {YEARS_RANGE[1]}")
    return wide.merge(macro, on="year", how="left")


def compute_real_grp(wide: pd.DataFrame) -> pd.DataFrame:
    """Расчёт реального ВРП на душу населения в ценах 2015 года.

    Алгоритм:
      1. Берём номинальный ВРП на душу (TARGET_NOM, Y477110006), руб.
      2. Берём ИФО ВРП по регионам (GRP_IFO, Y477110109), % к предыдущему году.
      3. Для каждого региона строим real_factor: 1.0 в 2015, рекурсивно по годам.
      4. real_GRP_pc(t) = nominal_GRP_pc(2015) * real_factor(t).

    Если ИФО недоступно — fallback на региональный CPI (Y477110111).

    Возвращает датафрейм с добавленными колонками:
      - grp_real_pc_2015  — реальный ВРП на душу, руб. (в ценах 2015)
      - real_factor       — мультипликатор относительно 2015
      - grp_nom_growth    — YoY номинального ВРП на душу, %
      - grp_real_growth   — YoY реального ВРП, %
      - grp_deflator_src  — источник дефлятора (ifo / cpi / none)
    """
    df = wide.sort_values(["object_name", "year"]).copy()

    # ── Шаг 1. Источник дефлятора по регионам:
    src_choice = {}
    for reg, sub in df.groupby("object_name", sort=False):
        has_ifo = sub[GRP_IFO].notna().sum() if GRP_IFO in sub.columns else 0
        has_cpi = sub[CPI_REG].notna().sum() if CPI_REG in sub.columns else 0
        if has_ifo >= 10:
            src_choice[reg] = "ifo"
        elif has_cpi >= 10:
            src_choice[reg] = "cpi"
        else:
            src_choice[reg] = "none"

    df["grp_deflator_src"] = df["object_name"].map(src_choice)

    # ── Шаг 2. real_factor рекурсивно
    def _build_factor(sub: pd.DataFrame, region_name: str | None = None) -> pd.DataFrame:
        sub = sub.sort_values("year").copy()
        # region_name можно передать снаружи (после groupby имя колонки исчезает)
        if region_name is None:
            region_name = sub["object_name"].iloc[0] if "object_name" in sub.columns else None
        src = src_choice.get(region_name, "none")
        if src == "ifo":
            idx_col = GRP_IFO
        elif src == "cpi":
            idx_col = CPI_REG
        else:
            sub["real_factor"] = np.nan
            return sub
        # Заполняем пропуски ИФО медианой по году (если есть)
        idx = sub[idx_col].astype(float).copy()
        # если 2015 пропущено — fillna(100)
        idx = idx.fillna(100.0)
        years = sub["year"].values
        factors = np.full(len(sub), np.nan)
        # Найти позицию 2015
        if BASE_YEAR not in years:
            return sub.assign(real_factor=np.nan)
        i0 = list(years).index(BASE_YEAR)
        factors[i0] = 1.0
        # Вперёд
        for j in range(i0 + 1, len(sub)):
            growth = idx.iloc[j] / 100.0
            factors[j] = factors[j - 1] * growth
        # Назад
        for j in range(i0 - 1, -1, -1):
            growth_next = idx.iloc[j + 1] / 100.0
            if growth_next > 0:
                factors[j] = factors[j + 1] / growth_next
        sub["real_factor"] = factors
        return sub

    pieces = []
    for reg, sub in df.groupby("object_name", sort=False):
        pieces.append(_build_factor(sub.assign(object_name=reg), region_name=reg))
    df = pd.concat(pieces, ignore_index=True)

    # ── Шаг 3. ВРП на душу в реальных ценах 2015 года
    # base level: nominal GRP per capita at BASE_YEAR
    base_level = (
        df[df["year"] == BASE_YEAR]
        .set_index("object_name")[TARGET_NOM]
        .to_dict()
    )
    df["_base_pc"] = df["object_name"].map(base_level)
    df[GRP_REAL_PC] = df["_base_pc"] * df["real_factor"]
    df.drop(columns=["_base_pc"], inplace=True)

    # ── Шаг 4. Темпы роста (для базового EDA)
    df["grp_nom_growth"] = df.groupby("object_name")[TARGET_NOM].pct_change() * 100
    df["grp_real_growth"] = df.groupby("object_name")[GRP_REAL_PC].pct_change() * 100

    return df


def drop_sparse_columns(wide: pd.DataFrame, threshold: float = 0.60) -> pd.DataFrame:
    """Удаляет признаки с долей пропусков > threshold."""
    keep_always = {
        "object_name", "year", TARGET_NOM, GRP_TOTAL, GRP_IFO, CPI_REG,
        GRP_REAL_PC, "real_factor", "grp_nom_growth", "grp_real_growth",
        "grp_deflator_src",
    } | set(MACRO_COLS)
    feat_cols = [c for c in wide.columns if c not in keep_always]
    miss = wide[feat_cols].isna().mean()
    drop = miss[miss > threshold].index.tolist()
    print(f"  Удаление разреженных признаков (>{int(threshold * 100)}%): {len(drop)}")
    return wide.drop(columns=drop)


def run_preprocess(parquet_path: Path | str = None,
                   macro_path: Path | str = None,
                   verbose: bool = True) -> pd.DataFrame:
    """Полный препроцессинг: parquet → wide-панель с реальным ВРП и макро."""
    if parquet_path is None:
        parquet_path = DATA_RAW / "data_regions_collection_102_v20260313.parquet"
    if macro_path is None:
        macro_path = DATA_RAW / "macro_external.csv"

    if verbose:
        print("=" * 70)
        print(" ПРЕПРОЦЕССИНГ СЫРЫХ ДАННЫХ")
        print("=" * 70)
        print(f"\n  Источник: {Path(parquet_path).name}")
        print(f"  Макро:    {Path(macro_path).name}")

    if verbose: print("\n  [1/6] Загрузка long-панели...")
    long_df = load_long_panel(parquet_path)
    if verbose:
        print(f"        строк: {len(long_df):,}  регионов: {long_df['object_name'].nunique()}  "
              f"индикаторов: {long_df['indicator_code'].nunique():,}")

    if verbose: print("\n  [2/6] Pivot long → wide...")
    wide, catalog = pivot_to_wide(long_df)
    if verbose:
        print(f"        wide: {wide.shape[0]} строк × {wide.shape[1]} колонок")

    if verbose: print("\n  [3/6] Подключение макрофакторов...")
    wide = attach_macro(wide, macro_path)

    if verbose: print(f"\n  [4/6] Расчёт реального ВРП в ценах {BASE_YEAR} года...")
    wide = compute_real_grp(wide)
    src_summary = wide.groupby("grp_deflator_src")["object_name"].nunique()
    if verbose:
        print(f"        Источник дефлятора: {dict(src_summary)}")
        cov = wide[GRP_REAL_PC].notna().sum() / len(wide) * 100
        print(f"        Покрытие реального ВРП: {cov:.1f}%")

    if verbose: print("\n  [5/6] Удаление разреженных колонок...")
    wide = drop_sparse_columns(wide, threshold=0.60)

    if verbose: print("\n  [6/6] Сохранение...")
    out = DATA_PROC / "panel_preprocessed.parquet"
    wide.to_parquet(out, index=False)
    catalog.to_csv(DATA_PROC / "feature_catalog.csv", index=False, encoding="utf-8-sig")
    if verbose:
        print(f"        [OK] {out}  shape={wide.shape}")
        print(f"        [OK] {DATA_PROC / 'feature_catalog.csv'}")

    return wide


if __name__ == "__main__":
    run_preprocess()
