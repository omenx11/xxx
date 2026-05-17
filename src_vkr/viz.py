"""viz.py — построение графиков и карты РФ для ВКР.

Создаёт:
- 01_target_distribution.png    — гистограммы nom/real ВРП на душу;
- 02_grp_dynamics.png            — динамика медианного/средн. ВРП по годам;
- 03_missing_target.png          — пропуски целевой по регионам;
- 04_clusters_profile.png        — профили кластеров (медиана ВРП по типам);
- 05_correlation_heatmap.png     — heatmap корреляции отобранных признаков;
- 06_model_comparison.png        — RMSLE/MAPE моделей walk-forward;
- 07_actual_vs_predicted.png     — scatter факт vs прогноз для лучшей ML;
- 08_mape_by_year.png            — динамика MAPE по моделям и годам;
- 09_shap_summary.png            — топ признаков SHAP;
- 10_region_dynamics.png         — отдельные регионы (Москва, ХМАО, Татарстан...);
- 11_scenario_forecast_<reg>.png — сценарный прогноз для региона;
- 12_corr_target.png             — корреляции с целевой переменной;
- map_actual.html                — карта РФ с фактическим ВРП;
- map_scenarios.html             — карты по 3 сценариям × 5 годам.

Стиль: matplotlib + seaborn, кириллический шрифт.
Карта: Plotly choropleth_mapbox + GeoJSON границ регионов.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Кириллический шрифт — DejaVu есть в matplotlib по умолчанию
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 110
sns.set_style("whitegrid")

from .config import (
    DATA_RAW, DATA_PROC, RESULTS, FIGURES, TABLES, MAPS,
    TARGET_NOM, GRP_REAL_PC, MACRO_COLS, TRAIN_END, TEST_START, TEST_END,
    BASE_YEAR, CLUSTER_NAMES, FORECAST_START, FORECAST_END,
)
from .typology import get_cluster

warnings.filterwarnings("ignore")


# ── 1. EDA: target distribution ─────────────────────────────────────────────
def fig_target_distribution(df: pd.DataFrame, save: bool = True) -> Path:
    last_year = int(df["year"].max())
    sub = df[df["year"] == last_year]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(sub[TARGET_NOM].dropna() / 1e6, bins=30, color="#3a78b0",
                 edgecolor="white")
    axes[0].set_title(f"Распределение номинального ВРП на душу нас., {last_year}")
    axes[0].set_xlabel("млн руб./чел.")
    axes[0].set_ylabel("Число регионов")
    axes[0].axvline(sub[TARGET_NOM].median() / 1e6, color="firebrick",
                    linestyle="--", label=f"медиана = {sub[TARGET_NOM].median()/1e6:.2f}")
    axes[0].legend()

    axes[1].hist(sub[GRP_REAL_PC].dropna() / 1e6, bins=30, color="#3a78b0",
                 edgecolor="white")
    axes[1].set_title(f"Распределение реального ВРП в ценах {BASE_YEAR}, {last_year}")
    axes[1].set_xlabel("млн руб./чел. (в ценах 2015)")
    axes[1].axvline(sub[GRP_REAL_PC].median() / 1e6, color="firebrick",
                    linestyle="--", label=f"медиана = {sub[GRP_REAL_PC].median()/1e6:.2f}")
    axes[1].legend()
    plt.tight_layout()
    out = FIGURES / "01_target_distribution.png"
    if save:
        fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_grp_dynamics(df: pd.DataFrame, save: bool = True) -> Path:
    g = df.groupby("year").agg(
        nom_median=(TARGET_NOM, "median"),
        nom_mean=(TARGET_NOM, "mean"),
        real_median=(GRP_REAL_PC, "median"),
        real_mean=(GRP_REAL_PC, "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(g["year"], g["nom_median"] / 1e6, "o-", color="#3a78b0",
            label="Ном. ВРП/чел., медиана")
    ax.plot(g["year"], g["real_median"] / 1e6, "s-", color="#d44",
            label=f"Реал. ВРП/чел. (цены {BASE_YEAR}), медиана")
    ax.fill_between(g["year"], g["nom_median"] / 1e6, g["real_median"] / 1e6,
                    alpha=0.1, color="#888", label="Эффект инфляции")
    ax.set_xlabel("Год")
    ax.set_ylabel("млн руб./чел.")
    ax.set_title("Динамика ВРП на душу населения в РФ: номинальный vs реальный")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIGURES / "02_grp_dynamics.png"
    if save:
        fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_missing_target(df: pd.DataFrame, save: bool = True) -> Path:
    n_years = df["year"].nunique()
    miss = (
        df.groupby("object_name")[TARGET_NOM]
        .apply(lambda x: x.isna().sum())
        .reset_index(name="missing")
        .sort_values("missing", ascending=False)
    )
    miss["coverage_pct"] = (1 - miss["missing"] / n_years) * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(miss)), miss["coverage_pct"], color="#3a78b0")
    ax.set_xlabel("Регионы (отсортированы по покрытию)")
    ax.set_ylabel("% непустых наблюдений")
    ax.set_title(f"Покрытие целевой переменной по регионам ({n_years} лет)")
    ax.axhline(80, color="firebrick", linestyle="--", label="80%")
    ax.legend()
    plt.tight_layout()
    out = FIGURES / "03_missing_target.png"
    if save:
        fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_clusters_profile(df: pd.DataFrame, save: bool = True) -> Path:
    df = df.copy()
    df["cluster_id"] = df["object_name"].apply(get_cluster)
    last_3y = df[df["year"] >= df["year"].max() - 2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    palette = sns.color_palette("Set2", n_colors=4)

    for cl, sub in last_3y.groupby("cluster_id"):
        axes[0].hist(sub[TARGET_NOM].dropna() / 1e6, bins=20,
                     color=palette[int(cl)], alpha=0.6,
                     label=CLUSTER_NAMES.get(int(cl), f"cl{cl}"))
    axes[0].set_xlabel("ВРП/чел., млн руб.")
    axes[0].set_title(f"Распределение по типам регионов ({df['year'].max()-2}–{df['year'].max()})")
    axes[0].legend(loc="upper right", fontsize=9)

    g = df.groupby(["year", "cluster_id"])[TARGET_NOM].median().reset_index()
    for cl, sub in g.groupby("cluster_id"):
        axes[1].plot(sub["year"], sub[TARGET_NOM] / 1e6, "o-",
                     color=palette[int(cl)],
                     label=CLUSTER_NAMES.get(int(cl), f"cl{cl}"))
    axes[1].set_xlabel("Год")
    axes[1].set_ylabel("Медиана ВРП/чел., млн руб.")
    axes[1].set_title("Динамика по типам регионов")
    axes[1].legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    out = FIGURES / "04_clusters_profile.png"
    if save:
        fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_correlation_heatmap(df: pd.DataFrame, features: list[str],
                              max_feat: int = 18, save: bool = True) -> Path:
    feat = features[:max_feat]
    sub = df[df["year"] <= TRAIN_END][feat + ["nom_log"]].dropna()
    corr = sub.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", cbar_kws={"label": "ρ Пирсона"},
                annot_kws={"size": 7}, square=False, ax=ax)
    ax.set_title("Корреляционная матрица отобранных признаков")
    plt.tight_layout()
    out = FIGURES / "05_correlation_heatmap.png"
    if save:
        fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_corr_with_target(df: pd.DataFrame, features: list[str],
                          save: bool = True) -> Path:
    train = df[df["year"] <= TRAIN_END]
    nom_corr = train[features].corrwith(train["nom_log"]).abs().sort_values()
    real_corr = train[features].corrwith(train["real_log"]).abs().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    nom_corr.tail(15).plot.barh(ax=axes[0], color="#3a78b0")
    axes[0].set_title("Топ-15 корреляций с ном. log(ВРП/чел.)")
    real_corr.tail(15).plot.barh(ax=axes[1], color="#c44")
    axes[1].set_title("Топ-15 корреляций с реал. log(ВРП/чел.)")
    plt.tight_layout()
    out = FIGURES / "12_corr_target.png"
    if save:
        fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 6. Model comparison ─────────────────────────────────────────────────────
def fig_model_comparison(summary: pd.DataFrame, save: bool = True) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    s = summary.sort_values("RMSLE")
    colors = ["#3a78b0" if m not in ("Naive", "MeanGrowth") else "#cc8400"
              for m in s.index]
    axes[0].barh(s.index, s["RMSLE"], color=colors)
    axes[0].set_title("RMSLE (меньше = лучше)")
    axes[0].set_xlabel("RMSLE на log(ВРП/чел.)")
    axes[0].invert_yaxis()

    s2 = summary.sort_values("MAPE")
    colors2 = ["#3a78b0" if m not in ("Naive", "MeanGrowth") else "#cc8400"
               for m in s2.index]
    axes[1].barh(s2.index, s2["MAPE"], color=colors2)
    axes[1].set_title("MAPE, % (меньше = лучше)")
    axes[1].set_xlabel("MAPE")
    axes[1].invert_yaxis()
    plt.suptitle(f"Walk-forward CV: {TEST_START}–{TEST_END}", y=1.02)
    plt.tight_layout()
    out = FIGURES / "06_model_comparison.png"
    if save:
        fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_actual_vs_predicted(oof: pd.DataFrame, model_name: str,
                              save: bool = True) -> Path:
    sub = oof[oof["model"] == model_name]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(np.log1p(sub["y_true"]), np.log1p(sub["y_pred"]),
               alpha=0.5, s=18, color="#3a78b0")
    lo = min(np.log1p(sub["y_true"]).min(), np.log1p(sub["y_pred"]).min())
    hi = max(np.log1p(sub["y_true"]).max(), np.log1p(sub["y_pred"]).max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = ŷ")
    ax.set_xlabel("log(факт + 1)")
    ax.set_ylabel("log(прогноз + 1)")
    ax.set_title(f"Факт vs прогноз — {model_name} (walk-forward)")
    ax.legend()
    plt.tight_layout()
    out = FIGURES / "07_actual_vs_predicted.png"
    if save:
        fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_mape_by_year(per_year: pd.DataFrame, save: bool = True) -> Path:
    """MAPE по годам для основных моделей."""
    keep = ["MeanGrowth", "Naive", "RandomForest", "GradientBoosting",
            "LightGBM", "XGBoost", "Ridge", "ElasticNet"]
    sub = per_year[per_year["model"].isin(keep)]
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, g in sub.groupby("model"):
        ax.plot(g["year"], g["MAPE"], "o-", label=name)
    ax.set_xlabel("Год тестового периода")
    ax.set_ylabel("MAPE, %")
    ax.set_title("Динамика MAPE моделей по тестовым годам")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIGURES / "08_mape_by_year.png"
    if save:
        fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_shap_summary(shap_df: pd.DataFrame, save: bool = True) -> Path:
    s = shap_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(s["feature"], s["mean_abs_shap"], color="#3a78b0")
    ax.set_xlabel("Среднее |SHAP|")
    ax.set_title("Топ-15 признаков по средней значимости SHAP")
    plt.tight_layout()
    out = FIGURES / "09_shap_summary.png"
    if save:
        fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_region_dynamics(df: pd.DataFrame, save: bool = True) -> Path:
    """Динамика для 6 выбранных регионов (по 1 на кластер + Москва + ХМАО)."""
    candidates = [
        "Москва", "г. Москва", "Московская область",
        "Ханты-Мансийский автономный округ - Югра",
        "Республика Татарстан", "Свердловская область",
        "Республика Дагестан", "Чеченская Республика",
    ]
    chosen = []
    for c in candidates:
        if (df["object_name"] == c).any():
            chosen.append(c)
        if len(chosen) >= 6:
            break
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("tab10", n_colors=len(chosen))
    for c, color in zip(chosen, colors):
        sub = df[df["object_name"] == c].sort_values("year")
        ax.plot(sub["year"], sub[GRP_REAL_PC] / 1e6, "o-", color=color, label=c)
    ax.set_xlabel("Год")
    ax.set_ylabel(f"ВРП/чел. в ценах {BASE_YEAR}, млн руб.")
    ax.set_title("Динамика реального ВРП на душу: ключевые регионы")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIGURES / "10_region_dynamics.png"
    if save:
        fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_scenario_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame,
                            region: str, target: str = "nominal",
                            save: bool = True) -> Path:
    """Прогноз 3 сценариев для одного региона."""
    sub_hist = df[df["object_name"] == region].sort_values("year")
    fc = forecast_df[forecast_df["object_name"] == region]

    target_col = TARGET_NOM if target == "nominal" else GRP_REAL_PC
    y_col = "y_pred_nominal" if target == "nominal" else "y_pred_real_2015"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sub_hist["year"], sub_hist[target_col] / 1e6, "o-",
            color="#222", label="Факт")
    color_map = {"baseline": "#3a78b0",
                  "optimistic": "#2ecc71",
                  "pessimistic": "#e74c3c"}
    for sc, sub in fc.groupby("scenario"):
        sub = sub.sort_values("year")
        ax.plot(sub["year"], sub[y_col] / 1e6, "s--",
                color=color_map.get(sc, "gray"), label=f"Прогноз ({sc})")
        if sc == "baseline" and "y_lo_nominal" in sub.columns and target == "nominal":
            ax.fill_between(sub["year"],
                            sub["y_lo_nominal"] / 1e6,
                            sub["y_hi_nominal"] / 1e6,
                            color=color_map[sc], alpha=0.18,
                            label="90% CI базовый")
    ax.axvline(df["year"].max() + 0.5, color="gray", linestyle=":", alpha=0.6)
    title_y = "номинального ВРП/чел." if target == "nominal" else f"реал. ВРП/чел. (цены {BASE_YEAR})"
    ax.set_title(f"Сценарный прогноз {title_y} — {region}")
    ax.set_xlabel("Год")
    ax.set_ylabel("млн руб./чел.")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIGURES / f"11_scenario_{target}_{_safe_name(region)}.png"
    if save:
        fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def _safe_name(s: str) -> str:
    return (
        s.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
         .replace(",", "").replace(".", "")
    )


# ── Map of Russia (choropleth) ──────────────────────────────────────────────
def _load_geojson() -> dict | None:
    """Пытается загрузить GeoJSON с границами регионов РФ.

    Ищет в следующем порядке:
      - data/raw/ru_regions.geojson
      - data/raw/russia.geojson
    Если нет — пытается скачать из стандартного источника (если есть кеш).
    """
    for name in ["ru_regions.geojson", "russia.geojson",
                 "russia_regions.geojson"]:
        p = DATA_RAW / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return None


# Унификация названий регионов (data → geojson click_that_hood)
# GeoJSON содержит 83 региона без Крыма/Севастополя
REGION_NAME_MAP = {
    "г. Москва":  "Москва",
    "г. Санкт-Петербург": "Санкт-Петербург",
    "г. Севастополь": "Севастополь",
    "Республика Адыгея": "Адыгея",
    "Республика Алтай":  "Алтай",
    "Республика Башкортостан": "Башкортостан",
    "Республика Бурятия": "Бурятия",
    "Республика Дагестан": "Дагестан",
    "Республика Ингушетия": "Ингушетия",
    "Республика Калмыкия": "Республика Калмыкия",
    "Республика Карелия":  "Республика Карелия",
    "Республика Коми":     "Республика Коми",
    "Республика Марий Эл": "Марий Эл",
    "Республика Мордовия": "Республика Мордовия",
    "Республика Саха (Якутия)": "Республика Саха (Якутия)",
    "Республика Татарстан":     "Татарстан",
    "Республика Тыва":          "Тыва",
    "Республика Хакасия":       "Республика Хакасия",
    "Чеченская Республика":     "Чеченская республика",
    "Кабардино-Балкарская Республика": "Кабардино-Балкарская республика",
    "Карачаево-Черкесская Республика": "Карачаево-Черкесская республика",
    "Удмуртская Республика":           "Удмуртская республика",
    "Чувашская Республика - Чувашия":  "Чувашия",
    "Чувашская Республика":            "Чувашия",
    "Республика Северная Осетия - Алания": "Северная Осетия - Алания",
    "Архангельская область (без автономного округа)": "Архангельская область",
    "Тюменская область (без автономных округов)":     "Тюменская область",
    "Кемеровская область - Кузбасс": "Кемеровская область",
    "Кемеровская область — Кузбасс": "Кемеровская область",
    # Варианты с em-dash (часто в Росстате):
    "Республика Северная Осетия — Алания": "Северная Осетия - Алания",
    "Ханты-Мансийский автономный округ — Югра":
        "Ханты-Мансийский автономный округ - Югра",
}


def normalize_region(name: str) -> str:
    """Приводит имя региона к виду в GeoJSON click_that_hood.

    Дополнительно убирает разделители (em-dash, ndash) к обычному дефису:
    в данных Росстата часто em-dash (U+2014), а в GeoJSON — обычный дефис.
    """
    # 1) явный mapping
    if name in REGION_NAME_MAP:
        return REGION_NAME_MAP[name]
    # 2) нормализация дефисов
    n = name.replace("—", "-").replace("–", "-")
    if n in REGION_NAME_MAP:
        return REGION_NAME_MAP[n]
    return n


def make_choropleth(df: pd.DataFrame, value_col: str,
                     title: str, output_html: Path | str,
                     hover_data: dict | None = None,
                     colorscale: str = "Viridis",
                     log_scale: bool = True) -> Path | None:
    """Choropleth-карта России с границами регионов.

    df       : DataFrame с object_name + value_col;
    value_col: имя колонки со значением для раскраски.

    Использует GeoJSON из data/raw/ru_regions.geojson.
    Логарифмическая шкала по умолчанию (т.к. ВРП на душу скошён в 10×).
    """
    import plotly.express as px
    import plotly.graph_objects as go

    geojson = _load_geojson()
    if geojson is None:
        print("  [WARN] GeoJSON границ регионов РФ не найден в data/raw/. "
              "Поставьте ru_regions.geojson для choropleth-карты.")
        return None

    df = df.copy()
    df["region_norm"] = df["object_name"].apply(normalize_region)
    df = df.dropna(subset=[value_col])

    # Логирование (только для положительных)
    if log_scale and (df[value_col] > 0).all():
        df["_color_value"] = np.log10(df[value_col].clip(lower=1))
        cb_title = f"{value_col} (log₁₀)"
    else:
        df["_color_value"] = df[value_col]
        cb_title = value_col

    feat0 = geojson["features"][0]
    candidates = ["name", "NAME", "name_ru", "region", "NAME_1"]
    key = None
    for c in candidates:
        if c in feat0.get("properties", {}):
            key = c
            break
    if key is None:
        for k, v in feat0.get("properties", {}).items():
            if isinstance(v, str):
                key = k; break
    if key is None:
        print("  [WARN] не определено поле имени региона в GeoJSON")
        return None

    # Проверка совпадения имён
    geo_names = {f["properties"][key] for f in geojson["features"]}
    matched = df["region_norm"].isin(geo_names).sum()
    if matched < 50:
        print(f"  [WARN] Сопоставлено только {matched}/{len(df)} регионов "
              f"с GeoJSON — проверьте REGION_NAME_MAP")

    # Hover-текст в исходных единицах
    fmt = lambda v: f"{v:,.0f} руб.".replace(",", " ")
    hover_text = [
        f"<b>{r}</b><br>{value_col}: {fmt(v)}"
        for r, v in zip(df["object_name"], df[value_col])
    ]
    if hover_data:
        for col in hover_data:
            if col in df.columns:
                hover_text = [
                    t + f"<br>{col}: {fmt(v)}" if isinstance(v, (int, float)) and not pd.isna(v)
                    else t + f"<br>{col}: {v}"
                    for t, v in zip(hover_text, df[col].values)
                ]

    fig = go.Figure(go.Choropleth(
        geojson=geojson,
        locations=df["region_norm"],
        z=df["_color_value"],
        featureidkey=f"properties.{key}",
        colorscale=colorscale,
        marker_line_width=0.6,
        marker_line_color="white",
        text=hover_text,
        hoverinfo="text",
        colorbar=dict(title=cb_title, thickness=12, len=0.7),
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        title=title,
        height=650,
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#f8fafc",
        template="plotly_white",
    )
    fig.update_geos(
        fitbounds="locations",
        visible=True,
        projection_type="mercator",
        bgcolor="#f8fafc",
        showframe=False,
        showcoastlines=False,
        showcountries=False,
        showland=True,
        landcolor="#eef2f7",
        showocean=True,
        oceancolor="#eaf3fb",
        showlakes=True,
        lakecolor="#eaf3fb",
    )
    output_html = Path(output_html)
    fig.write_html(output_html, include_plotlyjs="cdn")
    return output_html


def make_bubble_map(df: pd.DataFrame, value_col: str, title: str,
                     output_html: Path | str,
                     hover_extra: list[str] | None = None,
                     colorscale: str = "Viridis") -> Path:
    """Bubble-map России на основе region_centroids.json.

    Fallback к scattermapbox: круги размером пропорциональны value_col и
    раскрашены по value_col. Используется, когда GeoJSON границ нет.
    """
    import plotly.graph_objects as go

    cent_path = DATA_RAW / "region_centroids.json"
    if not cent_path.exists():
        print(f"  [WARN] {cent_path} нет, карта пропущена.")
        return None
    with open(cent_path, encoding="utf-8") as f:
        cent = json.load(f)

    df = df.copy()
    df["region_norm"] = df["object_name"].apply(normalize_region)
    df["lat"] = df["object_name"].map({k: v[0] for k, v in cent.items()})
    df["lon"] = df["object_name"].map({k: v[1] for k, v in cent.items()})
    # на случай если данные используют альтернативное имя
    df.loc[df["lat"].isna(), "lat"] = df.loc[df["lat"].isna(), "region_norm"].map(
        {k: v[0] for k, v in cent.items()})
    df.loc[df["lon"].isna(), "lon"] = df.loc[df["lon"].isna(), "region_norm"].map(
        {k: v[1] for k, v in cent.items()})
    df = df.dropna(subset=["lat", "lon", value_col])

    v = df[value_col].clip(lower=0)
    sizes = np.log1p(v.values)
    sizes = (sizes - sizes.min() + 1)
    sizes = sizes / max(sizes.max(), 1) * 40 + 8

    hovertxt = ["<b>" + r + "</b><br>" + value_col +
                f": {val:,.0f}".replace(",", " ")
                for r, val in zip(df["object_name"], df[value_col])]
    if hover_extra:
        for col in hover_extra:
            if col in df.columns:
                hovertxt = [t + f"<br>{col}: {v}"
                            for t, v in zip(hovertxt, df[col].values)]

    fig = go.Figure(go.Scattermapbox(
        lat=df["lat"], lon=df["lon"],
        mode="markers",
        marker=go.scattermapbox.Marker(
            size=sizes,
            color=df[value_col],
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=value_col),
            opacity=0.78,
        ),
        text=hovertxt,
        hoverinfo="text",
    ))
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(zoom=2.4, center=dict(lat=65, lon=95)),
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    output_html = Path(output_html)
    fig.write_html(output_html, include_plotlyjs="cdn")
    return output_html


def make_all_maps(df: pd.DataFrame, forecast_df: pd.DataFrame,
                   last_year: int) -> list[Path]:
    """Создаёт все карты для ВКР.

    Если GeoJSON границ регионов отсутствует — используем bubble-map
    на основе region_centroids.json (круги для каждого региона).
    """
    outs = []

    def try_both(df_in, val_col, title, fname, hover):
        # Сначала пробуем choropleth, если не получилось — bubble
        p = make_choropleth(df_in, val_col, title, MAPS / fname, hover_data=hover)
        if p is None:
            p = make_bubble_map(df_in, val_col, title, MAPS / fname,
                                hover_extra=list(hover.keys()) if hover else None)
        return p

    # 1) Факт за последний год
    last = df[df["year"] == last_year][["object_name", TARGET_NOM, GRP_REAL_PC]].copy()
    last = last.rename(columns={TARGET_NOM: "grp_nom",
                                GRP_REAL_PC: "grp_real_2015"})
    p1 = try_both(last, "grp_nom",
                   f"Номинальный ВРП на душу нас., {last_year} (факт)",
                   f"map_actual_nominal_{last_year}.html",
                   {"grp_real_2015": True})
    if p1: outs.append(p1)
    p2 = try_both(last, "grp_real_2015",
                   f"Реальный ВРП на душу (цены {BASE_YEAR}), {last_year} (факт)",
                   f"map_actual_real_{last_year}.html",
                   {"grp_nom": True})
    if p2: outs.append(p2)

    # 2) Сценарии × горизонты
    for sc in ["baseline", "optimistic", "pessimistic"]:
        for h in [1, 3, 5]:
            fc = forecast_df[(forecast_df["scenario"] == sc) &
                             (forecast_df["horizon"] == h)].copy()
            if len(fc) == 0:
                continue
            yr = int(fc["year"].iloc[0])
            p = try_both(fc, "y_pred_nominal",
                          f"Ном. ВРП/чел., прогноз ({sc}, год {yr}, h={h})",
                          f"map_forecast_{sc}_h{h}_nominal.html",
                          {"y_pred_real_2015": True})
            if p: outs.append(p)
            p = try_both(fc, "y_pred_real_2015",
                          f"Реал. ВРП/чел. (цены {BASE_YEAR}), прогноз ({sc}, {yr})",
                          f"map_forecast_{sc}_h{h}_real.html",
                          {"y_pred_nominal": True})
            if p: outs.append(p)
    return outs
