"""app_vkr/streamlit_app.py — интерфейс аналитической системы ВКР.

Запуск: streamlit run app_vkr/streamlit_app.py
"""
from __future__ import annotations
from copy import deepcopy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import folium
import branca.colormap as cm
from streamlit_folium import st_folium

from src_vkr.config import (
    DATA_RAW, DATA_PROC, RESULTS, BASE_YEAR, MACRO_COLS, TARGET_NOM,
    GRP_REAL_PC, TRAIN_END, TEST_START, TEST_END, FORECAST_START, FORECAST_END,
    CLUSTER_NAMES,
)
from src_vkr.typology import get_cluster
from src_vkr.viz import _load_geojson, normalize_region
from src_vkr import viz_common as vc


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ВКР: Прогноз ВРП регионов РФ",
    page_icon="📊", layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_panel():
    return pd.read_parquet(DATA_PROC / "panel_features.parquet")


@st.cache_data(show_spinner=False)
def load_results():
    summary = pd.read_csv(RESULTS / "model_summary.csv", index_col="model")
    oof = pd.read_csv(RESULTS / "oof_predictions.csv")
    fc = pd.read_csv(RESULTS / "forecast_all_scenarios.csv")
    shap_g = pd.read_csv(RESULTS / "shap_global.csv")
    try:
        shap_c = pd.read_csv(RESULTS / "shap_per_cluster.csv")
    except Exception:
        shap_c = pd.DataFrame()
    try:
        shap_r = pd.read_csv(RESULTS / "shap_per_region.csv")
    except Exception:
        shap_r = pd.DataFrame()
    try:
        py = pd.read_csv(RESULTS / "per_year_metrics.csv")
    except Exception:
        py = pd.DataFrame()
    return summary, oof, fc, shap_g, shap_c, shap_r, py


@st.cache_data(show_spinner=False)
def load_aux():
    with open(DATA_RAW / "region_to_fo.json", encoding="utf-8") as f:
        fo = json.load(f)
    return fo


@st.cache_data(show_spinner=False)
def load_map_sources():
    geojson = _load_geojson()
    with open(DATA_RAW / "region_centroids.json", encoding="utf-8") as f:
        centroids = json.load(f)
    return geojson, centroids


df = load_panel()
summary, oof, fc, shap_g, shap_c, shap_r, py = load_results()
region_to_fo = load_aux()
geojson_map, region_centroids = load_map_sources()
df["federal_district"] = df["object_name"].map(region_to_fo).fillna("Прочее")
df["cluster_id"] = df["object_name"].apply(get_cluster)
df["cluster_name"] = df["cluster_id"].map(CLUSTER_NAMES)

REGIONS = sorted(df["object_name"].unique())
LAST_YEAR = int(df["year"].max())
ML_NAMES = [m for m in summary.index if m not in ("Naive", "MeanGrowth")]
BEST_ML = ML_NAMES[0] if ML_NAMES else summary.index[0]


# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("📊 ВКР: ВРП регионов РФ")
st.sidebar.caption("ML-прогнозирование социально-экономического развития")
PAGE = st.sidebar.radio(
    "Раздел",
    [
        "🏠 Главная",
        "📈 Анализ данных",
        "🎯 Прогноз региона",
        "📊 Сценарный анализ",
        "🗺️ Карта регионов",
        "🔍 Интерпретация модели",
        "📚 Методология",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Данные:** {df['year'].min()}–{df['year'].max()}, "
    f"{df['object_name'].nunique()} регионов\n\n"
    f"**Тест:** {TEST_START}–{TEST_END}\n\n"
    f"**Прогноз:** {FORECAST_START}–{FORECAST_END}\n\n"
    f"**Лучшая ML:** {BEST_ML}"
)


def fmt_money(v):
    if pd.isna(v):
        return "—"
    return f"{v / 1e6:.2f} млн руб."


SCENARIO_ORDER = ["baseline", "optimistic", "pessimistic"]
SCENARIO_LABELS = {
    "baseline": "Базовый",
    "optimistic": "Оптимистичный",
    "pessimistic": "Пессимистичный",
}
SCENARIO_NOTES = {
    "baseline": "инерционное продолжение текущей траектории",
    "optimistic": "более мягкие макроусловия и умеренный сценарный рост с 2025 года",
    "pessimistic": "жёсткие макроусловия и отрицательная сценарная корректировка с 2025 года",
}


def fmt_pct(v):
    if pd.isna(v):
        return "—"
    return f"{v:.1f}%"


def scenario_value_col(target_key: str) -> str:
    return "y_pred_nominal" if target_key == "nominal" else "y_pred_real_2015"


def scenario_gap_frame(fc_in: pd.DataFrame, target_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_col = scenario_value_col(target_key)
    pivot = (
        fc_in.pivot_table(
            index=["object_name", "year", "horizon"],
            columns="scenario",
            values=y_col,
        )
        .dropna(subset=["baseline", "optimistic", "pessimistic"])
        .reset_index()
    )
    pivot["gap_abs"] = pivot["optimistic"] - pivot["pessimistic"]
    pivot["gap_pct"] = pivot["gap_abs"] / pivot["baseline"].replace(0, np.nan) * 100
    by_h = (
        pivot.groupby(["horizon", "year"], as_index=False)
        .agg(
            baseline_median=("baseline", "median"),
            optimistic_median=("optimistic", "median"),
            pessimistic_median=("pessimistic", "median"),
            gap_abs_median=("gap_abs", "median"),
            gap_pct_median=("gap_pct", "median"),
            gap_pct_p75=("gap_pct", lambda x: x.quantile(0.75)),
        )
    )
    return pivot, by_h


def fig_scenario_medians(fc_in: pd.DataFrame, target_key: str) -> go.Figure:
    y_col = scenario_value_col(target_key)
    g = (
        fc_in.groupby(["scenario", "year"], as_index=False)[y_col]
        .median()
        .sort_values(["scenario", "year"])
    )
    fig = go.Figure()
    for sc in SCENARIO_ORDER:
        sub = g[g["scenario"] == sc]
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub[y_col] / 1e6,
            mode="lines+markers",
            name=SCENARIO_LABELS[sc],
            line=dict(color=vc.SCENARIO_COLORS.get(sc, "#555"), width=3),
        ))
    label = "номинальный" if target_key == "nominal" else f"реальный, цены {BASE_YEAR}"
    fig.update_layout(
        title=f"Медианный прогноз по сценариям: {label}",
        xaxis_title="Год",
        yaxis_title="млн руб./чел.",
        template="plotly_white",
        height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


MAP_PALETTES = {
    "YlGnBu": ["#ffffd9", "#edf8b1", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8"],
    "OrRd": ["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#e34a33", "#b30000"],
    "Greens": ["#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#31a354", "#006d2c"],
    "Reds": ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"],
    "YlOrRd": ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"],
}


def fmt_map_value(v, percent: bool = False):
    if pd.isna(v):
        return "нет данных"
    if percent:
        return f"{v:.1f}%"
    return f"{v / 1e6:.2f} млн руб."


def _geojson_name_key(geojson: dict) -> str:
    props = geojson["features"][0].get("properties", {})
    for key in ["name", "NAME", "name_ru", "region", "NAME_1"]:
        if key in props:
            return key
    for key, value in props.items():
        if isinstance(value, str):
            return key
    return "name"


def _centroid_for(region: str, centroids: dict):
    if region in centroids:
        return centroids[region]
    norm = normalize_region(region)
    for name, coords in centroids.items():
        if normalize_region(name) == norm:
            return coords
    return None


def build_region_map(
    data: pd.DataFrame,
    value_col: str,
    title: str,
    palette_name: str,
    log_scale: bool,
    percent: bool = False,
):
    if geojson_map is None:
        return None

    data = data.dropna(subset=[value_col]).copy()
    data["region_norm"] = data["object_name"].apply(normalize_region)
    data["_display_value"] = data[value_col].map(lambda v: fmt_map_value(v, percent))
    data["_color_value"] = data[value_col]
    if log_scale and (data[value_col] > 0).all():
        data["_color_value"] = np.log10(data[value_col].clip(lower=1))

    q_low, q_high = data["_color_value"].quantile([0.03, 0.97])
    vmin = float(q_low) if pd.notna(q_low) else float(data["_color_value"].min())
    vmax = float(q_high) if pd.notna(q_high) else float(data["_color_value"].max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(data["_color_value"].min()) if len(data) else 0.0
        vmax = vmin + 1.0

    palette = MAP_PALETTES.get(palette_name, MAP_PALETTES["YlGnBu"])
    caption = title + (" (лог-шкала цвета)" if log_scale else "")
    colormap = cm.LinearColormap(palette, vmin=vmin, vmax=vmax, caption=caption)

    name_key = _geojson_name_key(geojson_map)
    geo = deepcopy(geojson_map)
    by_region = data.set_index("region_norm")
    for feature in geo["features"]:
        props = feature.setdefault("properties", {})
        region_name = props.get(name_key)
        if region_name in by_region.index:
            row = by_region.loc[region_name]
            props["_value_fmt"] = row["_display_value"]
            props["_fo"] = row.get("federal_district", "—")
            props["_has_data"] = True
        else:
            props["_value_fmt"] = "нет данных"
            props["_fo"] = "—"
            props["_has_data"] = False

    color_values = by_region["_color_value"].to_dict()

    def fill_color(region_name):
        value = color_values.get(region_name)
        if value is None or pd.isna(value):
            return "#d9e2ec"
        return colormap(float(np.clip(value, vmin, vmax)))

    def style_function(feature):
        region_name = feature["properties"].get(name_key)
        return {
            "fillColor": fill_color(region_name),
            "color": "#ffffff",
            "weight": 0.75,
            "fillOpacity": 0.86 if feature["properties"].get("_has_data") else 0.28,
        }

    fmap = folium.Map(
        location=[64.0, 96.0],
        zoom_start=3,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
    )
    folium.GeoJson(
        geo,
        name="Границы регионов",
        style_function=style_function,
        highlight_function=lambda _: {
            "weight": 2.2,
            "color": "#111827",
            "fillOpacity": 0.94,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[name_key, "_value_fmt", "_fo"],
            aliases=["Регион", "Значение", "ФО"],
            sticky=True,
            localize=True,
        ),
    ).add_to(fmap)

    geo_names = {feature["properties"].get(name_key) for feature in geo["features"]}
    missing = data[~data["region_norm"].isin(geo_names)]
    for _, row in missing.iterrows():
        coords = _centroid_for(row["object_name"], region_centroids)
        if not coords:
            continue
        folium.CircleMarker(
            location=[coords[0], coords[1]],
            radius=8,
            color="#111827",
            weight=1.5,
            fill=True,
            fill_color=fill_color(row["region_norm"]),
            fill_opacity=0.92,
            tooltip=(
                f"{row['object_name']}<br>"
                f"Значение: {row['_display_value']}<br>"
                f"ФО: {row.get('federal_district', '—')}"
            ),
        ).add_to(fmap)

    coords = [
        _centroid_for(region, region_centroids)
        for region in data["object_name"].tolist()
    ]
    coords = [c for c in coords if c and len(c) == 2]
    if coords:
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(18, 18))

    colormap.add_to(fmap)
    folium.LayerControl(position="topright", collapsed=True).add_to(fmap)
    return fmap


# ═══ 1. ГЛАВНАЯ ═════════════════════════════════════════════════════════════
if PAGE.startswith("🏠"):
    st.title("Аналитическая система прогнозирования ВРП на душу населения")
    st.caption(
        "**Тема ВКР:** Разработка алгоритмов машинного обучения для "
        "прогнозирования показателей социально-экономического развития."
    )

    last = df[df["year"] == LAST_YEAR]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Регионов", f"{df['object_name'].nunique()}")
    c2.metric("Лет данных", f"{df['year'].nunique()}")
    c3.metric(f"Медиана ВРП/чел. {LAST_YEAR}",
              fmt_money(last[TARGET_NOM].median()))
    c4.metric(f"Реал. ВРП ({BASE_YEAR})",
              fmt_money(last[GRP_REAL_PC].median()))
    if BEST_ML in summary.index:
        c5.metric(f"MAPE {BEST_ML}", f"{summary.loc[BEST_ML, 'MAPE']:.1f}%")

    st.markdown("### Динамика медианного ВРП по годам")
    st.plotly_chart(vc.fig_grp_dynamics(df), width="stretch")

    st.markdown("### Краткое описание")
    st.markdown(f"""
- **Объект:** 85 субъектов РФ.
- **Период:** {df['year'].min()}–{df['year'].max()}.
- **Целевая переменная:** ВРП на душу населения (номинальный + реальный
  в ценах {BASE_YEAR}).
- **Источник данных:** Росстат / ЕМИСС, World Bank, Банк России.
- **Walk-forward CV:** train ≤ {TRAIN_END}, test ∈ [{TEST_START}; {TEST_END}].
- **Сценарии прогноза:** базовый, оптимистичный, пессимистичный.
""")


# ═══ 2. АНАЛИЗ ДАННЫХ ═══════════════════════════════════════════════════════
elif PAGE.startswith("📈"):
    st.title("📈 Анализ данных")
    tab1, tab2, tab3 = st.tabs(["Распределения", "Корреляции", "Кластеры"])

    with tab1:
        yr = st.slider("Год", int(df["year"].min()), LAST_YEAR, LAST_YEAR)
        st.plotly_chart(vc.fig_target_distribution(df, year=yr),
                         width="stretch")
        st.markdown("##### Таблица регионов")
        sub = df[df["year"] == yr]
        view = sub[["object_name", "federal_district", "cluster_name",
                     TARGET_NOM, GRP_REAL_PC]].copy()
        view.columns = ["Регион", "ФО", "Тип", "ВРП ном., руб.",
                        f"ВРП реал. {BASE_YEAR}, руб."]
        st.dataframe(
            view.sort_values("ВРП ном., руб.", ascending=False),
            width="stretch", height=400, hide_index=True,
        )

    with tab2:
        feats = pd.read_csv(DATA_PROC / "selected_features.csv")["feature"].tolist()
        st.plotly_chart(vc.fig_correlation_heatmap(df, feats, max_feat=18),
                         width="stretch")
        st.markdown("##### Корреляция признаков с целевой переменной")
        train = df[df["year"] <= TRAIN_END][feats + ["nom_log", "real_log"]].dropna()
        nom_c = train[feats].corrwith(train["nom_log"]).sort_values()
        real_c = train[feats].corrwith(train["real_log"]).sort_values()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(nom_c.tail(15).iloc[::-1], orientation="h",
                          template="plotly_white",
                          title="Топ-15 ρ с ном. log(ВРП)",
                          color_discrete_sequence=[vc.COLOR_NOM])
            fig.update_layout(showlegend=False, yaxis_title="")
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.bar(real_c.tail(15).iloc[::-1], orientation="h",
                          template="plotly_white",
                          title="Топ-15 ρ с реал. log(ВРП)",
                          color_discrete_sequence=[vc.COLOR_REAL])
            fig.update_layout(showlegend=False, yaxis_title="")
            st.plotly_chart(fig, width="stretch")

    with tab3:
        cluster_counts = (
            df.drop_duplicates("object_name")["cluster_name"]
            .value_counts()
            .reset_index()
        )
        cluster_counts.columns = ["Кластер", "Регионов"]
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.dataframe(cluster_counts, hide_index=True,
                          width="stretch")
        with c2:
            g = df.groupby(["year", "cluster_name"])[TARGET_NOM].median().reset_index()
            fig = px.line(g, x="year", y=TARGET_NOM, color="cluster_name",
                           template="plotly_white", markers=True,
                           title="Медиана ВРП/чел. по кластерам")
            fig.update_yaxes(title="ВРП/чел., руб.")
            st.plotly_chart(fig, width="stretch")


# ═══ 3. ПРОГНОЗ РЕГИОНА ═════════════════════════════════════════════════════
elif PAGE.startswith("🎯"):
    st.title("🎯 Прогноз региона")
    st.caption(
        "Здесь показана траектория одного региона на 2024–2028 годы. "
        "Реальный прогноз строится от фактического уровня 2023 года; "
        "сценарные коэффициенты включаются только со второго прогнозного года, "
        "поэтому 2024 не должен давать искусственный скачок."
    )
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sel_region = st.selectbox("Регион", REGIONS,
            index=REGIONS.index("Москва") if "Москва" in REGIONS else 0)
    with c2:
        sel_metric = st.selectbox("Показатель",
            ["Номинальный ВРП/чел.", f"Реальный ВРП/чел. (цены {BASE_YEAR})"])
    with c3:
        sel_horizon = st.slider("Горизонт прогноза (лет)", 1, 5, 5)

    target = "nominal" if "Номинальный" in sel_metric else "real"
    y_col = scenario_value_col(target)
    fc_region_h = fc[(fc["object_name"] == sel_region) & (fc["horizon"] == sel_horizon)]
    if not fc_region_h.empty:
        vals = fc_region_h.set_index("scenario")[y_col]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Базовый", fmt_money(vals.get("baseline", np.nan)))
        m2.metric("Оптимистичный", fmt_money(vals.get("optimistic", np.nan)))
        m3.metric("Пессимистичный", fmt_money(vals.get("pessimistic", np.nan)))
        gap = vals.get("optimistic", np.nan) - vals.get("pessimistic", np.nan)
        base = vals.get("baseline", np.nan)
        m4.metric("Разрыв opt–pess", fmt_money(gap), fmt_pct(gap / base * 100))

    fc_filtered = fc[fc["horizon"] <= sel_horizon]
    fig = vc.fig_scenario_forecast(df, fc_filtered, sel_region, target=target)
    st.plotly_chart(fig, width="stretch")

    st.markdown("##### Таблица прогнозных значений")
    show = fc_filtered[fc_filtered["object_name"] == sel_region][
        ["year", "horizon", "scenario", "y_pred_nominal", "y_pred_real_2015",
          "y_lo_nominal", "y_hi_nominal"]
    ].copy()
    show.columns = ["Год", "h", "Сценарий", "Ном. ВРП/чел., руб.",
                    f"Реал. ВРП/чел. ({BASE_YEAR}), руб.",
                    "Ниж. CI (баз.)", "Верх. CI (баз.)"]
    st.dataframe(show.round(0), width="stretch", hide_index=True)

# ═══ 4. СЦЕНАРНЫЙ АНАЛИЗ ════════════════════════════════════════════════════
elif PAGE.startswith("📊"):
    st.title("📊 Сценарный анализ")
    st.caption(
        "Сценарный анализ отвечает не на вопрос «какая модель точнее», "
        "а на вопрос «как изменится прогноз, если макроусловия станут лучше или хуже». "
        "Главная метрика этого раздела — разрыв между оптимистичным и пессимистичным сценариями."
    )

    gap_nom, by_h_nom = scenario_gap_frame(fc, "nominal")
    gap_real, by_h_real = scenario_gap_frame(fc, "real")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        h = st.slider("Горизонт (год прогноза)", 1, 5, 5)
    with c2:
        target = st.selectbox("Показатель", ["Номинальный", "Реальный (цены 2015)"])

    target_key = "nominal" if target == "Номинальный" else "real"
    y_col = scenario_value_col(target_key)
    fc_h = fc[fc["horizon"] == h]
    yr = int(fc_h["year"].iloc[0]) if len(fc_h) else FORECAST_START + h - 1
    gap_df = gap_nom if target_key == "nominal" else gap_real
    by_h = by_h_nom if target_key == "nominal" else by_h_real
    cur = by_h[by_h["horizon"] == h].iloc[0]
    cur_nom = by_h_nom[by_h_nom["horizon"] == h].iloc[0]
    cur_real = by_h_real[by_h_real["horizon"] == h].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Год прогноза", str(yr))
    m2.metric("Медиана baseline", fmt_money(cur["baseline_median"]))
    m3.metric("Разрыв opt–pess", fmt_money(cur["gap_abs_median"]))
    m4.metric("Разрыв, % от baseline", fmt_pct(cur["gap_pct_median"]))

    st.info(
        f"На горизонте h={h} медианный разрыв сценариев: "
        f"номинальный — **{fmt_pct(cur_nom['gap_pct_median'])}**, "
        f"реальный — **{fmt_pct(cur_real['gap_pct_median'])}**. "
        "Номинальный прогноз сглаживается инерцией уровня, поэтому его вилка умеренная; "
        "реальный прогноз дополнительно отражает разницу инфляции и тренда."
    )

    with st.expander("Как читать сценарии", expanded=True):
        st.markdown("""
| Сценарий | Смысл |
|---|---|
| **Базовый** | инерционное продолжение текущих тенденций |
| **Оптимистичный** | выше нефть, ниже ставка и инфляция, умеренный номинальный и реальный рост с 2025 года |
| **Пессимистичный** | ниже нефть, выше ставка и инфляция, отрицательная сценарная корректировка с 2025 года |

Разрыв `opt–pess` — это не доверительный интервал модели. Это управляемая
сценарная вилка: насколько оценка меняется при разных макроусловиях.
""")

    summary_sc = (
        fc_h.groupby("scenario")
        .agg(median=(y_col, "median"),
              p25=(y_col, lambda x: x.quantile(0.25)),
              p75=(y_col, lambda x: x.quantile(0.75)))
        .round(0)
    )
    summary_sc.index = summary_sc.index.map(SCENARIO_LABELS)
    summary_sc.columns = ["Медиана", "25-й перцентиль", "75-й перцентиль"]

    st.markdown("##### Сводка по выбранному году")
    st.dataframe(summary_sc, width="stretch")

    st.plotly_chart(fig_scenario_medians(fc, target_key), width="stretch")

    fig_gap = go.Figure()
    fig_gap.add_trace(go.Scatter(
        x=by_h_nom["year"], y=by_h_nom["gap_pct_median"],
        mode="lines+markers", name="Номинальный",
        line=dict(color=vc.COLOR_NOM, width=3),
    ))
    fig_gap.add_trace(go.Scatter(
        x=by_h_real["year"], y=by_h_real["gap_pct_median"],
        mode="lines+markers", name=f"Реальный, цены {BASE_YEAR}",
        line=dict(color=vc.COLOR_REAL, width=3),
    ))
    fig_gap.update_layout(
        title="Как растёт разрыв optimistic–pessimistic по горизонту",
        xaxis_title="Год прогноза",
        yaxis_title="% от baseline",
        template="plotly_white",
        height=390,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_gap, width="stretch")

    fig = go.Figure()
    for sc, sub in fc_h.groupby("scenario"):
        fig.add_trace(go.Box(y=sub[y_col], name=sc,
                              marker_color=vc.SCENARIO_COLORS.get(sc, "gray")))
    fig.update_layout(
        title=f"Распределение прогноза по сценариям (год {yr})",
        yaxis_title="руб./чел.", template="plotly_white", height=400,
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("##### Где сценарная вилка максимальна")
    top_gap = (
        gap_df[gap_df["horizon"] == h]
        .sort_values("gap_pct", ascending=False)
        .head(20)
        [["object_name", "baseline", "optimistic", "pessimistic", "gap_abs", "gap_pct"]]
        .copy()
    )
    top_gap.columns = [
        "Регион", "Базовый", "Оптимистичный", "Пессимистичный",
        "Разрыв, руб.", "Разрыв, %",
    ]
    st.dataframe(top_gap.round({"Базовый": 0, "Оптимистичный": 0,
                                "Пессимистичный": 0, "Разрыв, руб.": 0,
                                "Разрыв, %": 1}),
                 width="stretch", hide_index=True)


# ═══ 5. КАРТА РЕГИОНОВ ══════════════════════════════════════════════════════
elif PAGE.startswith("🗺️"):
    st.title("🗺️ Карта регионов РФ")
    st.caption(
        "Карта показывает не только уровень ВРП, но и сценарную вилку. "
        "В режиме разрыва видно, где прогноз наиболее чувствителен к выбору макроусловий."
    )

    map_mode = st.radio(
        "Режим карты",
        ["Факт", "Прогноз по сценарию", "Разрыв сценариев"],
        horizontal=True,
    )

    c1, c2, c3, c4 = st.columns([1.2, 1.1, 1.2, 1.3])
    with c1:
        map_target = st.selectbox(
            "Показатель",
            ["Номинальный", f"Реальный (цены {BASE_YEAR})"],
            key="map_target",
        )
        target_key = "nominal" if map_target == "Номинальный" else "real"
    with c2:
        if map_mode == "Факт":
            sel_year = st.slider("Год", int(df["year"].min()), LAST_YEAR, LAST_YEAR)
        else:
            sel_h = st.slider("Горизонт (лет)", 1, 5, 5)
            sel_year = LAST_YEAR + sel_h
    with c3:
        sel_scenario_label = st.selectbox(
            "Сценарий",
            [SCENARIO_LABELS[s] for s in SCENARIO_ORDER],
            disabled=map_mode != "Прогноз по сценарию",
            key="map_scenario",
        )
        scenario_by_label = {v: k for k, v in SCENARIO_LABELS.items()}
        sel_scenario = scenario_by_label[sel_scenario_label]
    with c4:
        sel_fo = st.selectbox("Федеральный округ",
            ["Все"] + sorted(df["federal_district"].dropna().unique()))

    if map_mode == "Факт":
        sub = df[df["year"] == sel_year][[
            "object_name", "federal_district", TARGET_NOM, GRP_REAL_PC
        ]].copy()
        if target_key == "nominal":
            value_col = TARGET_NOM
            cscale = "YlGnBu"
            label = "Номинальный ВРП/чел., руб."
        else:
            value_col = GRP_REAL_PC
            cscale = "OrRd"
            label = f"Реальный ВРП/чел., цены {BASE_YEAR}, руб."
        map_note = "Фактические значения из исторической панели."
        log_scale = True
    elif map_mode == "Прогноз по сценарию":
        value_col = scenario_value_col(target_key)
        sub = fc[(fc["scenario"] == sel_scenario) & (fc["horizon"] == sel_h)].copy()
        sub["federal_district"] = sub["object_name"].map(region_to_fo)
        cscale = {"baseline": "YlGnBu", "optimistic": "Greens",
                  "pessimistic": "Reds"}[sel_scenario]
        target_label = "номинальный" if target_key == "nominal" else f"реальный, цены {BASE_YEAR}"
        label = f"Прогноз ВРП/чел., {target_label}, {SCENARIO_LABELS[sel_scenario].lower()}"
        map_note = f"Сценарий: {SCENARIO_NOTES[sel_scenario]}."
        log_scale = True
    else:
        gap_df, _ = scenario_gap_frame(fc, target_key)
        sub = gap_df[gap_df["horizon"] == sel_h].copy()
        sub["federal_district"] = sub["object_name"].map(region_to_fo)
        value_col = "gap_pct"
        cscale = "YlOrRd"
        target_label = "номинальный" if target_key == "nominal" else f"реальный, цены {BASE_YEAR}"
        label = f"Разрыв opt–pess, % от baseline, {target_label}"
        map_note = (
            "Чем темнее регион, тем сильнее расходятся оптимистичный и "
            "пессимистичный прогнозы относительно базового сценария."
        )
        log_scale = False

    if sel_fo != "Все":
        sub = sub[sub["federal_district"] == sel_fo]

    if sub.empty:
        st.warning("Для выбранных фильтров нет данных.")
        st.stop()

    stat_col = sub.dropna(subset=[value_col])
    max_row = stat_col.loc[stat_col[value_col].idxmax()]
    min_row = stat_col.loc[stat_col[value_col].idxmin()]
    vfmt = fmt_pct if value_col == "gap_pct" else fmt_money
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Регионов на карте", f"{len(stat_col)}")
    m2.metric("Медиана", vfmt(stat_col[value_col].median()))
    m3.metric("Максимум", vfmt(max_row[value_col]), max_row["object_name"])
    m4.metric("Минимум", vfmt(min_row[value_col]), min_row["object_name"])

    st.info(map_note)

    with st.expander("Как читать эту карту", expanded=False):
        st.markdown(f"""
**Цвет** показывает выбранный показатель: `{label}`. Для уровней ВРП используется
логарифмическая шкала цвета, потому что Москва, нефтегазовые и северные регионы
сильно выше медианы и иначе карта стала бы почти одноцветной.

В режиме **«Разрыв сценариев»** карта показывает не уровень ВРП, а чувствительность:
`(оптимистичный − пессимистичный) / базовый × 100%`. Чем темнее регион, тем сильнее
итоговый прогноз зависит от макроусловий.
""")

    fmap = build_region_map(
        sub,
        value_col=value_col,
        title=f"{label} — {sel_year}",
        palette_name=cscale,
        log_scale=log_scale,
        percent=value_col == "gap_pct",
    )
    if fmap is not None:
        st_folium(
            fmap,
            height=720,
            width=None,
            use_container_width=True,
            returned_objects=[],
            key=f"region_map_{map_mode}_{target_key}_{sel_year}_{sel_fo}_{sel_scenario}",
        )
    else:
        fig = vc.fig_choropleth(
            sub.rename(columns={value_col: label}),
            value_col=label, title=f"{label} — {sel_year}",
            colorscale=cscale,
            log_scale=log_scale,
            extra_hover=["federal_district"] if "federal_district" in sub.columns else None,
        )
        st.plotly_chart(fig, width="stretch")

    st.caption(
        "GeoJSON-границы покрывают 83 региона; Крым и Севастополь отображаются "
        "точками по центроидам. Для защиты от выбросов цветовая шкала обрезана "
        "по 3–97 перцентилям, численные значения в подсказках остаются исходными."
    )

    fig_dist = px.histogram(
        stat_col,
        x=value_col,
        nbins=28,
        color="federal_district" if "federal_district" in stat_col.columns else None,
        template="plotly_white",
        title="Распределение значений по регионам",
    )
    fig_dist.update_layout(height=320, xaxis_title=label, yaxis_title="Регионов")
    st.plotly_chart(fig_dist, width="stretch")

    if len(sub):
        table_cols = ["object_name", "federal_district", value_col]
        if map_mode == "Разрыв сценариев":
            table_cols += ["baseline", "optimistic", "pessimistic", "gap_abs"]
        table = sub[[c for c in table_cols if c in sub.columns]].copy()
        rename_map = {
            "object_name": "Регион",
            "federal_district": "ФО",
            value_col: "Значение" if value_col != "gap_pct" else "Разрыв, %",
            "baseline": "Базовый",
            "optimistic": "Оптимистичный",
            "pessimistic": "Пессимистичный",
            "gap_abs": "Разрыв, руб.",
        }
        table = table.rename(columns=rename_map)
        sort_col = "Разрыв, %" if value_col == "gap_pct" else "Значение"
        top_tab, low_tab = st.tabs(["Максимальные значения", "Минимальные значения"])
        with top_tab:
            st.dataframe(
                table.sort_values(sort_col, ascending=False).head(20).round(1),
                width="stretch", hide_index=True,
            )
        with low_tab:
            st.dataframe(
                table.sort_values(sort_col, ascending=True).head(20).round(1),
                width="stretch", hide_index=True,
            )


# ═══ 6. ИНТЕРПРЕТАЦИЯ ═══════════════════════════════════════════════════════
elif PAGE.startswith("🔍"):
    st.title("🔍 Интерпретация модели")
    tab1, tab2, tab3 = st.tabs([
        "SHAP модели уровня", "По кластерам", "Сравнение моделей",
    ])

    with tab1:
        st.caption(
            "**LightGBM на уровне log(ВРП).** "
            "Дает картину, кто отвечает за уровень ВРП. "
            "Доминирующий признак — `nom_log_lag1` (инерция, ~70-80% важности). "
            "Это структурное свойство авторегрессионных рядов."
        )
        st.plotly_chart(vc.fig_shap_global(shap_g, top_n=15),
                         width="stretch")
        st.dataframe(shap_g.head(20), width="stretch", hide_index=True)

    with tab2:
        st.caption("Чем по-разному движутся разные типы регионов.")
        if not shap_c.empty:
            st.plotly_chart(vc.fig_shap_per_cluster(shap_c, shap_g, top_n=10),
                             width="stretch")

    with tab3:
        st.markdown("##### Сравнение моделей walk-forward")
        st.plotly_chart(vc.fig_model_comparison(summary, metric="RMSLE"),
                         width="stretch")
        st.plotly_chart(vc.fig_model_comparison(summary, metric="MAPE"),
                         width="stretch")
        st.dataframe(summary.round(4), width="stretch")
        if not py.empty:
            sub = py[py["target"] == "nominal"] if "target" in py.columns else py
            st.plotly_chart(vc.fig_mape_by_year(sub), width="stretch")


# ═══ 7. МЕТОДОЛОГИЯ ════════════════════════════════════════════════════════
elif PAGE.startswith("📚"):
    st.title("📚 Методология")
    st.markdown(f"""
## Цель проекта
Разработать алгоритмы машинного обучения для прогнозирования ВРП на душу
населения по 85 регионам РФ с учётом 1) различения номинального и реального
показателей, 2) сценарного анализа и 3) интерпретируемости.

## Архитектура пайплайна

### 1. Препроцессинг
- Long-панель Росстата → wide-формат.
- Спецзначения (-99999999) → NaN.
- Нормализация единиц.

### 2. Реальный ВРП в ценах {BASE_YEAR}
**Официальный ИФО ВРП Росстата (Y477110109)**, единая база:
$$\\text{{ВРП}}_{{real}}(i,t) = \\text{{ВРП}}_{{nom}}(i,{BASE_YEAR}) \\cdot \\prod_{{s={BASE_YEAR}+1}}^{{t}} \\frac{{\\text{{ИФО}}(i,s)}}{{100}}$$
Покрытие 100%.

### 3. Контроль утечек
Все лаги через `groupby("object_name").shift(n≥1)`. Аудит — `leak_audit.csv`.

### 4. Walk-forward CV
`test_year ∈ [{TEST_START};{TEST_END}]`, train ≤ test_year-1.
Включает COVID и санкции — самый строгий out-of-sample.

### 5. Модели
| Группа | Список |
|---------|--------|
| Бенчмарки | Naive, MeanGrowth |
| Эконометрика | LinearRegression, Ridge, Lasso, ElasticNet |
| Деревья | RandomForest |
| Бустинг | GradientBoosting, XGBoost, LightGBM, CatBoost |
| Нейросеть | MLP |

### 6. Сценарии 2024–2028
| Параметр | Baseline | Optimistic | Pessimistic |
|----------|----------|------------|-------------|
| Нефть Brent, $ | 75 | 95 | 60 |
| Ключ. ставка, % | 12 | 8 | 16 |
| Инфляция, % | 6.5 | 4.5 | 9.5 |
| USD/RUB | 90 | 80 | 100 |
| Доп. номинальный коэффициент с 2025 г. | 0 | +1%/год | −1%/год |
| Доп. реальный коэффициент с 2025 г. | 0 | +1%/год | −1.2%/год |

### 7. SHAP-интерпретация
SHAP считается для модели уровня log(ВРП). Доминирование `nom_log_lag1`
интерпретируется как региональная инерция, а остальные признаки показывают
добавочный вклад макро- и социально-экономических факторов.
""")
