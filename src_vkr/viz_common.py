"""viz_common.py — единый источник интерактивных Plotly-фигур.

Используется И в ноутбуке, И в Streamlit-приложении — гарантирует,
что графики там и там показывают **идентичные данные и стиль**.

Различия с `viz.py`:
- `viz.py` строит matplotlib-PNG для отчётов (статика);
- `viz_common.py` строит Plotly-объекты (интерактив).

В ноутбуке нужно: `from src_vkr.viz_common import *; fig_xxx(...)`
В Streamlit: `import src_vkr.viz_common as vc; st.plotly_chart(vc.fig_xxx(...))`
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import (
    DATA_RAW, DATA_PROC, RESULTS, TARGET_NOM, GRP_REAL_PC,
    TRAIN_END, TEST_START, TEST_END, BASE_YEAR, CLUSTER_NAMES, FORECAST_START,
)
from .typology import get_cluster
from .viz import normalize_region, _load_geojson

# Палитра
COLOR_NOM = "#3a78b0"
COLOR_REAL = "#c44"
COLOR_BASE = "#3a78b0"
COLOR_OPT = "#2ecc71"
COLOR_PES = "#e74c3c"
COLOR_FACT = "#222"
SCENARIO_COLORS = {"baseline": COLOR_BASE, "optimistic": COLOR_OPT,
                    "pessimistic": COLOR_PES}


# ── 1. Динамика медианного ВРП ─────────────────────────────────────────────
def fig_grp_dynamics(df: pd.DataFrame) -> go.Figure:
    g = df.groupby("year").agg(
        nom_med=(TARGET_NOM, "median"),
        nom_mean=(TARGET_NOM, "mean"),
        real_med=(GRP_REAL_PC, "median"),
        real_mean=(GRP_REAL_PC, "mean"),
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["year"], y=g["nom_med"] / 1e6,
                              mode="lines+markers",
                              name="Номинальный, медиана",
                              line=dict(color=COLOR_NOM, width=3)))
    fig.add_trace(go.Scatter(x=g["year"], y=g["real_med"] / 1e6,
                              mode="lines+markers",
                              name=f"Реальный (цены {BASE_YEAR}), медиана",
                              line=dict(color=COLOR_REAL, width=3)))
    fig.update_layout(
        title="Динамика медианного ВРП/чел. по годам",
        yaxis_title="млн руб./чел.", xaxis_title="Год",
        template="plotly_white", height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ── 2. Гистограмма по году ─────────────────────────────────────────────────
def fig_target_distribution(df: pd.DataFrame, year: int | None = None) -> go.Figure:
    if year is None:
        year = int(df["year"].max())
    sub = df[df["year"] == year]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=sub[TARGET_NOM] / 1e6, name="Номинальный",
        marker_color=COLOR_NOM, opacity=0.7,
        nbinsx=30,
    ))
    fig.add_trace(go.Histogram(
        x=sub[GRP_REAL_PC] / 1e6, name=f"Реальный (цены {BASE_YEAR})",
        marker_color=COLOR_REAL, opacity=0.7,
        nbinsx=30,
    ))
    fig.update_layout(
        title=f"Распределение ВРП/чел. по регионам, {year}",
        xaxis_title="млн руб./чел.", yaxis_title="Число регионов",
        barmode="overlay", template="plotly_white", height=400,
    )
    return fig


# ── 3. Корреляционная матрица ──────────────────────────────────────────────
def fig_correlation_heatmap(df: pd.DataFrame, features: list[str],
                              max_feat: int = 18) -> go.Figure:
    feat = features[:max_feat]
    sub = df[df["year"] <= TRAIN_END][feat + ["nom_log"]].dropna()
    corr = sub.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmin=-1, zmax=1, zmid=0,
        text=corr.round(2).values, texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="ρ"),
    ))
    fig.update_layout(
        title=f"Корреляционная матрица отобранных признаков (train ≤ {TRAIN_END})",
        height=620, template="plotly_white",
    )
    return fig


# ── 4. Сравнение моделей ───────────────────────────────────────────────────
def fig_model_comparison(summary: pd.DataFrame, metric: str = "RMSLE") -> go.Figure:
    s = summary.sort_values(metric)
    colors = [COLOR_NOM if m not in ("Naive", "MeanGrowth") else "#cc8400"
              for m in s.index]
    fig = go.Figure(go.Bar(
        x=s[metric], y=s.index, orientation="h",
        marker_color=colors,
        text=s[metric].round(3),
    ))
    fig.update_layout(
        title=f"Сравнение моделей по {metric} (walk-forward {TEST_START}–{TEST_END})",
        xaxis_title=metric, yaxis=dict(autorange="reversed"),
        height=420, template="plotly_white",
    )
    return fig


# ── 5. MAPE по годам ───────────────────────────────────────────────────────
def fig_mape_by_year(per_year: pd.DataFrame,
                      models: list[str] | None = None) -> go.Figure:
    if models is None:
        models = ["MeanGrowth", "Naive", "RandomForest", "GradientBoosting",
                  "LightGBM", "XGBoost", "Ridge", "ElasticNet"]
    sub = per_year[per_year["model"].isin(models)]
    fig = go.Figure()
    for name, g in sub.groupby("model"):
        fig.add_trace(go.Scatter(x=g["year"], y=g["MAPE"], mode="lines+markers",
                                  name=name))
    fig.update_layout(
        title="MAPE моделей по тестовым годам",
        xaxis_title="Год", yaxis_title="MAPE, %",
        template="plotly_white", height=420,
    )
    return fig


# ── 6. Факт vs прогноз ─────────────────────────────────────────────────────
def fig_actual_vs_predicted(oof: pd.DataFrame, model_name: str) -> go.Figure:
    sub = oof[oof["model"] == model_name]
    lo = min(np.log1p(sub["y_true"]).min(), np.log1p(sub["y_pred"]).min())
    hi = max(np.log1p(sub["y_true"]).max(), np.log1p(sub["y_pred"]).max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.log1p(sub["y_true"]), y=np.log1p(sub["y_pred"]),
        mode="markers", marker=dict(size=6, color=COLOR_NOM, opacity=0.55),
        text=sub["object_name"] + " " + sub["year"].astype(str),
        hovertemplate="%{text}<br>факт: %{x:.2f}<br>прогн: %{y:.2f}",
        name="точки",
    ))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                              line=dict(dash="dash", color="gray"),
                              name="y = ŷ"))
    fig.update_layout(
        title=f"Факт vs прогноз — {model_name}",
        xaxis_title="log(факт+1)", yaxis_title="log(прогноз+1)",
        template="plotly_white", height=460,
    )
    return fig


# ── 7. Сценарный прогноз для региона ───────────────────────────────────────
def fig_scenario_forecast(df: pd.DataFrame, fc: pd.DataFrame,
                            region: str, target: str = "nominal") -> go.Figure:
    target_col = TARGET_NOM if target == "nominal" else GRP_REAL_PC
    y_col = "y_pred_nominal" if target == "nominal" else "y_pred_real_2015"
    hist = df[df["object_name"] == region].sort_values("year")
    fc_reg = fc[fc["object_name"] == region]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["year"], y=hist[target_col] / 1e6,
        mode="lines+markers", name="Факт",
        line=dict(color=COLOR_FACT, width=3),
    ))
    for sc, sub in fc_reg.groupby("scenario"):
        sub = sub.sort_values("year")
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub[y_col] / 1e6,
            mode="lines+markers", name=sc,
            line=dict(color=SCENARIO_COLORS.get(sc, "gray"), dash="dash"),
        ))
        if sc == "baseline" and target == "nominal":
            fig.add_trace(go.Scatter(
                x=list(sub["year"]) + list(sub["year"][::-1]),
                y=list(sub["y_hi_nominal"] / 1e6) + list(sub["y_lo_nominal"][::-1] / 1e6),
                fill="toself", fillcolor="rgba(58,120,176,0.18)",
                line=dict(color="rgba(0,0,0,0)"),
                name="90% CI (baseline)", showlegend=True,
            ))
    title_y = "Номинальный ВРП/чел." if target == "nominal" \
              else f"Реальный ВРП/чел. (цены {BASE_YEAR})"
    fig.update_layout(
        title=f"{title_y} — {region}",
        xaxis_title="Год", yaxis_title="млн руб./чел.",
        template="plotly_white", height=460,
    )
    return fig


# ── 8. SHAP global ─────────────────────────────────────────────────────────
def fig_shap_global(shap_df: pd.DataFrame, top_n: int = 15,
                      title: str = "Глобальная важность (mean |SHAP|)") -> go.Figure:
    s = shap_df.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=s["mean_abs_shap"], y=s["feature"], orientation="h",
        marker_color=COLOR_NOM,
        text=s["mean_abs_shap"].round(4),
    ))
    fig.update_layout(
        title=title, xaxis_title="mean |SHAP|", yaxis_title="",
        template="plotly_white", height=460,
    )
    return fig


# ── 9. SHAP по кластерам ───────────────────────────────────────────────────
def fig_shap_per_cluster(shap_c: pd.DataFrame, shap_g: pd.DataFrame,
                          top_n: int = 12) -> go.Figure:
    feat_cols = [c for c in shap_c.columns
                  if c not in ("cluster_id", "cluster_name")]
    top_global = shap_g.head(top_n)["feature"].tolist()
    keep = [c for c in feat_cols if c in top_global]
    long = shap_c.melt(id_vars=["cluster_id", "cluster_name"],
                        value_vars=keep, var_name="feature",
                        value_name="mean_abs_shap")
    fig = px.bar(long, x="mean_abs_shap", y="feature", color="cluster_name",
                  orientation="h", barmode="group", template="plotly_white",
                  height=520,
                  title="SHAP по типам регионов")
    return fig


# ── 10. Choropleth ─────────────────────────────────────────────────────────
def fig_choropleth(df: pd.DataFrame, value_col: str, title: str,
                    colorscale: str = "Viridis",
                    log_scale: bool = True,
                    extra_hover: list[str] | None = None) -> go.Figure:
    """Choropleth-карта России по region boundaries.

    Берёт GeoJSON из data/raw/ru_regions.geojson.
    Возвращает Plotly Figure (для st.plotly_chart / display в notebook).
    """
    geojson = _load_geojson()
    if geojson is None:
        # пустая фигура с уведомлением
        fig = go.Figure()
        fig.add_annotation(text="GeoJSON ru_regions.geojson не найден",
                            x=0.5, y=0.5, showarrow=False)
        return fig

    df = df.copy()
    df["region_norm"] = df["object_name"].apply(normalize_region)
    df = df.dropna(subset=[value_col])

    # featureidkey
    key = "name"  # для click_that_hood
    for c in ["name", "NAME", "name_ru"]:
        if c in geojson["features"][0]["properties"]:
            key = c; break

    if log_scale and (df[value_col] > 0).all():
        df["_color"] = np.log10(df[value_col].clip(lower=1))
        cb_title = f"log₁₀({value_col})"
    else:
        df["_color"] = df[value_col]
        cb_title = value_col
    cmin = float(df["_color"].min()) if len(df) else None
    cmax = float(df["_color"].max()) if len(df) else None

    is_percent = "%" in value_col or "процент" in value_col.lower()
    fmt = lambda v: f"{v:,.1f}".replace(",", " ") if is_percent else f"{v:,.0f}".replace(",", " ")
    unit = "%" if is_percent else " руб."
    hover = [
        f"<b>{r}</b><br>{value_col}: {fmt(v)}{unit}"
        for r, v in zip(df["object_name"], df[value_col])
    ]
    if extra_hover:
        for col in extra_hover:
            if col in df.columns:
                hover = [
                    t + (f"<br>{col}: {fmt(v)}"
                         if isinstance(v, (int, float, np.floating)) and not pd.isna(v)
                         else f"<br>{col}: {v}")
                    for t, v in zip(hover, df[col].values)
                ]
    df["_hover"] = hover

    geo_names = {f["properties"].get(key) for f in geojson["features"]}
    on_map = df["region_norm"].isin(geo_names)
    df_map = df[on_map].copy()

    fig = go.Figure(go.Choropleth(
        geojson=geojson, locations=df_map["region_norm"], z=df_map["_color"],
        featureidkey=f"properties.{key}",
        colorscale=colorscale,
        zmin=cmin, zmax=cmax,
        marker_line_width=0.6, marker_line_color="white",
        text=df_map["_hover"], hoverinfo="text",
        colorbar=dict(title=cb_title, thickness=12, len=0.7),
    ))

    # Some available boundary datasets do not include Crimea and Sevastopol.
    # Show such regions by centroid markers so the app still covers all rows.
    missing = df[~on_map].copy()
    if len(missing):
        try:
            with open(DATA_RAW / "region_centroids.json", encoding="utf-8") as f:
                centroids = json.load(f)
            centroids = {normalize_region(k): v for k, v in centroids.items()}
            missing["_lat"] = missing["region_norm"].map(
                lambda r: centroids.get(r, [None, None])[0])
            missing["_lon"] = missing["region_norm"].map(
                lambda r: centroids.get(r, [None, None])[1])
            pts = missing.dropna(subset=["_lat", "_lon"])
            if len(pts):
                fig.add_trace(go.Scattergeo(
                    lat=pts["_lat"], lon=pts["_lon"], mode="markers",
                    marker=dict(
                        size=11, color=pts["_color"], colorscale=colorscale,
                        cmin=cmin, cmax=cmax, showscale=False,
                        line=dict(color="white", width=1.5),
                    ),
                    text=pts["_hover"], hoverinfo="text",
                    name="Регион без границы в GeoJSON",
                ))
        except Exception:
            pass

    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        title=title, height=650,
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
    return fig


# ── 11. Spread сценариев ───────────────────────────────────────────────────
def fig_scenarios_spread(fc: pd.DataFrame, horizon: int = 5,
                          target: str = "nominal", top_n: int = 20) -> go.Figure:
    y_col = "y_pred_nominal" if target == "nominal" else "y_pred_real_2015"
    h_fc = fc[fc["horizon"] == horizon]
    pivot = (
        h_fc.pivot_table(index="object_name", columns="scenario", values=y_col)
        .assign(spread=lambda d: (d["optimistic"] - d["pessimistic"]) /
                                   d["baseline"] * 100)
        .sort_values("spread", ascending=False)
    )
    top = pivot.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["spread"], y=top.index, orientation="h",
        marker_color="#cc8400",
    ))
    fig.update_layout(
        title=f"Разрыв сценариев opt–pess, % от baseline (h={horizon})",
        xaxis_title="%", template="plotly_white", height=520,
    )
    return fig
