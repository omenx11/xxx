"""scripts/run_pipeline.py — единственная команда для запуска всего ВКР-пайплайна.

Запускает по очереди:
  1. preprocess        →  panel_preprocessed.parquet
  2. feature engineer  →  panel_features.parquet
  3. feature selection →  selected_features.csv, vif_check.csv
  4. grid search       →  grid_search_results.csv (TimeSeriesSplit)
  5. walk-forward CV   →  model_summary.csv, per_year_metrics.csv,
                            oof_predictions.csv (nominal + real targets)
  6. train final       →  models/<name>.pkl
  7. SHAP-анализ       →  shap_*.csv
  8. forecast 1–5 лет, 3 сценария →  forecast_all_scenarios.csv
  9. визуализация      →  reports/figures/*.png + reports/maps/*.html
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# Перенаправляем stdout в UTF-8 на Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src_vkr.config import (
    DATA_PROC, RESULTS, MODELS, TARGET_NOM, GRP_REAL_PC, MACRO_COLS,
    TRAIN_END, TEST_START, TEST_END,
)
from src_vkr.preprocess import run_preprocess
from src_vkr.features import build_features, get_candidate_features, write_leak_audit
from src_vkr.selection import select_features
from src_vkr.models import (
    make_specs, run_all_models, train_final, ModelSpec,
)
from src_vkr.grid import grid_search_all, _build_model
from src_vkr.shap_analysis import run_shap
from src_vkr.forecast import all_scenarios_forecast, compute_log_residuals
from src_vkr.viz import (
    fig_target_distribution, fig_grp_dynamics, fig_missing_target,
    fig_clusters_profile, fig_correlation_heatmap, fig_corr_with_target,
    fig_model_comparison, fig_actual_vs_predicted, fig_mape_by_year,
    fig_shap_summary, fig_region_dynamics, fig_scenario_forecast,
    make_all_maps,
)


def banner(text: str):
    print("\n" + "=" * 78)
    print(" " + text)
    print("=" * 78)


def step1_preprocess():
    return run_preprocess()


def step2_features():
    panel = pd.read_parquet(DATA_PROC / "panel_preprocessed.parquet")
    df = build_features(panel)
    cands = get_candidate_features(df)
    audit = write_leak_audit(df, cands)
    print(f"  Кандидатных признаков: {len(cands)} (подозрений на утечки: {audit['suspect_leak'].sum()})")
    return df


def step3_selection():
    df = pd.read_parquet(DATA_PROC / "panel_features.parquet")
    cands = get_candidate_features(df)
    must = [f"{c}_lag1" for c in MACRO_COLS if f"{c}_lag1" in df.columns]
    res = select_features(df, cands, target_col="nom_log", top_k=22, must_include=must)
    return res["selected"]


def step4_grid():
    df = pd.read_parquet(DATA_PROC / "panel_features.parquet")
    feats = pd.read_csv(DATA_PROC / "selected_features.csv")["feature"].tolist()
    return grid_search_all(df, feats, target_log="nom_log")


def step5_walkforward():
    df = pd.read_parquet(DATA_PROC / "panel_features.parquet")
    feats = pd.read_csv(DATA_PROC / "selected_features.csv")["feature"].tolist()

    banner("WALK-FORWARD CV: NOMINAL GRP/CAPITA")
    oof_n, sum_n, py_n = run_all_models(df, feats, target_raw=TARGET_NOM)
    sum_n.index.name = "model"
    sum_n.to_csv(RESULTS / "model_summary_nominal.csv",
                  encoding="utf-8-sig", index_label="model")
    py_n["target"] = "nominal"
    oof_n["target"] = "nominal"

    banner("WALK-FORWARD CV: REAL GRP/CAPITA (PRICES 2015)")
    oof_r, sum_r, py_r = run_all_models(df, feats, target_raw=GRP_REAL_PC)
    sum_r.index.name = "model"
    sum_r.to_csv(RESULTS / "model_summary_real.csv",
                  encoding="utf-8-sig", index_label="model")
    py_r["target"] = "real"
    oof_r["target"] = "real"

    # Объединённые таблицы (для удобства)
    oof_all = pd.concat([oof_n, oof_r], ignore_index=True)
    py_all = pd.concat([py_n, py_r], ignore_index=True)
    oof_all.to_csv(RESULTS / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    py_all.to_csv(RESULTS / "per_year_metrics.csv", index=False, encoding="utf-8-sig")

    # Главная сводка (по nominal — основной таргет)
    sum_n.round(4).to_csv(RESULTS / "model_summary.csv",
                            encoding="utf-8-sig", index_label="model")

    print("\n  Лучшие модели по RMSLE (nominal):")
    print(sum_n.head(5).round(4).to_string())
    return oof_all, sum_n, sum_r, py_all


def step6_train_final():
    df = pd.read_parquet(DATA_PROC / "panel_features.parquet")
    feats = pd.read_csv(DATA_PROC / "selected_features.csv")["feature"].tolist()

    # Загрузка лучших гиперпараметров (если был запущен grid)
    best_path = RESULTS / "grid_best.json"
    if best_path.exists():
        with open(best_path, encoding="utf-8") as f:
            best = json.load(f)
    else:
        best = {}

    # Финальные модели на nominal
    from src_vkr.models import make_specs, ModelSpec
    default_specs = {s.name: s for s in make_specs()}
    saved = []
    for name in ["LightGBM", "XGBoost", "GradientBoosting", "RandomForest",
                  "Ridge", "ElasticNet"]:
        if name not in default_specs:
            continue
        if name in best and best[name]:
            params = best[name]
            def _factory(n=name, p=params):
                return _build_model(n, p)
            spec = ModelSpec(name, _factory)
        else:
            spec = default_specs[name]

        bundle = train_final(df, feats, spec, target_raw=TARGET_NOM,
                              train_through=TEST_END)
        path = MODELS / f"{name.lower()}.pkl"
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        saved.append(name)
        print(f"  [OK] {name} → {path}")
    return saved


def step7_shap():
    df = pd.read_parquet(DATA_PROC / "panel_features.parquet")
    feats = pd.read_csv(DATA_PROC / "selected_features.csv")["feature"].tolist()
    return run_shap(df, feats, target_log="nom_log")


def step8_forecast(forecast_model: str = "GradientBoosting"):
    df = pd.read_parquet(DATA_PROC / "panel_features.parquet")
    # Берём лучший по RMSLE среди ML-моделей для CI (из OOF)
    oof = pd.read_csv(RESULTS / "oof_predictions.csv")
    oof_n = oof[oof["target"] == "nominal"]
    summary = pd.read_csv(RESULTS / "model_summary.csv", index_col="model")
    ml_models = [m for m in summary.index
                  if m not in ("Naive", "MeanGrowth")]
    best_ml = ml_models[0] if ml_models else summary.index[0]
    print(f"  CI считаются по OOF лучшей ML: {best_ml}")
    log_resid = compute_log_residuals(oof_n[oof_n["model"] == best_ml])

    # Загрузка модели для прогноза
    bundle_path = MODELS / f"{forecast_model.lower()}.pkl"
    if not bundle_path.exists():
        # fallback: лучший доступный
        for cand in ["gradientboosting.pkl", "lightgbm.pkl", "xgboost.pkl",
                      "randomforest.pkl"]:
            if (MODELS / cand).exists():
                bundle_path = MODELS / cand
                break
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    print(f"  Прогноз делает: {bundle['name']}")

    fc = all_scenarios_forecast(df, bundle, log_resid=log_resid)
    np.save(RESULTS / "log_residuals.npy", log_resid)
    return fc


def step9_visualize():
    df = pd.read_parquet(DATA_PROC / "panel_features.parquet")
    feats = pd.read_csv(DATA_PROC / "selected_features.csv")["feature"].tolist()
    summary = pd.read_csv(RESULTS / "model_summary.csv", index_col="model")
    oof = pd.read_csv(RESULTS / "oof_predictions.csv")
    py = pd.read_csv(RESULTS / "per_year_metrics.csv")
    shap_g = pd.read_csv(RESULTS / "shap_global.csv")
    fc_all = pd.read_csv(RESULTS / "forecast_all_scenarios.csv")

    banner("СТАТИЧЕСКИЕ ГРАФИКИ")
    fig_target_distribution(df)
    fig_grp_dynamics(df)
    fig_missing_target(df)
    fig_clusters_profile(df)
    fig_correlation_heatmap(df, feats, max_feat=18)
    fig_corr_with_target(df, feats)
    fig_model_comparison(summary)
    ml = [m for m in summary.index if m not in ("Naive", "MeanGrowth", "MLP")]
    best_ml = ml[0] if ml else summary.index[0]
    fig_actual_vs_predicted(oof[oof["target"] == "nominal"], best_ml)
    fig_mape_by_year(py[py["target"] == "nominal"])
    fig_shap_summary(shap_g)
    fig_region_dynamics(df)

    # Сценарные прогнозы для нескольких регионов
    for reg in ["Москва", "г. Москва", "Республика Татарстан",
                 "Ханты-Мансийский автономный округ - Югра",
                 "Республика Дагестан", "Свердловская область"]:
        if (df["object_name"] == reg).any():
            fig_scenario_forecast(df, fc_all, reg, target="nominal")
            fig_scenario_forecast(df, fc_all, reg, target="real")

    banner("ИНТЕРАКТИВНЫЕ КАРТЫ РФ")
    last_year = int(df["year"].max())
    maps = make_all_maps(df, fc_all, last_year)
    print(f"  Карт создано: {len(maps)}")
    for p in maps:
        print(f"    + {p}")


def run_all(start: int = 1, only: int | None = None):
    t0 = time.time()
    steps = {
        1: ("Preprocess",       step1_preprocess),
        2: ("Feature eng.",     step2_features),
        3: ("Feature select",   step3_selection),
        4: ("Grid search",      step4_grid),
        5: ("Walk-forward CV",  step5_walkforward),
        6: ("Train final",      step6_train_final),
        7: ("SHAP",             step7_shap),
        8: ("Forecast",         step8_forecast),
        9: ("Visualize",        step9_visualize),
    }
    if only:
        name, fn = steps[only]
        banner(f"STEP {only}/9: {name}")
        fn()
        print(f"\n  Готово ({time.time()-t0:.0f}s)")
        return
    for n in sorted(steps):
        if n < start:
            continue
        name, fn = steps[n]
        banner(f"STEP {n}/9: {name}")
        fn()
    print(f"\n  ВЕСЬ ПАЙПЛАЙН ЗАВЕРШЁН ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="from_step", type=int, default=1, help="Начать с шага")
    p.add_argument("--step", type=int, help="Запустить только этот шаг")
    args = p.parse_args()
    run_all(start=args.from_step, only=args.step)
