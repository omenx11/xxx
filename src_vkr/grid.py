"""grid.py — поиск гиперпараметров на временной кросс-валидации.

Используется TimeSeriesSplit ПО ГОДАМ внутри тренировочного периода
(year ≤ TRAIN_END). Тестовый период (TEST_START..TEST_END) не трогается.

Это гарантирует:
  - отсутствие утечек тестового периода;
  - реалистичную оценку качества подбора;
  - сравнение «до/после» GridSearch.
"""
from __future__ import annotations
import json
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from .config import (
    DATA_PROC, RESULTS, RANDOM_STATE, SKLEARN_N_JOBS, TRAIN_END,
)

warnings.filterwarnings("ignore")


# ── Param grids ─────────────────────────────────────────────────────────────
# Сетки сознательно скромные: для коротких рядов (17 train-лет) большие
# сетки приводят к переподбору. Главное — продемонстрировать механику CV.
GRIDS = {
    "Ridge":            {"alpha": [0.1, 1.0, 5.0, 10.0]},
    "ElasticNet":       {"alpha": [1e-3, 1e-2],
                          "l1_ratio": [0.2, 0.5, 0.8]},
    "RandomForest":     {"n_estimators": [200, 400],
                          "max_depth": [8, 12],
                          "min_samples_leaf": [1, 2]},
    "GradientBoosting": {"n_estimators": [200, 400],
                          "max_depth": [3, 4],
                          "learning_rate": [0.05, 0.08]},
    "XGBoost":          {"n_estimators": [400],
                          "max_depth": [4, 5, 6],
                          "learning_rate": [0.05, 0.08],
                          "subsample": [0.85],
                          "colsample_bytree": [0.85]},
    "LightGBM":         {"n_estimators": [400],
                          "num_leaves": [15, 31],
                          "learning_rate": [0.05, 0.08],
                          "subsample": [0.85],
                          "colsample_bytree": [0.85]},
}
# CatBoost — медленный, можно опционально (но для скорости отключаем)
if HAS_CATBOOST and False:
    GRIDS["CatBoost"] = {
        "iterations": [400],
        "depth": [4, 6],
        "learning_rate": [0.05, 0.08],
    }


def _build_model(name: str, params: dict):
    rs = RANDOM_STATE
    if name == "Ridge":
        return Pipeline([("sc", StandardScaler()), ("m", Ridge(random_state=rs, **params))])
    if name == "ElasticNet":
        return Pipeline([("sc", StandardScaler()),
                         ("m", ElasticNet(max_iter=10000, random_state=rs, **params))])
    if name == "RandomForest":
        return RandomForestRegressor(n_jobs=SKLEARN_N_JOBS, random_state=rs, **params)
    if name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=rs, **params)
    if name == "XGBoost":
        return xgb.XGBRegressor(n_jobs=-1, verbosity=0, tree_method="hist",
                                random_state=rs, **params)
    if name == "LightGBM":
        return lgb.LGBMRegressor(n_jobs=-1, verbose=-1, random_state=rs, **params)
    if name == "CatBoost":
        return CatBoostRegressor(
            verbose=0, random_seed=rs, allow_writing_files=False, **params)
    raise ValueError(name)


def timeseries_year_splits(years: np.ndarray, n_splits: int = 4):
    """TimeSeriesSplit по уникальным годам, не по строкам.

    Каждый split: train_years (растущее окно), val_years (1 год).
    """
    uniq = sorted(np.unique(years))
    if len(uniq) < n_splits + 1:
        n_splits = max(1, len(uniq) - 1)
    val_years = uniq[-n_splits:]
    for val_y in val_years:
        train_y = [y for y in uniq if y < val_y]
        if not train_y:
            continue
        yield train_y, [val_y]


def _params_iter(grid: dict):
    keys = list(grid)
    for combo in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


def grid_search_one(df: pd.DataFrame, features: list[str],
                    name: str, target_log: str = "nom_log",
                    n_splits: int = 4,
                    max_combos: int = 60) -> tuple[dict, float, list[dict]]:
    train = df[df["year"] <= TRAIN_END].dropna(subset=[target_log])
    years = train["year"].values
    splits = list(timeseries_year_splits(years, n_splits=n_splits))

    history = []
    grid = GRIDS[name]
    combos = list(_params_iter(grid))
    # Если слишком много — рандомизируем
    if len(combos) > max_combos:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(combos), size=max_combos, replace=False)
        combos = [combos[i] for i in idx]

    best_score, best_params = float("inf"), None
    for params in combos:
        rmses = []
        for tr_y, va_y in splits:
            tr = train[train["year"].isin(tr_y)]
            va = train[train["year"].isin(va_y)]
            if len(va) == 0:
                continue
            X_tr = tr[features].values
            y_tr = tr[target_log].values
            X_va = va[features].values
            y_va = va[target_log].values
            imp = SimpleImputer(strategy="median").fit(X_tr)
            X_tr_i = imp.transform(X_tr)
            X_va_i = imp.transform(X_va)
            try:
                mdl = _build_model(name, params)
                mdl.fit(X_tr_i, y_tr)
                pred = mdl.predict(X_va_i)
                rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
            except Exception as e:
                rmse = float("inf")
            rmses.append(rmse)
        mean_rmse = float(np.mean(rmses)) if rmses else float("inf")
        history.append({"params": params, "rmse_log": mean_rmse})
        if mean_rmse < best_score:
            best_score, best_params = mean_rmse, params

    return best_params, best_score, history


def grid_search_all(df: pd.DataFrame, features: list[str],
                    target_log: str = "nom_log") -> pd.DataFrame:
    print("=" * 70)
    print(" GRID SEARCH ГИПЕРПАРАМЕТРОВ (TimeSeriesSplit по годам)")
    print("=" * 70)
    rows = []
    for name in GRIDS:
        print(f"\n  → {name}")
        try:
            best, sc, hist = grid_search_one(df, features, name, target_log)
            print(f"    best params: {best}  RMSE_log={sc:.4f}")
            rows.append({"model": name, "best_params": best, "rmse_log": sc})
        except Exception as e:
            print(f"    [FAIL] {e}")
            rows.append({"model": name, "best_params": None, "rmse_log": None})
    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS / "grid_search_results.csv", index=False,
                   encoding="utf-8-sig")
    # Также json для удобства
    js = {r["model"]: r["best_params"] for r in rows if r["best_params"]}
    with open(RESULTS / "grid_best.json", "w", encoding="utf-8") as f:
        json.dump(js, f, ensure_ascii=False, indent=2)
    print(f"\n  [OK] {RESULTS / 'grid_search_results.csv'}")
    return df_out
