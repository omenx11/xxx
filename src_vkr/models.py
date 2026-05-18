"""models.py — реестр моделей и walk-forward кросс-валидация.

В проекте используются три группы подходов:

Базовые и классические временные бенчмарки:
  - Naive       — ŷ_t = y_{t-1};
  - CAGR        — среднегодовой темп роста региона за исторический период;
  - MeanGrowth  — средний годовой прирост региона;
  - ARIMA       — одномерная ARIMA(1, 1, 0) по каждому региону.

Регуляризованные линейные модели:
  - Ridge;
  - Lasso;
  - ElasticNet.

Ансамблевые ML-модели:
  - RandomForest;
  - GradientBoosting;
  - XGBoost;
  - LightGBM;
  - CatBoost, если установлен.

MLPRegressor намеренно исключён из реестра: на коротких региональных временных
рядах 2001–2023 он нестабилен и ухудшает качество по сравнению с простыми
инерционными и ансамблевыми моделями.

Walk-forward CV:
  для каждого test_year ∈ [TEST_START, TEST_END]:
      train  : year ≤ test_year - 1
      test   : year == test_year
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from .config import (
    RANDOM_STATE, SKLEARN_N_JOBS, TARGET_NOM, GRP_REAL_PC,
    TEST_START, TEST_END,
)
from .metrics import all_metrics

warnings.filterwarnings("ignore")


# ─── 1. Baseline-модели ─────────────────────────────────────────────────────

class NaiveBaseline:
    """ŷ_t = y_{t-1}. Сильный инерционный бенчмарк для ВРП."""
    name = "Naive"

    def fit(self, df_train, target_col):
        self.target_col = target_col
        return self

    def predict(self, df_test):
        return df_test[f"{self.target_col}_lag1_raw"].values


class CAGRBaseline:
    """ŷ_t = y_{t-1} * (1 + CAGR_region).

    CAGR считается по первому и последнему доступному значению региона в train:
        CAGR = (y_last / y_first) ** (1 / n_years) - 1.
    Если для региона не хватает истории, используется глобальный CAGR.
    """
    name = "CAGR"

    def fit(self, df_train, target_col):
        self.target_col = target_col
        self.region_growth = {}
        values_for_global = []

        for region, sub in df_train.sort_values("year").groupby("object_name"):
            s = sub[["year", target_col]].dropna()
            s = s[s[target_col] > 0]
            if len(s) >= 2:
                first = float(s[target_col].iloc[0])
                last = float(s[target_col].iloc[-1])
                n_years = int(s["year"].iloc[-1] - s["year"].iloc[0])
                if first > 0 and last > 0 and n_years > 0:
                    g = (last / first) ** (1.0 / n_years) - 1.0
                    g = float(np.clip(g, -0.50, 1.00))
                    self.region_growth[region] = g
                    values_for_global.append(g)

        self.global_growth = float(np.nanmedian(values_for_global)) if values_for_global else 0.0
        return self

    def predict(self, df_test):
        g = df_test["object_name"].map(self.region_growth).fillna(self.global_growth)
        return df_test[f"{self.target_col}_lag1_raw"].values * (1.0 + g.values)


class MeanGrowthBaseline:
    """ŷ_t = y_{t-1} * (1 + средний YoY региона на train)."""
    name = "MeanGrowth"

    def fit(self, df_train, target_col):
        self.target_col = target_col
        df_train = df_train.copy()
        df_train["_yoy"] = df_train.groupby("object_name")[target_col].pct_change()
        self.region_growth = (
            df_train.groupby("object_name")["_yoy"]
            .mean()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(df_train["_yoy"].replace([np.inf, -np.inf], np.nan).mean())
            .to_dict()
        )
        self.global_growth = float(df_train["_yoy"].replace([np.inf, -np.inf], np.nan).mean())
        if not np.isfinite(self.global_growth):
            self.global_growth = 0.0
        return self

    def predict(self, df_test):
        g = df_test["object_name"].map(self.region_growth).fillna(self.global_growth)
        return df_test[f"{self.target_col}_lag1_raw"].values * (1.0 + g.values)


class ARIMABaseline:
    """Региональная ARIMA(1, 1, 0) на log1p(target).

    Модель обучается отдельно для каждого региона на доступной истории train и
    даёт одношаговый прогноз. При недостатке наблюдений или ошибке сходимости
    выполняется fallback на последнее известное значение.
    """
    name = "ARIMA"

    def __init__(self, order: tuple[int, int, int] = (1, 1, 0), min_obs: int = 8):
        self.order = order
        self.min_obs = min_obs

    def fit(self, df_train, target_col):
        self.target_col = target_col
        self.history = {}
        for region, sub in df_train.sort_values("year").groupby("object_name"):
            s = sub[target_col].dropna().astype(float)
            s = s[s > 0]
            self.history[region] = np.log1p(s.values)
        return self

    def predict(self, df_test):
        preds = []
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except Exception:
            ARIMA = None

        for _, row in df_test.iterrows():
            region = row["object_name"]
            lag1 = float(row.get(f"{self.target_col}_lag1_raw", np.nan))
            hist = self.history.get(region, np.array([], dtype=float))

            if ARIMA is None or len(hist) < self.min_obs:
                preds.append(lag1)
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(hist, order=self.order, enforce_stationarity=False,
                                  enforce_invertibility=False)
                    fit = model.fit()
                    pred_log = float(fit.forecast(steps=1)[0])
                pred = float(np.expm1(pred_log))
                if not np.isfinite(pred) or pred <= 0:
                    pred = lag1
            except Exception:
                pred = lag1
            preds.append(pred)
        return np.asarray(preds, dtype=float)


# ─── 2. Реестр ML-моделей ───────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    factory: callable


def make_specs() -> list[ModelSpec]:
    rs = RANDOM_STATE

    def _ridge():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=rs)),
        ])

    def _lasso():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.001, max_iter=10000, random_state=rs)),
        ])

    def _enet():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.001, l1_ratio=0.5,
                                  max_iter=10000, random_state=rs)),
        ])

    def _rf():
        return RandomForestRegressor(
            n_estimators=400, max_depth=12, min_samples_leaf=2,
            n_jobs=SKLEARN_N_JOBS, random_state=rs,
        )

    def _gbdt():
        return GradientBoostingRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.85, random_state=rs,
        )

    def _xgb():
        return xgb.XGBRegressor(
            n_estimators=600, max_depth=5, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85,
            reg_lambda=1.0, random_state=rs, n_jobs=-1, verbosity=0,
            tree_method="hist",
        )

    def _lgb():
        return lgb.LGBMRegressor(
            n_estimators=600, num_leaves=31, learning_rate=0.04,
            min_data_in_leaf=20, subsample=0.85, colsample_bytree=0.85,
            random_state=rs, n_jobs=-1, verbose=-1,
        )

    specs = [
        ModelSpec("Ridge", _ridge),
        ModelSpec("Lasso", _lasso),
        ModelSpec("ElasticNet", _enet),
        ModelSpec("RandomForest", _rf),
        ModelSpec("GradientBoosting", _gbdt),
        ModelSpec("XGBoost", _xgb),
        ModelSpec("LightGBM", _lgb),
    ]
    if HAS_CATBOOST:
        def _cat():
            return CatBoostRegressor(
                iterations=600, depth=6, learning_rate=0.04,
                random_seed=rs, verbose=0, allow_writing_files=False,
            )
        specs.append(ModelSpec("CatBoost", _cat))
    return specs


# ─── 3. Walk-forward CV ─────────────────────────────────────────────────────

def prepare_xy(df: pd.DataFrame, features: list[str], target_log: str):
    sub = df.dropna(subset=[target_log]).copy()
    return sub[features], sub[target_log], sub[["object_name", "year"]]


def _make_baseline(name: str):
    if name == "Naive":
        return NaiveBaseline()
    if name == "CAGR":
        return CAGRBaseline()
    if name == "MeanGrowth":
        return MeanGrowthBaseline()
    if name == "ARIMA":
        return ARIMABaseline()
    raise ValueError(name)


def walk_forward_cv(df: pd.DataFrame,
                     features: list[str],
                     target_raw: str,
                     spec: ModelSpec | str,
                     test_years: tuple[int, ...] = None) -> pd.DataFrame:
    """Walk-forward CV для одной модели.

    ML-модели обучаются на log(target), прогноз возвращается в исходной шкале.
    Baseline-модели работают с исходной шкалой, кроме ARIMA, которая внутри
    использует log1p для устойчивости.
    """
    if test_years is None:
        test_years = tuple(range(TEST_START, TEST_END + 1))

    target_log = "nom_log" if target_raw == TARGET_NOM else "real_log"
    df = df.copy()
    df[f"{target_raw}_lag1_raw"] = df.groupby("object_name")[target_raw].shift(1)

    is_baseline = isinstance(spec, str)
    name = spec if is_baseline else spec.name

    oof = []
    for yr in test_years:
        train = df[df["year"] <= yr - 1].copy()
        test = df[df["year"] == yr].copy()
        train = train.dropna(subset=[target_log])
        test = test.dropna(subset=[target_raw, f"{target_raw}_lag1_raw"])

        if is_baseline:
            model = _make_baseline(spec)
            model.fit(train, target_raw)
            y_pred = model.predict(test)
            y_true = test[target_raw].values
        else:
            X_tr = train[features].values
            y_tr = train[target_log].values
            X_te = test[features].values
            y_te = test[target_raw].values
            imputer = SimpleImputer(strategy="median").fit(X_tr)
            X_tr_i = imputer.transform(X_tr)
            X_te_i = imputer.transform(X_te)
            mdl = spec.factory()
            mdl.fit(X_tr_i, y_tr)
            y_log_pred = mdl.predict(X_te_i)
            y_pred = np.expm1(y_log_pred)
            y_true = y_te

        for r, t, yt, yp in zip(test["object_name"], test["year"], y_true, y_pred):
            oof.append({
                "object_name": r,
                "year": int(t),
                "y_true": float(yt),
                "y_pred": float(yp),
                "model": name,
            })
    return pd.DataFrame(oof)


def run_all_models(df: pd.DataFrame,
                    features: list[str],
                    target_raw: str = TARGET_NOM,
                    test_years: tuple[int, ...] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Запускает walk-forward для всех baseline и ML-моделей."""
    target_label = "nom" if target_raw == TARGET_NOM else "real"
    print(f"\n  Walk-forward CV для target = {target_raw} ({target_label})")
    all_oof = []

    for bname in ["Naive", "CAGR", "MeanGrowth", "ARIMA"]:
        print(f"    {bname}...")
        oof_b = walk_forward_cv(df, features, target_raw, bname, test_years)
        all_oof.append(oof_b)

    for sp in make_specs():
        print(f"    {sp.name}...")
        try:
            oof_m = walk_forward_cv(df, features, target_raw, sp, test_years)
            all_oof.append(oof_m)
        except Exception as e:
            print(f"      [FAIL] {sp.name}: {e}")

    oof = pd.concat(all_oof, ignore_index=True)

    summary_rows = []
    per_year_rows = []
    for name, sub in oof.groupby("model", sort=False):
        m = all_metrics(sub["y_true"].values, sub["y_pred"].values)
        m["n"] = len(sub)
        summary_rows.append({"model": name, **m})
        for yr, sub_yr in sub.groupby("year"):
            my = all_metrics(sub_yr["y_true"].values, sub_yr["y_pred"].values)
            my["model"], my["year"], my["n"] = name, int(yr), len(sub_yr)
            per_year_rows.append(my)

    summary = pd.DataFrame(summary_rows).set_index("model").sort_values("RMSLE")
    per_year = pd.DataFrame(per_year_rows)
    return oof, summary, per_year


# ─── 4. Финальное обучение и сохранение бандла ──────────────────────────────

def train_final(df: pd.DataFrame, features: list[str], spec: ModelSpec,
                target_raw: str = TARGET_NOM,
                train_through: int = TEST_END) -> dict:
    """Обучает финальную ML-модель на полном наборе до train_through."""
    target_log = "nom_log" if target_raw == TARGET_NOM else "real_log"
    train = df[df["year"] <= train_through].dropna(subset=[target_log])
    X = train[features].values
    y = train[target_log].values
    imputer = SimpleImputer(strategy="median").fit(X)
    Xi = imputer.transform(X)
    model = spec.factory()
    model.fit(Xi, y)
    return {
        "model": model,
        "imputer": imputer,
        "features": list(features),
        "target_raw": target_raw,
        "target_log": target_log,
        "trained_through": train_through,
        "name": spec.name,
    }
