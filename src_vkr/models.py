"""models.py — реестр моделей и walk-forward кросс-валидация.

Группы моделей (для отчёта главы ВКР «Сравнение классических и современных
методов прогнозирования»):

Базовые (бенчмарки):
  - Naive          — ŷ_t = y_{t-1};
  - MeanGrowth     — ŷ_t = y_{t-1} * (1 + средний YoY региона);

Эконометрика / Регуляризованные регрессии:
  - LinearRegression;
  - Ridge;
  - Lasso;
  - ElasticNet;

Деревья и ансамбли:
  - RandomForest;
  - GradientBoosting (sklearn);
  - XGBoost;
  - LightGBM;
  - CatBoost.

Нейросеть:
  - MLP (MLPRegressor от sklearn).

Прогноз обучается на log(target) — стандартный приём для скошённых
распределений; обратное преобразование делается через expm1.

Walk-forward CV:
  для каждого test_year ∈ [TEST_START, TEST_END]:
      train  : year ≤ test_year - 1
      test   : year == test_year
  Это честный out-of-sample protocol для панельных данных.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
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
    TRAIN_END, TEST_START, TEST_END,
)
from .metrics import all_metrics

warnings.filterwarnings("ignore")


# ─── 1. Baseline modelы ─────────────────────────────────────────────────────

class NaiveBaseline:
    """ŷ_t = y_{t-1}. Идеально для инерционных рядов."""
    name = "Naive"

    def fit(self, df_train, target_col):
        self.target_col = target_col
        return self

    def predict(self, df_test):
        return df_test[f"{self.target_col}_lag1_raw"].values


class MeanGrowthBaseline:
    """ŷ_t = y_{t-1} * (1 + средний YoY региона на train)."""
    name = "MeanGrowth"

    def fit(self, df_train, target_col):
        self.target_col = target_col
        # YoY по target в долях
        df_train = df_train.copy()
        df_train["_yoy"] = (
            df_train.groupby("object_name")[target_col].pct_change()
        )
        self.region_growth = (
            df_train.groupby("object_name")["_yoy"]
            .mean()
            .fillna(df_train["_yoy"].mean())
            .to_dict()
        )
        self.global_growth = float(df_train["_yoy"].mean())
        return self

    def predict(self, df_test):
        g = df_test["object_name"].map(self.region_growth).fillna(self.global_growth)
        return df_test[f"{self.target_col}_lag1_raw"].values * (1 + g.values)


# ─── 2. Реестр ML-моделей (log-target) ──────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    factory: callable  # function() -> sklearn-compatible regressor


def make_specs() -> list[ModelSpec]:
    rs = RANDOM_STATE

    def _ridge():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=rs))
        ])

    def _lasso():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.001, max_iter=10000, random_state=rs))
        ])

    def _enet():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.001, l1_ratio=0.5,
                                  max_iter=10000, random_state=rs))
        ])

    def _linreg():
        return Pipeline([("scaler", StandardScaler()),
                         ("model", LinearRegression())])

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

    def _mlp():
        # tanh лучше для коротких рядов с year_norm выходящим за train-диапазон;
        # alpha и более мелкая сеть — для стабильности.
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="tanh", solver="adam", alpha=1e-2,
                learning_rate_init=0.005, max_iter=800,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=30, random_state=rs,
            )),
        ])

    specs = [
        ModelSpec("LinearRegression", _linreg),
        ModelSpec("Ridge", _ridge),
        ModelSpec("Lasso", _lasso),
        ModelSpec("ElasticNet", _enet),
        ModelSpec("RandomForest", _rf),
        ModelSpec("GradientBoosting", _gbdt),
        ModelSpec("XGBoost", _xgb),
        ModelSpec("LightGBM", _lgb),
        ModelSpec("MLP", _mlp),
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
    """Возвращает X, y (DataFrame, Series) с дропом NaN в y."""
    sub = df.dropna(subset=[target_log]).copy()
    return sub[features], sub[target_log], sub[["object_name", "year"]]


def walk_forward_cv(df: pd.DataFrame,
                     features: list[str],
                     target_raw: str,
                     spec: ModelSpec | str,
                     test_years: tuple[int, ...] = None) -> pd.DataFrame:
    """Walk-forward CV для одной модели.

    target_raw : "Y477110006" (nom) или "grp_real_pc_2015" (real)

    Модель обучается на log(target). Прогноз обратно через expm1.
    Возвращает DataFrame OOF-прогнозов с колонками
    [object_name, year, y_true, y_pred].

    spec может быть ModelSpec, либо строкой "Naive"/"MeanGrowth".
    """
    if test_years is None:
        test_years = tuple(range(TEST_START, TEST_END + 1))

    target_log = "nom_log" if target_raw == TARGET_NOM else "real_log"
    # Уже есть в df: target_log. Лаг для бейзлайнов:
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
            if spec == "Naive":
                model = NaiveBaseline()
            elif spec == "MeanGrowth":
                model = MeanGrowthBaseline()
            else:
                raise ValueError(spec)
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

        for r, t, yt, yp in zip(
            test["object_name"], test["year"], y_true, y_pred
        ):
            oof.append({"object_name": r, "year": int(t),
                        "y_true": float(yt), "y_pred": float(yp),
                        "model": name})
    return pd.DataFrame(oof)


def run_all_models(df: pd.DataFrame,
                    features: list[str],
                    target_raw: str = TARGET_NOM,
                    test_years: tuple[int, ...] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Запускает walk-forward для всех моделей.

    Возвращает (oof, summary, per_year):
      - oof       : long DF с прогнозами всех моделей;
      - summary   : средние метрики на test-периоде по моделям;
      - per_year  : метрики по годам для каждой модели.
    """
    target_label = "nom" if target_raw == TARGET_NOM else "real"
    print(f"\n  Walk-forward CV для target = {target_raw} ({target_label})")
    all_oof = []

    # Baselines
    for bname in ["Naive", "MeanGrowth"]:
        print(f"    {bname}...")
        oof_b = walk_forward_cv(df, features, target_raw, bname, test_years)
        all_oof.append(oof_b)

    # ML модели
    for sp in make_specs():
        print(f"    {sp.name}...")
        try:
            oof_m = walk_forward_cv(df, features, target_raw, sp, test_years)
            all_oof.append(oof_m)
        except Exception as e:
            print(f"      [FAIL] {sp.name}: {e}")

    oof = pd.concat(all_oof, ignore_index=True)

    # Метрики по моделям
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
    """Обучает финальную модель на полном наборе (включая тест) — для прогноза в будущее."""
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
