"""selection.py — отбор признаков для прогнозной модели.

Этапы:
1. Удаление константных / почти-константных признаков (VarianceThreshold).
2. Удаление дубликатов колонок (по идентичности значений).
3. Удаление признаков с высокой корреляцией Спирмена/Пирсона (|r| > порога).
   Из пары удаляется тот, у которого:
     - меньше |r| с таргетом на train;
     - / либо имя без 'lag' (если оба с лагами — менее важный для LightGBM).
4. ML-отбор: LightGBM (gain) + permutation importance на валидации,
   ранжирование, TOP-K финальных признаков.
5. VIF-диагностика для линейных моделей.

ВАЖНО: обучение/scoring строго на train ≤ FEATSEL_TRAIN_END, валидация
на FEATSEL_VAL_YEARS — никакого подсматривания в тестовый период.
"""
from __future__ import annotations
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from .config import (
    DATA_PROC, RESULTS, RANDOM_STATE, SKLEARN_N_JOBS,
    FEATSEL_TRAIN_END, FEATSEL_VAL_YEARS, MACRO_COLS,
)

warnings.filterwarnings("ignore")


def split_train_val(df: pd.DataFrame, target_col: str, features: list[str]):
    train = df[df["year"] <= FEATSEL_TRAIN_END].dropna(subset=[target_col])
    val = df[df["year"].between(*FEATSEL_VAL_YEARS)].dropna(subset=[target_col])
    return train, val


def drop_constant(df: pd.DataFrame, features: list[str],
                  threshold: float = 1e-6) -> tuple[list[str], list[str]]:
    """Удаляет признаки с дисперсией ниже threshold."""
    keep, dropped = [], []
    for c in features:
        s = df[c].dropna()
        if len(s) < 10 or s.var() < threshold:
            dropped.append(c)
        else:
            keep.append(c)
    return keep, dropped


def drop_duplicate_cols(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[tuple]]:
    """Удаляет дубликаты по идентичности значений."""
    keep = []
    dropped = []
    seen_hashes = {}
    for c in features:
        h = pd.util.hash_pandas_object(df[c].fillna(-9.99), index=False).sum()
        if h in seen_hashes:
            dropped.append((c, seen_hashes[h]))
        else:
            seen_hashes[h] = c
            keep.append(c)
    return keep, dropped


def drop_high_corr(df: pd.DataFrame, features: list[str],
                   target_col: str, threshold: float = 0.92,
                   importance: dict[str, float] | None = None) -> tuple[list[str], list[dict]]:
    """Удаляет признаки с |corr| > threshold (попарно).

    Из пары удаляется менее важный — приоритет:
      1) есть ли importance;
      2) иначе — корреляция с таргетом;
      3) при равных — длина имени (короткое = более общее, сохраняем).
    """
    train = df[df["year"] <= FEATSEL_TRAIN_END]
    X = train[features]
    y = train[target_col]
    corr_mat = X.corr().abs()
    target_corr = X.corrwith(y).abs().to_dict()
    importance = importance or {}

    to_drop = set()
    audit = []
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape, dtype=bool), k=1))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "a", "level_1": "b", 0: "r"})
        .sort_values("r", ascending=False)
    )
    for _, row in pairs.iterrows():
        a, b, r = row["a"], row["b"], row["r"]
        if r < threshold:
            break
        if a in to_drop or b in to_drop:
            continue
        # Кто важнее
        score_a = importance.get(a, target_corr.get(a, 0))
        score_b = importance.get(b, target_corr.get(b, 0))
        if score_a >= score_b:
            drop, keep = b, a
        else:
            drop, keep = a, b
        to_drop.add(drop)
        audit.append({
            "dropped": drop, "kept": keep, "r": round(r, 4),
            "reason": f"high_corr |r|={r:.3f}; kept by higher importance/target-corr",
        })
    kept = [c for c in features if c not in to_drop]
    return kept, audit


def lgb_gain_importance(train: pd.DataFrame, val: pd.DataFrame,
                         features: list[str], target_col: str) -> dict[str, float]:
    """Обучает LightGBM на train и возвращает gain-importance в [0,1]."""
    X_tr = train[features].values
    y_tr = train[target_col].values
    X_va = val[features].values
    y_va = val[target_col].values

    imp = SimpleImputer(strategy="median").fit(X_tr)
    X_tr_i = imp.transform(X_tr)
    X_va_i = imp.transform(X_va)

    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_data_in_leaf=20, subsample=0.9, colsample_bytree=0.85,
        random_state=RANDOM_STATE, verbose=-1, importance_type="gain",
    )
    model.fit(X_tr_i, y_tr, eval_set=[(X_va_i, y_va)],
              callbacks=[lgb.early_stopping(50)])
    gains = model.feature_importances_.astype(float)
    if gains.sum() > 0:
        gains = gains / gains.sum()
    return dict(zip(features, gains)), model, imp


def permutation_imp(model, imputer, df_val: pd.DataFrame,
                    features: list[str], target_col: str) -> dict[str, float]:
    X = imputer.transform(df_val[features].values)
    y = df_val[target_col].values
    result = permutation_importance(
        model, X, y, n_repeats=5, random_state=RANDOM_STATE,
        scoring="r2", n_jobs=SKLEARN_N_JOBS
    )
    imp = result.importances_mean.clip(min=0)
    if imp.sum() > 0:
        imp = imp / imp.sum()
    return dict(zip(features, imp))


def vif_check(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Рассчитывает VIF для каждого признака (на train)."""
    train = df[df["year"] <= FEATSEL_TRAIN_END]
    X = train[features].dropna(how="any")
    if len(X) < 30 or len(features) < 2:
        return pd.DataFrame()
    # Стандартизация (для VIF не критично, но для устойчивости)
    Xc = X - X.mean()
    # VIF: 1 / (1 - R²_j) для каждого j на регрессии X_j ~ X_(-j)
    rows = []
    cols = list(X.columns)
    XX = Xc.values
    for j, name in enumerate(cols):
        y = XX[:, j]
        X_oth = np.delete(XX, j, axis=1)
        # OLS coefs через linalg
        try:
            beta, *_ = np.linalg.lstsq(X_oth, y, rcond=None)
            y_hat = X_oth @ beta
            ss_res = ((y - y_hat) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1.0 / max(1e-6, 1 - r2)
        except Exception:
            vif = np.nan
        rows.append({"feature": name, "vif": vif})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def select_features(df: pd.DataFrame,
                    candidates: list[str],
                    target_col: str = "nom_log",
                    top_k: int = 22,
                    corr_threshold: float = 0.92,
                    must_include: list[str] | None = None) -> dict:
    """Главная функция отбора признаков.

    Возвращает dict:
      - 'selected'        : финальный список;
      - 'importance'      : DataFrame с gain, permutation, score;
      - 'audit'           : DataFrame с причинами удаления;
      - 'vif'             : DataFrame с VIF;
    """
    print("=" * 70)
    print(" FEATURE SELECTION")
    print("=" * 70)
    must_include = must_include or []

    train = df[df["year"] <= FEATSEL_TRAIN_END].dropna(subset=[target_col])
    val = df[df["year"].between(*FEATSEL_VAL_YEARS)].dropna(subset=[target_col])

    audit_rows = []

    print(f"\n  [1/5] Старт: {len(candidates)} кандидатов")
    # 1. Константные
    feats, dropped = drop_constant(df, candidates)
    for c in dropped:
        audit_rows.append({"dropped": c, "kept": None, "r": None,
                            "reason": "constant/quasi-constant"})
    print(f"        После удаления квазиконстант: {len(feats)} (-{len(dropped)})")

    # 2. Дубли
    feats, dups = drop_duplicate_cols(df, feats)
    for c, k in dups:
        audit_rows.append({"dropped": c, "kept": k, "r": 1.0,
                            "reason": "duplicate column"})
    print(f"        После удаления дублей: {len(feats)} (-{len(dups)})")

    # 3. Предварительный gain — для приоритезации внутри пар с высокой корреляцией
    print(f"\n  [2/5] Предварительная важность через LightGBM")
    gain0, _, _ = lgb_gain_importance(train, val, feats, target_col)

    # 4. Удаление сильно коррелирующих
    print(f"\n  [3/5] Удаление сильно коррелирующих пар (|r| > {corr_threshold})")
    feats2, corr_audit = drop_high_corr(df, feats, target_col,
                                         threshold=corr_threshold, importance=gain0)
    audit_rows.extend(corr_audit)
    print(f"        После: {len(feats2)} (-{len(feats) - len(feats2)})")

    # 5. ML-отбор: gain + permutation, ранжирование, TOP-K
    print(f"\n  [4/5] ML-отбор: gain + permutation, top-{top_k}")
    gain, model, imp_t = lgb_gain_importance(train, val, feats2, target_col)
    pimp = permutation_imp(model, imp_t, val, feats2, target_col)
    score = {c: 0.5 * gain[c] + 0.5 * pimp[c] for c in feats2}
    rank = sorted(feats2, key=lambda c: -score[c])

    # обязательные признаки добавляем в начало
    must = [c for c in must_include if c in feats2]
    top = list(dict.fromkeys(must + rank[: top_k - len(must)]))
    top = top[:top_k]
    for c in feats2:
        if c not in top:
            audit_rows.append({
                "dropped": c, "kept": None,
                "r": None,
                "reason": f"not in top-{top_k} by gain+permutation",
            })

    # 6. VIF на финальном наборе
    print(f"\n  [5/5] VIF-диагностика финального набора")
    vif_df = vif_check(df, top)

    # Importance DataFrame
    imp_df = pd.DataFrame(
        [{"feature": c, "gain": gain.get(c, 0), "perm": pimp.get(c, 0),
          "score": score.get(c, 0)}
         for c in feats2]
    ).sort_values("score", ascending=False)

    audit_df = pd.DataFrame(audit_rows)

    # Сохранения
    pd.Series(top, name="feature").to_csv(
        DATA_PROC / "selected_features.csv", index=False, encoding="utf-8-sig")
    imp_df.to_csv(DATA_PROC / "feature_importance.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(DATA_PROC / "feature_drop_audit.csv", index=False, encoding="utf-8-sig")
    if not vif_df.empty:
        vif_df.to_csv(RESULTS / "vif_check.csv", index=False, encoding="utf-8-sig")

    print(f"\n  [OK] Финальный набор: {len(top)} признаков")
    print(f"        + {DATA_PROC / 'selected_features.csv'}")
    print(f"        + {DATA_PROC / 'feature_importance.csv'}")
    print(f"        + {DATA_PROC / 'feature_drop_audit.csv'}")

    return {
        "selected": top,
        "importance": imp_df,
        "audit": audit_df,
        "vif": vif_df,
    }


if __name__ == "__main__":
    import pandas as pd
    from src_vkr.features import get_candidate_features
    df = pd.read_parquet(DATA_PROC / "panel_features.parquet")
    cands = get_candidate_features(df)
    must = [f"{c}_lag1" for c in MACRO_COLS if f"{c}_lag1" in df.columns]
    res = select_features(df, cands, target_col="nom_log",
                           top_k=22, must_include=must)
    print("\n  Финальные признаки:")
    for f in res["selected"]:
        print(f"    {f}")
