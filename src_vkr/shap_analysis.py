"""shap_analysis.py — SHAP-интерпретация ML-моделей.

Этапы:
1. Обучаем интерпретируемую модель (LightGBM) на train ≤ TRAIN_END;
2. Считаем SHAP-значения на test ≥ TEST_START;
3. Сохраняем:
   - shap_global.csv          — средние |SHAP| по признакам;
   - shap_per_cluster.csv     — средние |SHAP| по кластерам регионов;
   - shap_per_region.csv      — средние |SHAP| по регионам.
4. Возвращает объект для дальнейшей визуализации.

Используем TreeExplainer (точные значения для деревьев).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import shap

from .config import (
    DATA_PROC, RESULTS, RANDOM_STATE,
    TRAIN_END, TEST_START, TEST_END, CLUSTER_NAMES,
)


def run_shap(df: pd.DataFrame, features: list[str],
              target_log: str = "nom_log",
              save: bool = True) -> dict:
    print("=" * 70)
    print(" SHAP-АНАЛИЗ")
    print("=" * 70)
    train = df[df["year"] <= TRAIN_END].dropna(subset=[target_log])
    test = df[df["year"].between(TEST_START, TEST_END)].copy()

    X_tr = train[features].values
    y_tr = train[target_log].values
    imp = SimpleImputer(strategy="median").fit(X_tr)
    X_tr_i = imp.transform(X_tr)
    X_te_i = imp.transform(test[features].values)

    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.04, num_leaves=31,
        min_data_in_leaf=20, subsample=0.85, colsample_bytree=0.85,
        random_state=RANDOM_STATE, verbose=-1,
    )
    model.fit(X_tr_i, y_tr)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te_i)
    # shap_values: (n_samples, n_features)

    # ── Global
    mean_abs = np.abs(shap_values).mean(axis=0)
    global_imp = (
        pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # ── Per region
    per_region_rows = []
    test = test.reset_index(drop=True)
    for r_idx, row in test.iterrows():
        reg = row["object_name"]
        yr = int(row["year"])
        per_region_rows.append({
            "object_name": reg, "year": yr,
            "base_value": float(explainer.expected_value),
            **{f: float(shap_values[r_idx, i]) for i, f in enumerate(features)},
        })
    per_region_df = pd.DataFrame(per_region_rows)

    # ── Per cluster (mean abs SHAP)
    from .typology import get_cluster
    test["cluster_id"] = test["object_name"].apply(get_cluster)
    per_cluster_rows = []
    for cl, sub in test.groupby("cluster_id"):
        idx = sub.index.values
        ma = np.abs(shap_values[idx]).mean(axis=0)
        row = {"cluster_id": int(cl), "cluster_name": CLUSTER_NAMES.get(int(cl))}
        row.update({f: float(ma[i]) for i, f in enumerate(features)})
        per_cluster_rows.append(row)
    per_cluster_df = pd.DataFrame(per_cluster_rows)

    if save:
        global_imp.to_csv(RESULTS / "shap_global.csv", index=False,
                          encoding="utf-8-sig")
        per_region_df.to_csv(RESULTS / "shap_per_region.csv", index=False,
                             encoding="utf-8-sig")
        per_cluster_df.to_csv(RESULTS / "shap_per_cluster.csv", index=False,
                              encoding="utf-8-sig")
        print(f"  [OK] shap_global / per_region / per_cluster в {RESULTS}")

    return {
        "global": global_imp,
        "per_region": per_region_df,
        "per_cluster": per_cluster_df,
        "explainer": explainer,
        "shap_values": shap_values,
        "X_test": X_te_i,
        "test_meta": test[["object_name", "year", "cluster_id"]].reset_index(drop=True),
        "features": features,
        "model": model,
    }
