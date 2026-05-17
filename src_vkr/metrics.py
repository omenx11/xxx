"""metrics.py — метрики для оценки прогнозов.

Все метрики работают в исходной шкале (рубли на душу), кроме RMSLE.
RMSLE — на log-шкале, удобен для скошённых распределений, как ВРП.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _safe(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else np.nan


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _safe(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else np.nan


def mape(y_true, y_pred) -> float:
    y_true, y_pred = _safe(y_true, y_pred)
    mask = np.abs(y_true) > 1
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    y_true, y_pred = _safe(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom > 1
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def wape(y_true, y_pred) -> float:
    y_true, y_pred = _safe(y_true, y_pred)
    s = np.abs(y_true).sum()
    if s == 0:
        return np.nan
    return float(np.abs(y_true - y_pred).sum() / s * 100)


def r2(y_true, y_pred) -> float:
    y_true, y_pred = _safe(y_true, y_pred)
    if len(y_true) < 2:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def rmsle(y_true, y_pred) -> float:
    y_true, y_pred = _safe(y_true, y_pred)
    if len(y_true) < 2:
        return np.nan
    a = np.log1p(np.clip(y_pred, 0, None))
    b = np.log1p(np.clip(y_true, 0, None))
    return float(np.sqrt(np.mean((a - b) ** 2)))


def all_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "RMSLE": rmsle(y_true, y_pred),
    }


def metrics_table(metrics_by_model: dict[str, dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(metrics_by_model).T
    return df.round(4)
