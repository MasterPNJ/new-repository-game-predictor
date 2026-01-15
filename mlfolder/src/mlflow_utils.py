import os
import json
from datetime import datetime

import pandas as pd
import mlflow


def setup_mlflow(experiment_name: str = "github_repos_forecasting"):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri


def load_baseline(path: str) -> dict:
    if not os.path.exists(path):
        return {"rmse": None, "mae": None, "updated_at": None, "model": None}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_baseline(path: str, best_model: str, mae: float, rmse: float):
    payload = {
        "model": best_model,
        "mae": float(mae),
        "rmse": float(rmse),
        "updated_at": datetime.utcnow().isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def log_data_monitoring(ts_weekly: pd.Series, ref_window: int = 52) -> dict:
    if len(ts_weekly) < 2 * ref_window:
        mlflow.log_param("data_drift", False)
        mlflow.log_param("data_drift_reason", "not_enough_history")
        return {"drift": False, "reason": "not_enough_history"}

    recent = ts_weekly.iloc[-ref_window:]
    prev = ts_weekly.iloc[-2 * ref_window : -ref_window]

    recent_mean = float(recent.mean())
    prev_mean = float(prev.mean())
    recent_std = float(recent.std(ddof=1))
    prev_std = float(prev.std(ddof=1))

    mean_shift = abs(recent_mean - prev_mean) / (abs(prev_mean) + 1e-9)
    std_shift = abs(recent_std - prev_std) / (abs(prev_std) + 1e-9)

    mlflow.log_metric("data_recent_mean", recent_mean)
    mlflow.log_metric("data_prev_mean", prev_mean)
    mlflow.log_metric("data_recent_std", recent_std)
    mlflow.log_metric("data_prev_std", prev_std)
    mlflow.log_metric("data_mean_shift_ratio", float(mean_shift))
    mlflow.log_metric("data_std_shift_ratio", float(std_shift))

    drift = (mean_shift > 0.25) or (std_shift > 0.25)
    reasons = []
    if mean_shift > 0.25:
        reasons.append("mean_shift")
    if std_shift > 0.25:
        reasons.append("std_shift")

    reason = ",".join(reasons) if reasons else "none"
    mlflow.log_param("data_drift", drift)
    mlflow.log_param("data_drift_reason", reason)
    return {"drift": drift, "reason": reason}


def should_retrain(current_rmse: float, baseline_rmse: float | None, data_drift: bool) -> tuple[bool, str]:
    if baseline_rmse is None:
        return True, "no_baseline"
    if data_drift:
        return True, "data_drift"
    if current_rmse > 1.30 * baseline_rmse:
        return True, "perf_degradation"
    return False, "ok"
