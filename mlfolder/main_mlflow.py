import os
import json
from datetime import datetime
from itertools import product

from dotenv import load_dotenv
import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn

from pathlib import Path

load_dotenv("/app/env/.env")


# 0. MLflow + baseline (monitoring/retrain decision)

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
    # Simple drift indicator: compare last window vs previous window (mean/std shift ratios).
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


# 1. DB + weekly series

def load_db_config_from_env() -> dict:
    return {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT")),
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }


def get_mysql_connection():
    cfg = load_db_config_from_env()
    return mysql.connector.connect(
        host=cfg["host"],
        port=cfg["port"],
        database=cfg["database"],
        user=cfg["user"],
        password=cfg["password"],
        connection_timeout=10,
        autocommit=True,
    )


def load_weekly_series(
    game_name: str,
    start_date: str | None = None,
    verbose: bool = True,
) -> pd.Series:
    conn = get_mysql_connection()

    query = """
    SELECT create_at
    FROM repositories
    WHERE game_name = %s
    ORDER BY create_at;
    """

    df = pd.read_sql(query, conn, params=(game_name,))
    conn.close()

    df["create_at"] = pd.to_datetime(df["create_at"])
    df.set_index("create_at", inplace=True)

    if start_date is not None:
        df = df[df.index >= pd.to_datetime(start_date)]

    ts_daily = df.resample("D").size()

    ts = ts_daily.resample(
        "W-MON",
        label="left",
        closed="left",
    ).sum()

    # Remove incomplete last week if needed
    if len(ts) > 0:
        last_week_start = ts.index[-1]
        last_week_end = last_week_start + pd.Timedelta(days=6)
        raw_max_date = df.index.max()
        if raw_max_date < last_week_end:
            print(f"⚠️ Removing incomplete last week: {last_week_start.date()}")
            ts = ts.iloc[:-1]

    if verbose:
        print("\nWeekly time series tail:")
        print(ts.tail())

    return ts


# 2. Time series splits (Train / Dev / Test)

def split_series_last_n_weeks(ts: pd.Series, n_test: int, n_dev: int) -> dict:
    if n_test <= 0 or n_dev <= 0:
        raise ValueError("n_test and n_dev must be > 0")
    if n_test + n_dev >= len(ts):
        raise ValueError("Not enough data to split into train/dev/test with these sizes.")

    test = ts.iloc[-n_test:]
    dev = ts.iloc[-(n_test + n_dev) : -n_test]
    train = ts.iloc[: -(n_test + n_dev)]

    print("\n=== Time series splits ===")
    print(f"Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} weeks)")
    print(f"Dev:   {dev.index[0].date()} → {dev.index[-1].date()} ({len(dev)} weeks)")
    print(f"Test:  {test.index[0].date()} → {test.index[-1].date()} ({len(test)} weeks)")

    return {"train": train, "dev": dev, "test": test}


# 3. Metrics helper

def compute_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse}


# 4. ML features + LightGBM/XGBoost

def make_ml_features(ts: pd.Series, max_lag: int = 12) -> pd.DataFrame:
    df = pd.DataFrame({"y": ts})

    for lag in range(1, max_lag + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    df["ma_4"] = df["y"].rolling(4).mean()
    df["ma_8"] = df["y"].rolling(8).mean()
    df["ma_12"] = df["y"].rolling(12).mean()

    df["weekofyear"] = df.index.isocalendar().week.astype(int)
    df["month"] = df.index.month
    df["year"] = df.index.year

    return df.dropna()


def train_lightgbm_model(train_df: pd.DataFrame) -> LGBMRegressor:
    X = train_df.drop(columns=["y"])
    y = train_df["y"]

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression",
        random_state=42,
    )
    model.fit(X, y)
    return model


def train_xgboost_model(train_df: pd.DataFrame) -> XGBRegressor:
    X = train_df.drop(columns=["y"])
    y = train_df["y"]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X, y)
    return model


def evaluate_ml_model(model, df_full: pd.DataFrame, test_index: pd.Index) -> tuple[dict, pd.DataFrame]:
    df_test = df_full.loc[df_full.index.intersection(test_index)]
    X_test = df_test.drop(columns=["y"])
    y_test = df_test["y"]

    preds = model.predict(X_test)
    metrics = compute_mae_rmse(y_test.values, preds)

    df_pred = pd.DataFrame({"y_true": y_test.values, "y_pred": preds}, index=y_test.index)
    return metrics, df_pred


# 5. SARIMA tuning + evaluation

def sarima_grid_search(train: pd.Series, dev: pd.Series, order_candidates=None, seasonal_candidates=None) -> dict:
    if order_candidates is None:
        order_candidates = [(1, 1, 0), (1, 1, 1), (2, 1, 0)]
    if seasonal_candidates is None:
        seasonal_candidates = [(0, 1, 1, 52), (1, 1, 1, 52)]

    best_config = None
    best_score = np.inf
    all_results = []

    for order, seasonal_order in product(order_candidates, seasonal_candidates):
        try:
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            results = model.fit(disp=False)
            forecast = results.forecast(steps=len(dev))
            metrics = compute_mae_rmse(dev.values, forecast.values)
            mae = metrics["mae"]

            config = {
                "order": order,
                "seasonal_order": seasonal_order,
                "mae_dev": mae,
                "rmse_dev": metrics["rmse"],
            }
            all_results.append(config)
            if mae < best_score:
                best_score = mae
                best_config = config

        except Exception as e:
            print(f"Failed SARIMA order={order}, seasonal={seasonal_order}: {e}")

    return {"best": best_config, "all_results": all_results}


def evaluate_sarima_on_test(train_dev: pd.Series, test: pd.Series, order: tuple, seasonal_order: tuple) -> dict:
    model = SARIMAX(
        train_dev,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)
    forecast = results.forecast(steps=len(test))
    metrics = compute_mae_rmse(test.values, forecast.values)

    df_pred = pd.DataFrame({"y_true": test.values, "y_pred": forecast.values}, index=test.index)
    return {"model_name": "SARIMA", "order": order, "seasonal_order": seasonal_order, "metrics": metrics, "pred": df_pred}


# 6. Prophet tuning + evaluation

def prophet_grid_search(train: pd.Series, dev: pd.Series, param_grid: list | None = None) -> dict:
    if param_grid is None:
        param_grid = [
            {"yearly_seasonality": True, "seasonality_mode": "additive", "changepoint_prior_scale": 0.05},
            {"yearly_seasonality": True, "seasonality_mode": "multiplicative", "changepoint_prior_scale": 0.05},
            {"yearly_seasonality": True, "seasonality_mode": "additive", "changepoint_prior_scale": 0.5},
            {"yearly_seasonality": True, "seasonality_mode": "multiplicative", "changepoint_prior_scale": 0.5},
        ]

    best_config = None
    best_score = np.inf
    all_results = []

    train_df = train.reset_index()
    train_df.columns = ["ds", "y"]

    for cfg in param_grid:
        try:
            m = Prophet(
                yearly_seasonality=cfg["yearly_seasonality"],
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode=cfg["seasonality_mode"],
                changepoint_prior_scale=cfg["changepoint_prior_scale"],
            )
            m.fit(train_df)

            future = m.make_future_dataframe(periods=len(dev), freq="W-MON")
            forecast = m.predict(future)

            forecast_dev = forecast.set_index("ds").loc[dev.index, "yhat"]
            metrics = compute_mae_rmse(dev.values, forecast_dev.values)
            mae = metrics["mae"]

            result_cfg = {"params": cfg, "mae_dev": mae, "rmse_dev": metrics["rmse"]}
            all_results.append(result_cfg)

            if mae < best_score:
                best_score = mae
                best_config = result_cfg

        except Exception as e:
            print(f"Failed Prophet cfg={cfg}: {e}")

    return {"best": best_config, "all_results": all_results}


def evaluate_prophet_on_test(train_dev: pd.Series, test: pd.Series, best_params: dict) -> dict:
    train_dev_df = train_dev.reset_index()
    train_dev_df.columns = ["ds", "y"]

    p_cfg = best_params["params"]
    m = Prophet(
        yearly_seasonality=p_cfg["yearly_seasonality"],
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=p_cfg["seasonality_mode"],
        changepoint_prior_scale=p_cfg["changepoint_prior_scale"],
    )
    m.fit(train_dev_df)

    future = m.make_future_dataframe(periods=len(test), freq="W-MON")
    forecast = m.predict(future)

    forecast_test = forecast.set_index("ds").loc[test.index, "yhat"]
    metrics = compute_mae_rmse(test.values, forecast_test.values)

    df_pred = pd.DataFrame({"y_true": test.values, "y_pred": forecast_test.values}, index=test.index)
    return {"model_name": "Prophet", "params": p_cfg, "metrics": metrics, "pred": df_pred}


# 7. Plot + artifacts

def save_test_comparison_plot(
    test: pd.Series,
    sarima_pred: pd.Series,
    prophet_pred: pd.Series,
    lgbm_pred: pd.Series | None,
    xgb_pred: pd.Series | None,
    game_name: str,
    out_path: str,
):
    plt.figure(figsize=(12, 5))
    plt.plot(test.index, test.values, label="Real (Test)", linewidth=2)
    plt.plot(test.index, sarima_pred.values, label="SARIMA pred", linestyle="--")
    plt.plot(test.index, prophet_pred.values, label="Prophet pred", linestyle="--")

    if lgbm_pred is not None:
        plt.plot(lgbm_pred.index, lgbm_pred.values, label="LightGBM pred", linestyle=":")

    if xgb_pred is not None:
        plt.plot(xgb_pred.index, xgb_pred.values, label="XGBoost pred", linestyle="-.")

    plt.title(f"Test comparison – {game_name}")
    plt.xlabel("Date (week start)")
    plt.ylabel("Number of repos created")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# 8. Main (pipeline + nested runs)

if __name__ == "__main__":
    GAME_NAME = "minecraft"
    N_TEST = 24
    N_DEV = 48
    MAX_LAG = 12
    START_DATES = [None, "2020-01-01"]

    tracking_uri = setup_mlflow("github_repos_forecasting")
    print("MLflow version:", mlflow.__version__)
    print("MLflow tracking URI:", tracking_uri)

    # One run = one pipeline execution (weekly batch)
    for START_DATE in START_DATES:
        with mlflow.start_run(run_name=f"{GAME_NAME}_pipeline_{START_DATE or 'all'}"):
            strategy_tag = START_DATE or "all"
            mlflow.set_tag("strategy", strategy_tag)
            baseline_path = f"./baseline_{strategy_tag}.json"
            mlflow.log_param("baseline_path", baseline_path)
            mlflow.log_param("game", GAME_NAME)
            mlflow.log_param("pipeline_version", "v1")
            mlflow.log_param("eval_freq", "weekly_batch")
            mlflow.log_param("n_test", N_TEST)
            mlflow.log_param("n_dev", N_DEV)
            mlflow.log_param("max_lag", MAX_LAG)
            mlflow.log_param("start_date", START_DATE or "all")

            # Load data ONCE, using START_DATE
            ts_weekly = load_weekly_series(
                game_name=GAME_NAME,
                start_date=START_DATE,
                verbose=True,
            )
            mlflow.log_metric("n_weeks_total", float(len(ts_weekly)))

            # Data monitoring (drift)
            drift_info = log_data_monitoring(ts_weekly, ref_window=52)

            # Split
            splits = split_series_last_n_weeks(ts_weekly, n_test=N_TEST, n_dev=N_DEV)
            train, dev, test = splits["train"], splits["dev"], splits["test"]
            train_dev = pd.concat([train, dev])

            # SARIMA: tuning (dev) + test
            sarima_tuning = sarima_grid_search(train, dev)
            best_sarima = sarima_tuning["best"]

            with mlflow.start_run(run_name=f"{GAME_NAME}_SARIMA", nested=True):
                mlflow.log_param("model", "SARIMA")
                mlflow.log_param("order", str(best_sarima["order"]))
                mlflow.log_param("seasonal_order", str(best_sarima["seasonal_order"]))
                mlflow.log_metric("mae_dev", float(best_sarima["mae_dev"]))
                mlflow.log_metric("rmse_dev", float(best_sarima["rmse_dev"]))

                sarima_eval = evaluate_sarima_on_test(
                    train_dev=train_dev,
                    test=test,
                    order=best_sarima["order"],
                    seasonal_order=best_sarima["seasonal_order"],
                )
                mlflow.log_metric("mae_test", sarima_eval["metrics"]["mae"])
                mlflow.log_metric("rmse_test", sarima_eval["metrics"]["rmse"])

                sarima_pred_path = f"pred_sarima_{strategy_tag}.csv"
                sarima_eval["pred"].to_csv(sarima_pred_path)
                mlflow.log_artifact(sarima_pred_path)

            # Prophet: tuning (dev) + test
            prophet_tuning = prophet_grid_search(train, dev)
            best_prophet = prophet_tuning["best"]

            with mlflow.start_run(run_name=f"{GAME_NAME}_Prophet", nested=True):
                mlflow.log_param("model", "Prophet")
                for k, v in best_prophet["params"].items():
                    mlflow.log_param(f"prophet_{k}", v)
                mlflow.log_metric("mae_dev", float(best_prophet["mae_dev"]))
                mlflow.log_metric("rmse_dev", float(best_prophet["rmse_dev"]))

                prophet_eval = evaluate_prophet_on_test(train_dev=train_dev, test=test, best_params=best_prophet)
                mlflow.log_metric("mae_test", prophet_eval["metrics"]["mae"])
                mlflow.log_metric("rmse_test", prophet_eval["metrics"]["rmse"])

                prophet_pred_path = f"pred_prophet_{strategy_tag}.csv"
                prophet_eval["pred"].to_csv(prophet_pred_path)
                mlflow.log_artifact(prophet_pred_path)

            # ML features + LightGBM/XGB
            df_ml = make_ml_features(ts_weekly, max_lag=MAX_LAG)
            train_dev_index = train.index.union(dev.index)
            train_ml_df = df_ml.loc[df_ml.index.intersection(train_dev_index)]

            # LightGBM
            lgbm_model = train_lightgbm_model(train_ml_df)
            lgbm_metrics, lgbm_pred_df = evaluate_ml_model(lgbm_model, df_full=df_ml, test_index=test.index)

            with mlflow.start_run(run_name=f"{GAME_NAME}_LightGBM", nested=True):
                mlflow.log_param("model", "LightGBM")
                mlflow.log_param("max_lag", MAX_LAG)
                # Log the model hyperparams (key ones)
                mlflow.log_param("n_estimators", 500)
                mlflow.log_param("learning_rate", 0.05)
                mlflow.log_metric("mae_test", lgbm_metrics["mae"])
                mlflow.log_metric("rmse_test", lgbm_metrics["rmse"])

                lgbm_pred_path = f"pred_lightgbm_{strategy_tag}.csv"
                lgbm_pred_df.to_csv(lgbm_pred_path)
                mlflow.log_artifact(lgbm_pred_path)

                mlflow.sklearn.log_model(lgbm_model, name="model")

            # XGBoost
            xgb_model = train_xgboost_model(train_ml_df)
            xgb_metrics, xgb_pred_df = evaluate_ml_model(xgb_model, df_full=df_ml, test_index=test.index)

            with mlflow.start_run(run_name=f"{GAME_NAME}_XGBoost", nested=True):
                mlflow.log_param("model", "XGBoost")
                mlflow.log_param("max_lag", MAX_LAG)
                mlflow.log_param("n_estimators", 500)
                mlflow.log_param("learning_rate", 0.05)
                mlflow.log_metric("mae_test", xgb_metrics["mae"])
                mlflow.log_metric("rmse_test", xgb_metrics["rmse"])

                xgb_pred_path = f"pred_xgboost_{strategy_tag}.csv"
                xgb_pred_df.to_csv(xgb_pred_path)
                mlflow.log_artifact(xgb_pred_path)

                mlflow.sklearn.log_model(xgb_model, name="model")

            # Compare & decision (promote baseline)
            results = [
                ("SARIMA", sarima_eval["metrics"]["rmse"], sarima_eval["metrics"]["mae"]),
                ("Prophet", prophet_eval["metrics"]["rmse"], prophet_eval["metrics"]["mae"]),
                ("LightGBM", lgbm_metrics["rmse"], lgbm_metrics["mae"]),
                ("XGBoost", xgb_metrics["rmse"], xgb_metrics["mae"]),
            ]
            best_name, best_rmse, best_mae = min(results, key=lambda x: x[1])

            mlflow.log_param("best_model", best_name)
            mlflow.log_metric("best_rmse", float(best_rmse))
            mlflow.log_metric("best_mae", float(best_mae))

            baseline = load_baseline(baseline_path)
            do_retrain, reason = should_retrain(best_rmse, baseline["rmse"], drift_info["drift"])
            mlflow.log_param("retrain_decision", do_retrain)
            mlflow.log_param("retrain_reason", reason)
            if baseline["rmse"] is not None:
                mlflow.log_metric("baseline_rmse", float(baseline["rmse"]))

            # Save a global comparison plot as artifact
            plot_path = f"test_comparison_{strategy_tag}.png"
            save_test_comparison_plot(
                test=test,
                sarima_pred=sarima_eval["pred"]["y_pred"],
                prophet_pred=prophet_eval["pred"]["y_pred"],
                lgbm_pred=lgbm_pred_df["y_pred"],
                xgb_pred=xgb_pred_df["y_pred"],
                game_name=GAME_NAME,
                out_path=plot_path,
            )
            mlflow.log_artifact(plot_path)

            # Promote/update baseline if decision is True
            if do_retrain:
                save_baseline(baseline_path, best_name, best_mae, best_rmse)
                mlflow.log_param("baseline_updated", True)
            else:
                mlflow.log_param("baseline_updated", False)

    print("Done.")
