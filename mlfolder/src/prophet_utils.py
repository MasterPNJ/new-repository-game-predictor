import numpy as np
import pandas as pd
from prophet import Prophet
import joblib

from .metrics import compute_mae_rmse
from . import config

def prophet_grid_search(train: pd.Series, dev: pd.Series, param_grid: list | None = None) -> dict:
    if param_grid is None:
        param_grid = config.PROPHET_PARAM_GRID

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

            joblib.dump(m, "models/prophet_model.pkl")

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
