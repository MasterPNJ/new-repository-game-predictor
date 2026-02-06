import numpy as np
import pandas as pd
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

from .metrics import compute_mae_rmse
from . import config


def sarima_grid_search(train: pd.Series, dev: pd.Series, order_candidates=None, seasonal_candidates=None) -> dict:
    if order_candidates is None:
        order_candidates = config.SARIMA_ORDER_CANDIDATES
    if seasonal_candidates is None:
        seasonal_candidates = config.SARIMA_SEASONAL_CANDIDATES

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

            run_config = {
                "order": order,
                "seasonal_order": seasonal_order,
                "mae_dev": mae,
                "rmse_dev": metrics["rmse"],
            }
            all_results.append(run_config)
            if mae < best_score:
                best_score = mae
                best_config = run_config

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

    joblib.save(model, "models/sarima_model.pkl")

    results = model.fit(disp=False)
    forecast = results.forecast(steps=len(test))
    metrics = compute_mae_rmse(test.values, forecast.values)

    df_pred = pd.DataFrame({"y_true": test.values, "y_pred": forecast.values}, index=test.index)
    return {"model_name": "SARIMA", "order": order, "seasonal_order": seasonal_order, "metrics": metrics, "pred": df_pred}
