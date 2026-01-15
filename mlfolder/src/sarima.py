import numpy as np
import pandas as pd
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .metrics import compute_mae_rmse


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
