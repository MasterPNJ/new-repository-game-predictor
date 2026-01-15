import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from .metrics import compute_mae_rmse
from . import config


def train_lightgbm_model(train_df: pd.DataFrame) -> LGBMRegressor:
    X = train_df.drop(columns=["y"])
    y = train_df["y"]

    model = LGBMRegressor(
        n_estimators=config.N_ESTIMATORS,
        learning_rate=config.LEARNING_RATE,
        max_depth=config.LGBM_MAX_DEPTH,
        subsample=config.SUBSAMPLE,
        colsample_bytree=config.COLSAMPLE_BYTREE,
        objective="regression",
        random_state=config.RANDOM_STATE,
    )
    model.fit(X, y)
    return model


def train_xgboost_model(train_df: pd.DataFrame) -> XGBRegressor:
    X = train_df.drop(columns=["y"])
    y = train_df["y"]

    model = XGBRegressor(
        n_estimators=config.N_ESTIMATORS,
        learning_rate=config.LEARNING_RATE,
        max_depth=config.XGB_MAX_DEPTH,
        subsample=config.SUBSAMPLE,
        colsample_bytree=config.COLSAMPLE_BYTREE,
        objective="reg:squarederror",
        random_state=config.RANDOM_STATE,
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
