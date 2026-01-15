import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from .metrics import compute_mae_rmse


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
