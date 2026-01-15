import matplotlib.pyplot as plt
import pandas as pd


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
