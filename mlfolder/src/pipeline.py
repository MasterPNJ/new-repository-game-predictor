import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from . import config  # load_dotenv("/app/env/.env")

from .mlflow_utils import (
    setup_mlflow,
    load_baseline,
    save_baseline,
    log_data_monitoring,
    should_retrain,
)
from .db import load_weekly_series
from .splits import split_series_last_n_weeks
from .features import make_ml_features
from .sarima import sarima_grid_search, evaluate_sarima_on_test
from .prophet_utils import prophet_grid_search, evaluate_prophet_on_test
from .models_ml import train_lightgbm_model, train_xgboost_model, evaluate_ml_model
from .plotting import save_test_comparison_plot
from .predict import predict_next_week


def main():
    GAME_NAME = config.GAME_NAME
    N_TEST = config.N_TEST
    N_DEV = config.N_DEV
    MAX_LAG = config.MAX_LAG
    START_DATES = config.START_DATES

    tracking_uri = setup_mlflow(config.MLFLOW_EXPERIMENT_NAME)
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
            drift_info = log_data_monitoring(ts_weekly, ref_window=config.DRIFT_REF_WINDOW)

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
            joblib.save(lgbm_model, "models/lightgbm_model.pkl")
            lgbm_metrics, lgbm_pred_df = evaluate_ml_model(lgbm_model, df_full=df_ml, test_index=test.index)

            with mlflow.start_run(run_name=f"{GAME_NAME}_LightGBM", nested=True):
                mlflow.log_param("model", "LightGBM")
                mlflow.log_param("max_lag", MAX_LAG)
                # Log the model hyperparams
                mlflow.log_param("n_estimators", config.N_ESTIMATORS)
                mlflow.log_param("learning_rate", config.LEARNING_RATE)
                mlflow.log_metric("mae_test", lgbm_metrics["mae"])
                mlflow.log_metric("rmse_test", lgbm_metrics["rmse"])

                lgbm_pred_path = f"pred_lightgbm_{strategy_tag}.csv"
                lgbm_pred_df.to_csv(lgbm_pred_path)
                mlflow.log_artifact(lgbm_pred_path)

                mlflow.sklearn.log_model(lgbm_model, name="model")

            # XGBoost
            xgb_model = train_xgboost_model(train_ml_df)
            joblib.save(xgb_model, "models/xgboost_model.pkl")
            xgb_metrics, xgb_pred_df = evaluate_ml_model(xgb_model, df_full=df_ml, test_index=test.index)

            with mlflow.start_run(run_name=f"{GAME_NAME}_XGBoost", nested=True):
                mlflow.log_param("model", "XGBoost")
                mlflow.log_param("max_lag", MAX_LAG)
                mlflow.log_param("n_estimators", config.N_ESTIMATORS)
                mlflow.log_param("learning_rate", config.LEARNING_RATE)
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

            model_configs_map = {
                "SARIMA": best_sarima,
                "Prophet": best_prophet,
                "LightGBM": None,
                "XGBoost": None
            }
            
            # On appelle la fonction en lui passant la config du meilleur
            target_date, pred_value = predict_next_week(
                ts_weekly=ts_weekly,
                model_name=best_name,
                model_config=model_configs_map.get(best_name),
                max_lag=MAX_LAG
            )

            print(f"Prédiction pour la semaine du {target_date.date()} : {pred_value:.2f}")
            mlflow.log_metric("forecast_t_plus_1", pred_value)
            
            forecast_df = pd.DataFrame({
                "date_prediction": [pd.Timestamp.now()],
                "target_week": [target_date],
                "model_used": [best_name],
                "predicted_value": [pred_value]
            })
            
            forecast_path = f"forecast_next_week_{strategy_tag}.csv"
            forecast_df.to_csv(forecast_path, index=False)
            mlflow.log_artifact(forecast_path)

    print("Done.")
