import pandas as pd
import mlflow
import mlflow.sklearn
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

from .features import make_ml_features
from .models_ml import train_lightgbm_model, train_xgboost_model

def predict_next_week(ts_weekly, model_name, model_config=None, max_lag=4):
    """
    Ré-entraîne le modèle choisi sur l'ensemble des données et prédit T+1.
    """
    next_week_date = ts_weekly.index[-1] + pd.Timedelta(weeks=1)
    prediction = None

    # SARIMA
    if model_name == "SARIMA":
        if model_config is None:
            raise ValueError("model_config (order, seasonal_order) est requis pour SARIMA")
            
        model_full = SARIMAX(
            ts_weekly,
            order=model_config["order"],
            seasonal_order=model_config["seasonal_order"],
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        
        pred_res = model_full.get_forecast(steps=1)
        prediction = pred_res.predicted_mean.iloc[0]

    # Prophet
    elif model_name == "Prophet":
        params = model_config["params"] if model_config else {}
        m = Prophet(**params)
        
        df_full_prophet = ts_weekly.reset_index()
        df_full_prophet.columns = ['ds', 'y']
        
        m.fit(df_full_prophet)
        
        future = m.make_future_dataframe(periods=1, freq='W')
        forecast = m.predict(future)
        prediction = forecast.iloc[-1]['yhat']

    # ML (LightGBM / XGBoost)
    elif model_name in ["LightGBM", "XGBoost"]:
        ts_extended = ts_weekly.copy()
        ts_extended.loc[next_week_date] = np.nan 
        
        # Génération features
        df_ml_extended = make_ml_features(ts_extended, max_lag=max_lag)
        
        # Séparation Train / Prediction
        X_future = df_ml_extended.iloc[[-1]].drop(columns=['target'])
        df_train_full = df_ml_extended.iloc[:-1].dropna()
        
        if model_name == "LightGBM":
            model_full = train_lightgbm_model(df_train_full)
            prediction = model_full.predict(X_future)[0]
            
        elif model_name == "XGBoost":
            model_full = train_xgboost_model(df_train_full)
            prediction = model_full.predict(X_future)[0]
    
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    return next_week_date, prediction