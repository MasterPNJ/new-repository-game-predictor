import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import joblib
import os

from .features import make_ml_features
from .models_ml import train_lightgbm_model, train_xgboost_model


def load_model(model_name, model_path=None):
    """Charge un modèle sauvegardé"""
    if model_path is None:
        model_path = f"models/{model_name}_model.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    
    return joblib.load(model_path)



def predict_next_week(ts_weekly, model_name, model_config=None, max_lag=4):
    """
    Ré-entraîne le modèle choisi sur l'ensemble des données et prédit T+1.
    """
    next_week_date = ts_weekly.index[-1] + pd.Timedelta(weeks=1)
    prediction = None

    model_name = model_name.lower()

    # SARIMA
    if model_name == "sarima":
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
    elif model_name == "prophet":

        m = load_model(model_name)
        
        future = m.make_future_dataframe(periods=1, freq='W')
        forecast = m.predict(future)
        prediction = forecast.iloc[-1]['yhat']

    # ML (LightGBM / XGBoost)
    elif model_name in ["lightgbm", "xgboost"]:
        ts_extended = ts_weekly.copy()
        ts_extended.loc[next_week_date] = np.nan 
    
        # Génération features
        df_ml_extended = make_ml_features(ts_extended, max_lag=max_lag)
    
        # Récupérer les features de la dernière ligne (SANS target/label)
        X_future = df_ml_extended.iloc[[-1]].drop(columns=['target'], errors='ignore')
    
        model_full = load_model(model_name)
        prediction = model_full.predict(X_future)[0]

    
    else:
        return {"error": f"Modèle inconnu : {model_name}"}

    return next_week_date, prediction