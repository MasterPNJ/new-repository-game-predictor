from dotenv import load_dotenv

load_dotenv("/app/env/.env")

# GLOBAL SETTINGS
GAME_NAME = "minecraft"

# DATA PIPELINE
# How many weeks to use for testing
N_TEST = 24
# How many weeks for development/tuning
N_DEV = 48
# Possible start dates for the training data (None means full history)
START_DATES = [None, "2020-01-01"]

# MLFLOW
MLFLOW_EXPERIMENT_NAME = "github_repos_forecasting"

# FEATURE ENGINEERING
MAX_LAG = 12

# SARIMA HYPERPARAMETERS
SARIMA_ORDER_CANDIDATES = [
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 0)
]
SARIMA_SEASONAL_CANDIDATES = [
    (0, 1, 1, 52),
    (1, 1, 1, 52)
]

# PROPHET HYPERPARAMETERS
PROPHET_PARAM_GRID = [
    {"yearly_seasonality": True, "seasonality_mode": "additive", "changepoint_prior_scale": 0.05},
    {"yearly_seasonality": True, "seasonality_mode": "multiplicative", "changepoint_prior_scale": 0.05},
    {"yearly_seasonality": True, "seasonality_mode": "additive", "changepoint_prior_scale": 0.5},
    {"yearly_seasonality": True, "seasonality_mode": "multiplicative", "changepoint_prior_scale": 0.5},
]

# ML MODELS (LightGBM & XGBoost)
# hyperparameters
N_ESTIMATORS = 500
LEARNING_RATE = 0.05
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
RANDOM_STATE = 42

# LightGBM specific
LGBM_MAX_DEPTH = -1

# XGBoost specific
XGB_MAX_DEPTH = 5

# DRIFT DETECTION
DRIFT_REF_WINDOW = 52
DRIFT_MEAN_SHIFT_THRESHOLD = 0.25
DRIFT_STD_SHIFT_THRESHOLD = 0.25

# RETRAINING LOGIC
RETRAIN_RMSE_THRESHOLD_RATIO = 1.30
