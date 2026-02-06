from fastapi import FastAPI
from .predict import predict_next_week
from .db import load_weekly_series
from . import config

app = FastAPI()

@app.get("/predict")
def run_function(model: str):
    prediction = predict_next_week(
        ts_weekly=load_weekly_series(
                game_name=config.GAME_NAME,
                start_date=config.START_DATES[1],
                verbose=True,
            ),
        model_name=model,
        model_config=None,
        max_lag=config.MAX_LAG
    )
    return {"prediction": prediction}

@app.get("/models")
def run_function():
    pass