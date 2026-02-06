from fastapi import FastAPI
from .fonctions import predict, get_models

app = FastAPI()

@app.get("/predict")
def run_function(model: str):
    return predict(model)

@app.get("/models")
def run_function():
    return get_models()