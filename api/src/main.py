from fastapi import FastAPI
from .fonctions import predict, get_models, train, load_data

app = FastAPI()

@app.get("/predict")
def run_function(model: str):
    return predict(model)

@app.get("/models")
def run_function():
    return get_models()

@app.get("/train")
def run_train():
    return train()

@app.get("/load_data")
def run_load_data():
    return load_data()