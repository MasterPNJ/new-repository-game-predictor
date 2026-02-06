from fastapi import FastAPI
from .fonctions import predict

app = FastAPI()

@app.get("/predict")
def run_function():
    return predict()