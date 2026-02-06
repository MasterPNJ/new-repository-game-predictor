from fastapi import FastAPI
from .fonctions import predict, get_models

app = FastAPI()

@app.get("/predict")
def run_function(model: str):
    return {"message": f"Tu as bien accédé au container mlflow et tu as demandé une prédiction pour le modèle: {model}"}

@app.get("/models")
def run_function():
    return get_models()