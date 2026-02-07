from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Union
from .fonctions import predict, get_models, train, load_data

app = FastAPI(
    title="API MLOps Gateway",
    description="API d'orchestration pour la pipeline MLOps : gestion des modèles, prédictions, entraînement et chargement des données.",
    version="1.0.0"
)



class ErrorResponse(BaseModel):
    error: str


class InnerPrediction(BaseModel):
    jeu: str
    model: str
    date: str
    prediction: float


class PredictionWrapper(BaseModel):
    prediction: InnerPrediction


class PredictionResponse(BaseModel):
    model: str
    prediction: PredictionWrapper


class ModelsResponse(BaseModel):
    models: List[str]


class MessageResponse(BaseModel):
    message: str




app = FastAPI(
    title="API MLOps Gateway",
    description="API d'orchestration pour la pipeline MLOps : gestion des modèles, prédictions, entraînement et chargement des données.",
    version="1.0.0"
)


@app.get(
    "/predict",
    summary="Faire une prédiction avec un modèle",
    description=f"Appelle le service **trainer** pour effectuer une prédiction à partir du modèle spécifié.",
    response_model=Union[PredictionResponse, ErrorResponse],
    tags=["Prédiction"]
)
def run_predict(model: str):
    return predict(model)


@app.get(
    "/models",
    summary="Lister les modèles disponibles",
    description="Récupère depuis le service **trainer** la liste des modèles actuellement disponibles.",
    response_model=Union[ModelsResponse, ErrorResponse],
    tags=["Modèles"]
)
def run_models():
    return get_models()


@app.get(
    "/train",
    summary="Lancer l'entraînement des modèles",
    description="Déclenche un entraînement complet via le service **trainer**. La progression peut être suivie dans MLflow.",
    response_model=Union[MessageResponse, ErrorResponse],
    tags=["Entraînement"]
)
def run_train():
    return train()


@app.get(
    "/load_data",
    summary="Charger les données pour l'entraînement",
    description="Appelle le service **script_chargement_donnnees** pour charger les données nécessaires à l'entraînement des modèles.",
    response_model=Union[MessageResponse, ErrorResponse],
    tags=["Données"]
)
def run_load_data():
    return load_data()
