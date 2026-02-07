from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Any, List, Dict, Union
from .fonctions import predict, get_models, train, load_data

app = FastAPI(
    title="API MLOps Gateway",
    description="API d'orchestration pour la pipeline MLOps : gestion des modèles, prédictions, entraînement et chargement des données.",
    version="1.0.0"
)

# =========================
# 🔹 Modèles de réponses
# =========================

class ErrorResponse(BaseModel):
    error: str

class PredictionResponse(BaseModel):
    model: str
    prediction: Any

class ModelsResponse(BaseModel):
    models: List[str]

class TrainResponse(BaseModel):
    status: str
    details: Union[str, Dict[str, Any]]

class LoadDataResponse(BaseModel):
    status: str
    details: Union[str, Dict[str, Any]]


# =========================
# 🔹 Endpoints
# =========================

@app.get(
    "/predict",
    summary="Faire une prédiction avec un modèle",
    description="Appelle le service **trainer** pour effectuer une prédiction à partir du modèle spécifié.",
    response_model=Union[PredictionResponse, ErrorResponse],
    tags=["Prédiction"],
    responses={
        200: {"description": "Prédiction effectuée avec succès"},
        500: {"description": "Erreur lors de l'appel au service trainer"}
    }
)
def run_predict(
    model: str = Query(..., description="Nom du modèle à utiliser pour la prédiction")
):
    result = predict(model)

    if "error" in result:
        return result

    return {
        "model": model,
        "prediction": result
    }


@app.get(
    "/models",
    summary="Lister les modèles disponibles",
    description="Récupère depuis le service **trainer** la liste des modèles actuellement disponibles.",
    response_model=Union[ModelsResponse, ErrorResponse],
    tags=["Modèles"],
)
def run_models():
    result = get_models()

    if "error" in result:
        return result

    return {"models": result}


@app.get(
    "/train",
    summary="Lancer l'entraînement des modèles",
    description="Déclenche un entraînement complet via le service **trainer** dans la pipeline MLOps.",
    response_model=Union[TrainResponse, ErrorResponse],
    tags=["Entraînement"],
)
def run_train():
    result = train()

    if "error" in result:
        return result

    return {
        "status": "training_started",
        "details": result
    }


@app.get(
    "/load_data",
    summary="Charger les données pour l'entraînement",
    description="Appelle le service **script_chargement_donnnees** pour charger les données nécessaires à l'entraînement des modèles.",
    response_model=Union[LoadDataResponse, ErrorResponse],
    tags=["Données"],
)
def run_load_data():
    result = load_data()

    if "error" in result:
        return result

    return {
        "status": "data_loaded",
        "details": result
    }
