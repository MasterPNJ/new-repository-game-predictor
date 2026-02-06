from fastapi import FastAPI
from .predict import predict_next_week
from .db import load_weekly_series
from . import __main__ as main_module
from . import config
import logging
import threading

logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/predict")
def run_function(model: str):

    if model.lower() == "sarima":
        return {"error": f"Modèle inconnu : {model}"}
    
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
    return {"models": ["Prophet", "LightGBM", "XGBoost"]}

@app.get("/train")
def run_train():
    """Lance l'entraînement complet en arrière-plan"""
    def train_in_background():
        try:
            logger.info("=" * 60)
            logger.info("🚀 Démarrage de l'entraînement complet...")
            logger.info("=" * 60)
            
            
            main_module.main()
            
            logger.info("=" * 60)
            logger.info("✅ Entraînement terminé avec succès")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Erreur: {str(e)}")
    
    train_thread = threading.Thread(target=train_in_background, daemon=True)
    train_thread.start()
    
    return {"message": "Entraînement des modèles en cours. Consulter MLflow pour suivre la progression."}