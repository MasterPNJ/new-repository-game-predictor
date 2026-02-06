from fastapi import FastAPI
from . import __main__ as main_module
from . import config
import logging
import threading

logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/load_data")
def run_script_chargement():
    """Lance le chargement des données en arrière-plan"""
    def load_in_background():
        try:
            logger.info("=" * 60)
            logger.info("🚀 Démarrage du chargement des données...")
            logger.info("=" * 60)
            
            
            main_module.main()
            
            logger.info("=" * 60)
            logger.info("✅ Chargement des données terminé avec succès")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Erreur: {str(e)}")
    
    train_thread = threading.Thread(target=load_in_background, daemon=True)
    train_thread.start()
    
    return {"message": "Chargement des données a démarré."}