from apscheduler.schedulers.blocking import BlockingScheduler
import logging
import subprocess
import sys

# Configuration des logs pour qu'ils s'affichent dans stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_extraction():
    """Exécute le script d'extraction"""
    logger.info("=" * 60)
    logger.info("🚀 DÉMARRAGE DE L'EXTRACTION PLANIFIÉE")
    logger.info("=" * 60)
    
    try:
        # Exécuter le script SANS capture_output pour voir les logs en direct
        result = subprocess.run(
            [sys.executable, '/app/main_mlflow.py'],
            check=True
        )
        
        logger.info("=" * 60)
        logger.info("✅ EXTRACTION TERMINÉE AVEC SUCCÈS")
        logger.info("=" * 60)
        
    except subprocess.CalledProcessError as e:
        logger.error("=" * 60)
        logger.error("❌ ERREUR LORS DE L'EXTRACTION")
        logger.error(f"Code de retour: {e.returncode}")
        logger.error("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Erreur inattendue: {e}")

if __name__ == '__main__':
    scheduler = BlockingScheduler()
    
    # Planifier l'exécution quotidienne à 7h00
    job = scheduler.add_job(
        run_extraction,
        'cron',
        hour=15, # heure UTC donc mettre 1h de moins pour heure française
        minute=50,
        id='extraction_github_daily'
    )

    """
    job = scheduler.add_job(
        run_extraction,
        'cron',
        day_of_week='mon',
        hour=6,
        minute=0,
        id='pipeline_mlflow_weekly'
    )
    """
    
    logger.info("=" * 60)
    logger.info("🕐 SCHEDULER DÉMARRÉ")
    logger.info("📅 Exécution planifiée : Tous les lundis à 7h")
    logger.info("=" * 60)

    # OPTIONNEL : Décommenter pour exécuter immédiatement au démarrage
    # logger.info("▶️  Exécution immédiate au démarrage...")
    # run_extraction()
    
    try:
        scheduler.start()
        # Une fois démarré, on peut afficher la prochaine exécution
        logger.info(f"⏰ Prochaine exécution : {job.next_run_time}")
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Arrêt du scheduler")