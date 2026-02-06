import requests

TRAINER_API_URL = "http://trainer:6768"
SCRIPT_API_URL = "http://script_chargement_donnnees:6769"

def predict(model: str):
    """Appelle l'API du container trainer pour faire une prédiction"""
    try:
        response = requests.get(
            f"{TRAINER_API_URL}/predict",
            params={"model": model}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Erreur lors de l'appel à trainer: {str(e)}"}


def get_models():
    """Appelle l'API du container trainer pour récupérer la liste des modèles disponibles"""
    try:
        response = requests.get(f"{TRAINER_API_URL}/models")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Erreur lors de l'appel à trainer: {str(e)}"}


def train():
    """Appelle l'API du container trainer pour lancer l'entraînement complet"""
    try:
        response = requests.get(f"{TRAINER_API_URL}/train")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Erreur lors de l'appel à trainer: {str(e)}"}
    

def load_data():
    """Appelle l'API du container script pour charger les données"""
    try:
        response = requests.get(f"{SCRIPT_API_URL}/load_data")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Erreur lors de l'appel à script: {str(e)}"}