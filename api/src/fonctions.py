import requests

TRAINER_API_URL = "http://trainer:6767"

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
    pass