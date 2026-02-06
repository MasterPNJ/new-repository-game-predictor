import requests

MLFOLDER_API_URL = "http://mlfolder:6767"

def predict(model: str):
    """Appelle l'API du container mlfolder pour faire une prédiction"""
    try:
        response = requests.get(
            f"{MLFOLDER_API_URL}/predict",
            params={"model": model}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Erreur lors de l'appel à mlfolder: {str(e)}"}


def get_models():
    pass