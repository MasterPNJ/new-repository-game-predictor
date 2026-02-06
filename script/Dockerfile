FROM python:3.11-slim

WORKDIR /app

# Copier les fichiers
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Variables d'environnement
ENV PYTHONUNBUFFERED=1

# Lancer le scheduler
CMD ["sh", "-c", "python scheduler.py & uvicorn src.api:app --host 0.0.0.0 --port 6769"]