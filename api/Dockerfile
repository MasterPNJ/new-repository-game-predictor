FROM python:3.11-slim

WORKDIR /app

# Dépendance système nécessaire à LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Variables d'environnement
ENV PYTHONUNBUFFERED=1

# Lancer le scheduler
CMD ["uvicorn", "src.scheduler:app", "--host", "0.0.0.0", "--port", "6767"]
