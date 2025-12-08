FROM python:3.11-slim

# Installer cron + dépendances
RUN apt-get update && apt-get install -y cron && apt-get clean

# Définir répertoire de travail
WORKDIR /app

# Copier les fichiers
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Charger les variables d'environnement depuis /app/.env
ENV PYTHONUNBUFFERED=1

# Tâche cron : exécuter le script tous les jours à 10:05
RUN echo "5 10 * * * root /usr/local/bin/python /app/script_multi_topic.py >> /proc/1/fd/1 2>&1" > /etc/cron.d/chargement_repos_daily
RUN chmod 0644 /etc/cron.d/chargement_repos_daily

# Eviter d'utiliser 'crontab' sur le fichier dans /etc/cron.d (cron lit ce dossier automatiquement)

# Démarrer cron en avant-plan pour que le conteneur reste actif
CMD ["cron", "-f"]
