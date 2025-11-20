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

# Tâche cron : exécuter main.py tous les jours à 7h
RUN echo "0 7 * * * python /app/script_multi_topic.py >> /var/log/cron.log 2>&1" > /etc/cron.d/chargement_repos_daily

# Permissions nécessaires
RUN chmod 0644 /etc/cron.d/chargement_repos_daily

# Démarrer cron + suivre les logs
CMD cron && tail -f /var/log/cron.log
