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

# Créer le fichier cron avec la syntaxe correcte
# IMPORTANT: pas de "root" dans les fichiers /etc/cron.d, et il faut un saut de ligne final
RUN echo "54 10 * * * python /app/script_multi_topic.py >> /proc/1/fd/1 2>&1" > /etc/cron.d/chargement_repos_daily && \
    echo "" >> /etc/cron.d/chargement_repos_daily

# Permissions correctes pour cron
RUN chmod 0644 /etc/cron.d/chargement_repos_daily

# Appliquer la crontab
RUN crontab /etc/cron.d/chargement_repos_daily

# Créer le fichier de log
RUN touch /var/log/cron.log

# Démarrer cron et afficher les logs
CMD cron && tail -f /var/log/cron.log