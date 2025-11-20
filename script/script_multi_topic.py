import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# Config GitHub
GITHUB_API_BASE = "https://api.github.com"
HEADERS = {"Accept": "application/vnd.github+json"}
if os.getenv("GITHUB_TOKEN"):
    HEADERS["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"

GAMES = ["minecraft"]
DELAY = 1

def check_rate_limit():
    r = requests.get(f"{GITHUB_API_BASE}/rate_limit", headers=HEADERS)
    if r.status_code == 200:
        data = r.json()
        remaining = data['resources']['search']['remaining']
        reset_time = data['resources']['search']['reset']
        if remaining < 5:
            wait_time = reset_time - time.time()
            if wait_time > 0:
                print(f"Limite d'API proche, attente de {wait_time:.0f} secondes...")
                time.sleep(wait_time)
    return True

# Config BDD
DB_TYPE = os.environ.get("DB_TYPE", "postgresql").lower()
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ["DB_PORT"]
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]

# Connexion
if DB_TYPE == "mysql":
    import mysql.connector

    conn = mysql.connector.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
    )
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS repositories (
        id_repo BIGINT NOT NULL PRIMARY KEY,
        game_name VARCHAR(100) NOT NULL,
        topic VARCHAR(100) NOT NULL,
        create_at DATE NOT NULL,
        updated_at DATE NOT NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
    """)
    conn.commit()

    def insert_rows(rows):
        sql = """
        INSERT IGNORE INTO repositories
        (id_repo, game_name, topic, create_at, updated_at)
        VALUES (%s, %s, %s, DATE(%s), DATE(%s))
        """
        cur.executemany(sql, rows)
        conn.commit()
else:
    raise RuntimeError("DB_TYPE doit être 'mysql'")

# GitHub: topics & repos
"""
def get_first_topic(game_name: str):
    check_rate_limit()
    url = f"{GITHUB_API_BASE}/search/topics?q={game_name}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        print(f"Erreur ({r.status_code}) recherche topic pour {game_name}: {r.text[:200]}")
        return None
    items = r.json().get("items", [])
    if not items:
        print(f"Aucun topic trouvé pour {game_name}")
        return None
    return items[0]["name"]
"""

def get_topics(game_name: str, limit=2, skip=1):
    check_rate_limit()
    url = f"{GITHUB_API_BASE}/search/topics?q={game_name}"
    r = requests.get(url, headers=HEADERS)

    if r.status_code != 200:
        print(f"Erreur ({r.status_code}) recherche topic pour {game_name}: {r.text[:200]}")
        return []

    items = r.json().get("items", [])
    if not items:
        print(f"Aucun topic trouvé pour {game_name}")
        return []

    topics = [item["name"] for item in items][skip:skip+limit]

    print(f"Topics retenus pour {game_name} : {topics}")
    return topics

def get_repositories_for_topic(game_name: str, topic: str):
    # Création dynamique des plages mensuelles de 2008 à l'année actuelle
    current_year = time.localtime().tm_year
    current_month = time.localtime().tm_mon
    
    date_ranges = []
    for year in range(2025, current_year + 1): # de base 2008, mais je reprend pour les plus recent
        # Pour l'année en cours, on ne va que jusqu'au mois actuel
        max_month = 12 if year < current_year else current_month
        for month in range(1, max_month + 1):
            # Déterminer le dernier jour du mois
            if month in [4, 6, 9, 11]:
                last_day = 30
            elif month == 2:
                # Gestion basique des années bissextiles
                last_day = 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
            else:
                last_day = 31
                
            date_ranges.append((
                f"{year}-{month:02d}-01",
                f"{year}-{month:02d}-{last_day}"
            ))
    
    total_repos = 0
    for start_date, end_date in date_ranges:
        page = 1
        while True:
            check_rate_limit()
            query = f"topic:{topic} created:{start_date}..{end_date}"
            url = f"{GITHUB_API_BASE}/search/repositories?q={query}&per_page=100&page={page}"
            r = requests.get(url, headers=HEADERS)
            
            if r.status_code == 422:
                print(f"Période {start_date} à {end_date} terminée pour {topic}")
                break
            if r.status_code != 200:
                print(f"Erreur ({r.status_code}) pour {topic}, page {page}: {r.text[:200]}")
                break

            items = r.json().get("items", [])
            if not items:
                break

            rows = []
            for repo in items:
                rows.append((
                    repo["id"],
                    game_name,
                    topic,
                    repo["created_at"][:10],
                    repo["updated_at"][:10]
                ))

            insert_rows(rows)
            total_repos += len(items)
            print(f"{len(items)} dépôts ajoutés pour {topic} ({start_date} à {end_date}), page {page}")
            print(f"Total pour {topic} : {total_repos} dépôts")
            page += 1
            time.sleep(DELAY)

# Main
try:
    for game in GAMES:
        print(f"\n=== Recherche pour le jeu : {game} ===")
        """
        topic = get_first_topic(game)
        if topic:
            print(f"Topic trouvé : {topic}")
            get_repositories_for_topic(game, topic)
        else:
            print(f"⚠️ Aucun topic trouvé pour {game}")
        time.sleep(DELAY)
        """
        topics = get_topics(game, limit=12, skip=0) # avant skip 1 et limit 11
        if topics:
            for topic in topics:
                print(f"Topic utilisé : {topic}")
                get_repositories_for_topic(game, topic)
                time.sleep(DELAY)
        else:
            print(f"⚠️ Aucun topic trouvé pour {game}")

finally:
    try:
        cur.close()
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass

print("✅ Extraction terminée.")
