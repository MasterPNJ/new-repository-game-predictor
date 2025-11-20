import os
import time
import requests
from dotenv import load_dotenv
from datetime import date, timedelta
import calendar

load_dotenv("/app/env/.env")

# Config GitHub
GITHUB_API_BASE = "https://api.github.com"
HEADERS = {"Accept": "application/vnd.github+json"}
if os.getenv("GITHUB_TOKEN"):
    HEADERS["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"

GAMES = ["minecraft"]
DELAY = 1

START_YEAR = 2025
START_MONTH = 9
START_DAY = 1

END_YEAR = None
END_MONTH = None
END_DAY = None

# "week", "month", "year"
DATE_GRANULARITY = "week"

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

def get_repositories_for_topic(
    game_name: str,
    topic: str,
    start_year=2008,
    start_month=1,
    start_day=1,
    end_year=None,
    end_month=None,
    end_day=None,
    granularity: str = None,
):
    """Fetch repositories for a topic between two dates (inclusive).

    Dates are specified as year/month/day.
    The search period is split into ranges according to the granularity:
    - "month": month by month
    - "week": 7-day blocks
    - "day": day by day
    """

    # Use global default if not provided
    if granularity is None:
        granularity = DATE_GRANULARITY

    granularity = granularity.lower()
    if granularity not in ("month", "week", "day"):
        raise ValueError(f"Invalid granularity '{granularity}', expected 'month', 'week' or 'day'.")

    # Build start date
    start_date = date(start_year, start_month, start_day)

    # If no end date is provided, use today's date
    if end_year is None or end_month is None or end_day is None:
        today = date.today()
        if end_year is None:
            end_year = today.year
        if end_month is None:
            end_month = today.month
        if end_day is None:
            end_day = today.day

    end_date = date(end_year, end_month, end_day)

    if end_date < start_date:
        raise ValueError(f"End date {end_date} is before start date {start_date}")

    print(
        f"Collecting repos for topic '{topic}' from {start_date} to {end_date} "
        f"with granularity = {granularity}"
    )

    # Build ranges according to granularity
    date_ranges = []

    if granularity == "month":
        # Month by month (your previous behavior)
        current = date(start_date.year, start_date.month, 1)

        while current <= end_date:
            year = current.year
            month = current.month

            # First and last day of this month
            first_day_of_month = date(year, month, 1)
            last_day_of_month = date(
                year,
                month,
                calendar.monthrange(year, month)[1]
            )

            # Clip by start_date/end_date
            range_start = max(first_day_of_month, start_date)
            range_end = min(last_day_of_month, end_date)

            if range_start <= range_end:
                date_ranges.append((
                    range_start.strftime("%Y-%m-%d"),
                    range_end.strftime("%Y-%m-%d"),
                ))

            # Next month
            if month == 12:
                current = date(year + 1, 1, 1)
            else:
                current = date(year, month + 1, 1)

    elif granularity == "week":
        # 7-day blocks
        current = start_date
        while current <= end_date:
            range_start = current
            range_end = min(current + timedelta(days=6), end_date)

            date_ranges.append((
                range_start.strftime("%Y-%m-%d"),
                range_end.strftime("%Y-%m-%d"),
            ))

            # Next block
            current = range_end + timedelta(days=1)

    elif granularity == "day":
        # One range per day
        current = start_date
        while current <= end_date:
            date_ranges.append((
                current.strftime("%Y-%m-%d"),
                current.strftime("%Y-%m-%d"),
            ))
            current = current + timedelta(days=1)

    # Request GitHub for each date range
    total_repos = 0
    for start_date_str, end_date_str in date_ranges:
        page = 1
        while True:
            check_rate_limit()
            query = f"topic:{topic} created:{start_date_str}..{end_date_str}"
            url = f"{GITHUB_API_BASE}/search/repositories?q={query}&per_page=100&page={page}"
            r = requests.get(url, headers=HEADERS)

            if r.status_code == 422:
                print(f"Period {start_date_str} to {end_date_str} finished for {topic}")
                break
            if r.status_code != 200:
                print(f"Error ({r.status_code}) for {topic}, page {page}: {r.text[:200]}")
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
            batch_count = len(items)
            total_repos += batch_count
            print(
                f"{batch_count} repos added for {topic} "
                f"({start_date_str} to {end_date_str}), page {page}"
            )
            print(f"Total for {topic}: {total_repos} repos")
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
            print(f"Aucun topic trouvé pour {game}")
        time.sleep(DELAY)
        """
        topics = get_topics(game, limit=12, skip=0)
        if topics:
            for topic in topics:
                print(f"Topic utilisé : {topic}")
                get_repositories_for_topic(
                    game,
                    topic,
                    START_YEAR,
                    START_MONTH,
                    START_DAY,
                    END_YEAR,
                    END_MONTH,
                    END_DAY,
                    DATE_GRANULARITY,
                )
                time.sleep(DELAY)

        else:
            print(f"Aucun topic trouvé pour {game}")

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
