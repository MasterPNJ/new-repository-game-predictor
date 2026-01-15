"""Main extraction pipeline for multi-topic GitHub repositories."""

import time
from script.config import GAMES, DELAY, START_YEAR, START_MONTH, START_DAY, END_YEAR, END_MONTH, END_DAY, DATE_GRANULARITY
from script.database import connect_to_database, close_connection, insert_rows as db_insert_rows
from script.github_api import get_topics, get_repositories_for_topic


def main():
    """Main entry point for the extraction pipeline."""
    print("🚀 Démarrage de l'extraction multi-topics...")
    
    # Connect to database
    conn, cur = connect_to_database()
    
    # Create a wrapper for insert_rows to pass cursor and connection
    def insert_rows(rows):
        db_insert_rows(cur, conn, rows)
    
    try:
        for game in GAMES:
            print(f"\n=== Recherche pour le jeu : {game} ===")
            
            topics = get_topics(game, limit=12, skip=0)
            if topics:
                for topic in topics:
                    print(f"Topic utilisé : {topic}")
                    get_repositories_for_topic(
                        insert_rows,
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
        close_connection(cur, conn)
        print("✅ Extraction terminée.")


if __name__ == "__main__":
    main()
