"""Database connection and operations module."""

import mysql.connector
from script.config import DB_TYPE, DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def connect_to_database():
    """Establish connection to the database.
    
    Returns:
        tuple: (connection, cursor) for MySQL
        
    Raises:
        RuntimeError: If DB_TYPE is not 'mysql'
    """
    print(f"Connexion à la base de données ({DB_TYPE})...")
    
    if DB_TYPE == "mysql":
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        cur = conn.cursor()
        
        # Create table if not exists
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
        
        return conn, cur
    else:
        raise RuntimeError("DB_TYPE doit être 'mysql'")


def insert_rows(cur, conn, rows):
    """Insert repository rows into database.
    
    Args:
        cur: Database cursor
        conn: Database connection
        rows: List of tuples (id_repo, game_name, topic, create_at, updated_at)
    """
    sql = """
    INSERT IGNORE INTO repositories
    (id_repo, game_name, topic, create_at, updated_at)
    VALUES (%s, %s, %s, DATE(%s), DATE(%s))
    """
    cur.executemany(sql, rows)
    conn.commit()


def close_connection(cur, conn):
    """Close database connection safely.
    
    Args:
        cur: Database cursor
        conn: Database connection
    """
    try:
        cur.close()
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass
