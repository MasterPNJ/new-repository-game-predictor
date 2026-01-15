import os
import mysql.connector
import pandas as pd


def load_db_config_from_env() -> dict:
    return {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT")),
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }


def get_mysql_connection():
    cfg = load_db_config_from_env()
    return mysql.connector.connect(
        host=cfg["host"],
        port=cfg["port"],
        database=cfg["database"],
        user=cfg["user"],
        password=cfg["password"],
        connection_timeout=10,
        autocommit=True,
    )


def load_weekly_series(
    game_name: str,
    start_date: str | None = None,
    verbose: bool = True,
) -> pd.Series:
    conn = get_mysql_connection()

    query = """
    SELECT create_at
    FROM repositories
    WHERE game_name = %s
    ORDER BY create_at;
    """

    df = pd.read_sql(query, conn, params=(game_name,))
    conn.close()

    df["create_at"] = pd.to_datetime(df["create_at"])
    df.set_index("create_at", inplace=True)

    if start_date is not None:
        df = df[df.index >= pd.to_datetime(start_date)]

    ts_daily = df.resample("D").size()

    ts = ts_daily.resample(
        "W-MON",
        label="left",
        closed="left",
    ).sum()

    # Remove incomplete last week if needed
    if len(ts) > 0:
        last_week_start = ts.index[-1]
        last_week_end = last_week_start + pd.Timedelta(days=6)
        raw_max_date = df.index.max()
        if raw_max_date < last_week_end:
            print(f"Removing incomplete last week: {last_week_start.date()}")
            ts = ts.iloc[:-1]

    if verbose:
        print("\nWeekly time series tail:")
        print(ts.tail())

    return ts
