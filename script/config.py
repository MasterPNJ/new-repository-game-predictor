"""Configuration module for extraction pipeline."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
from pathlib import Path

env_path = Path(__file__).parent / "env" / ".env"
if env_path.exists():
    load_dotenv(str(env_path))

# GitHub Configuration
GITHUB_API_BASE = "https://api.github.com"
HEADERS = {"Accept": "application/vnd.github+json"}
if os.getenv("GITHUB_TOKEN"):
    HEADERS["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"

# Extraction Parameters
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

# Database Configuration
DB_TYPE = os.environ.get("DB_TYPE")
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

# MLflow Configuration
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
BASELINE_PATH = os.environ.get("BASELINE_PATH")
