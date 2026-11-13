"""Project configuration."""

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def get_database_config():

    config = {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "username": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME"),
        "table": os.getenv("DB_TABLE", "").strip()
    }

    # Validate required configuration
    required_fields = [
        "host",
        "username",
        "password",
        "database",
        "table"
    ]

    missing_fields = [
        field for field in required_fields
        if not config[field]
    ]

    if missing_fields:
        raise ValueError(
            f"Missing database configuration: {missing_fields}"
        )

    return config
