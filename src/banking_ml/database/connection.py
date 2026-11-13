"""Database connection setup."""

from sqlalchemy import create_engine, URL

from banking_ml.config import get_database_config


def get_db_engine():

    config = get_database_config()

    # Construct the MySQL connection URL
    connection_url = URL.create(
        drivername="mysql+pymysql",
        username=config["username"],
        password=config["password"],
        host=config["host"],
        port=config["port"],
        database=config["database"]
    )

    # Create SQLAlchemy engine
    engine = create_engine(
        connection_url,
        pool_pre_ping=True,
        echo=False
    )

    return engine
