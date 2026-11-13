"""Entry point for the first data-loading step."""


from sqlalchemy import text

from banking_ml.config import get_database_config
from banking_ml.database.connection import get_db_engine
from banking_ml.data.data_loader import load_data


def main():

    print("Starting data ingestion...")

    # Read configuration
    config = get_database_config()

    # Create database engine
    engine = get_db_engine()

    try:

        # Test database connection
        with engine.connect() as connection:

            connection.execute(text("SELECT 1"))

        print("MySQL connection successful!")

        # Load data from MySQL
        df = load_data(
            engine,
            config["table"]
        )

        print("\nData loaded successfully!")

        # Display basic information
        print("\nDataset Shape:")
        print(df.shape)

        print("\nFirst 5 Records:")
        print(df.head())

        print("\nColumn Information:")
        df.info()

        print("\nMissing Values:")
        print(df.isnull().sum())

        print("\nDuplicate Records:")
        print(df.duplicated().sum())

        # Check target column if present
        if "fraud_flag" in df.columns:
            print("\nTarget Distribution:")
            print(df["fraud_flag"].value_counts(dropna=False))

    except Exception as error:

        print(f"Data ingestion failed: {error}")
        raise

    finally:

        engine.dispose()


if __name__ == "__main__":
    main()

