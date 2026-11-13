
from pathlib import Path

from sqlalchemy import text

from banking_ml.config import get_database_config
from banking_ml.database.connection import get_db_engine
from banking_ml.data.data_loader import load_data

from banking_ml.data.data_cleaner import clean_data
from banking_ml.data.data_validation import validate_data


def main():

    print("\nStarting Step 3: Data Cleaning and Validation")

    # 1. Load database configuration
    db_config = get_database_config()

    engine = get_db_engine()

    try:
        # Test database connection
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        # Load data
        df = load_data(
            engine,
            db_config["table"]
        )


        print("Original dataset shape:", df.shape)

        # 4. Perform basic cleaning
        cleaned_df = clean_data(df)

        print("\nCleaned dataset shape:", cleaned_df.shape)

        # 5. Validate cleaned dataset
        print("\nRunning data validation...")

        validation_report = validate_data(cleaned_df)

        # 6. Create output directories
        output_dir = Path("data/interim")
        report_dir = Path("reports/cleaning")

        output_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        # 7. Save cleaned data
        cleaned_df.to_csv(
            output_dir / "validated_data.csv",
            index=False
        )

        # 8. Save validation report
        validation_report.to_csv(
            report_dir / "validation_report.csv",
            index=False
        )

        # 9. Display results
        print("\n========== VALIDATION REPORT ==========")

        print(validation_report.to_string(index=False))

        print("\nMissing values after cleaning:")

        print(cleaned_df.isnull().sum())

        print("\nFinal dataset shape:", cleaned_df.shape)

        print("\nValidated data saved successfully.")

        print("Data: data/interim/validated_data.csv")

        print("Report: reports/cleaning/validation_report.csv")

    finally:

        engine.dispose()


if __name__ == "__main__":

    main()
