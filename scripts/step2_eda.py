"""Entry point for the exploratory data analysis step."""

import os
from pathlib import Path

from sqlalchemy import text

from banking_ml.config import get_database_config
from banking_ml.database.connection import get_db_engine
from banking_ml.data.data_loader import load_data

from banking_ml.eda.data_summary import (
    get_dataset_overview,
    get_column_summary,
    get_numeric_summary,
    get_target_distribution
)

from banking_ml.eda.data_quality import (
    get_missing_values_report,
    get_duplicate_count,
    get_unique_values_report,
    get_outlier_report
)

from banking_ml.eda.visualization import (
    plot_missing_values,
    plot_numeric_distributions,
    plot_categorical_distributions
)

from banking_ml.eda.correlation import (
    get_correlation_matrix,
    plot_correlation_heatmap
)


def main():

    print("\nStarting EDA process...")

    config = get_database_config()

    # Read optional EDA settings
    target_column = os.getenv("EDA_TARGET_COLUMN", "").strip()

    exclude_columns = [
        column.strip()
        for column in os.getenv(
            "EDA_EXCLUDE_COLUMNS", ""
        ).split(",")
        if column.strip()
    ]

    max_categories = int(
        os.getenv("EDA_MAX_CATEGORIES", "20")
    )

    output_dir = Path("reports/eda")
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = get_db_engine()

    try:

        # 1. Test database connection
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        print("Database connection successful!")

        # 2. Load data
        df = load_data(
            engine,
            config["table"]
        )

        if df.empty:
            raise ValueError("The database table is empty.")

        print("\nData loaded successfully!")

        # 3. Dataset overview
        get_dataset_overview(df)

        # 4. Column summary
        column_summary = get_column_summary(df)

        column_summary.to_csv(
            output_dir / "column_summary.csv"
        )

        print("\nColumn Summary:")
        print(column_summary)

        # 5. Numerical descriptive statistics
        numeric_summary = get_numeric_summary(
            df,
            exclude_columns=exclude_columns
        )

        if not numeric_summary.empty:

            numeric_summary.to_csv(
                output_dir / "numeric_summary.csv"
            )

            print("\nNumerical Summary:")
            print(numeric_summary)

        # 6. Missing values
        missing_report = get_missing_values_report(df)

        missing_report.to_csv(
            output_dir / "missing_values.csv"
        )

        print("\nMissing Values:")
        print(missing_report)

        # 7. Duplicate records
        duplicate_count = get_duplicate_count(df)

        print("\nDuplicate Records:", duplicate_count)

        # 8. Unique values
        unique_report = get_unique_values_report(df)

        unique_report.to_csv(
            output_dir / "unique_values.csv"
        )

        print("\nUnique Values:")
        print(unique_report)

        # 9. Outlier analysis
        outlier_report = get_outlier_report(
            df,
            exclude_columns=exclude_columns
        )

        if not outlier_report.empty:

            outlier_report.to_csv(
                output_dir / "outlier_report.csv",
                index=False
            )

            print("\nPotential Outliers:")
            print(outlier_report)

        # 10. Target distribution
        if target_column:

            target_report = get_target_distribution(
                df,
                target_column
            )

            target_report.to_csv(
                output_dir / "target_distribution.csv"
            )

            print("\nTarget Distribution:")
            print(target_report)

        # 11. Visualizations
        plot_missing_values(df, output_dir)

        plot_numeric_distributions(
            df,
            output_dir,
            exclude_columns=exclude_columns
        )

        plot_categorical_distributions(
            df,
            output_dir,
            exclude_columns=exclude_columns,
            max_categories=max_categories
        )

        # 12. Correlation analysis
        correlation = get_correlation_matrix(
            df,
            exclude_columns=exclude_columns
        )

        if not correlation.empty:

            correlation.to_csv(
                output_dir / "correlation_matrix.csv"
            )

            plot_correlation_heatmap(
                df,
                output_dir,
                exclude_columns=exclude_columns
            )

        print("\nEDA completed successfully!")

        print("Reports saved in:", output_dir)

    except Exception as error:

        print(f"\nEDA process failed: {error}")
        raise

    finally:

        engine.dispose()


if __name__ == "__main__":
    main()

