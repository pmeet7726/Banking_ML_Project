
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from banking_ml.features.feature_engineering import create_features


def main():

    print("\nStarting Step 4: Feature Engineering and Train/Test Split")

    # 1. Load environment variables
    load_dotenv()

    reference_date = os.getenv("FEATURE_REFERENCE_DATE")

    if not reference_date:
        raise ValueError(
            "FEATURE_REFERENCE_DATE is missing from .env"
        )

    # 2. Define project paths
    project_root = Path(__file__).resolve().parents[1]

    input_path = (
        project_root / "data/interim/validated_data.csv"
    )

    output_dir = project_root / "data/processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load validated dataset
    print("\nLoading validated dataset...")

    df = pd.read_csv(
        input_path,
        parse_dates=["last_login_date"]
    )

    print("Original dataset shape:", df.shape)

    # 4. Create engineered features
    df = create_features(
        df,
        reference_date=reference_date
    )

    print("\nDataset after feature engineering:", df.shape)

    # 5. Check missing and infinite values
    print("\nMissing values in engineered dataset:")

    print(df.isnull().sum()[df.isnull().sum() > 0])

    numeric_df = df.select_dtypes(include="number")

    print(
        "\nInfinite values:",
        int(numeric_df.isin([float("inf"), float("-inf")]).sum().sum())
    )

    # 6. Separate target and customer IDs
    X = df.drop(
        columns=["fraud_flag", "customer_id"]
    )

    y = df["fraud_flag"]

    customer_ids = df["customer_id"]

    # 7. Train/test split
    print("\nPerforming stratified train/test split...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        ids_train,
        ids_test
    ) = train_test_split(
        X,
        y,
        customer_ids,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 8. Save training and testing features
    X_train.to_csv(
        output_dir / "X_train.csv",
        index=False
    )

    X_test.to_csv(
        output_dir / "X_test.csv",
        index=False
    )

    # 9. Save target variables
    y_train.to_frame(name="fraud_flag").to_csv(
        output_dir / "y_train.csv",
        index=False
    )

    y_test.to_frame(name="fraud_flag").to_csv(
        output_dir / "y_test.csv",
        index=False
    )

    # 10. Save IDs separately for traceability
    ids_train.to_frame(name="customer_id").to_csv(
        output_dir / "train_customer_ids.csv",
        index=False
    )

    ids_test.to_frame(name="customer_id").to_csv(
        output_dir / "test_customer_ids.csv",
        index=False
    )

    # 11. Print final results
    print("\n========== SPLIT RESULTS ==========")

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    print("\nTraining fraud percentage:")
    print(round(y_train.mean() * 100, 2), "%")

    print("\nTesting fraud percentage:")
    print(round(y_test.mean() * 100, 2), "%")

    print("\nStep 4 completed successfully!")

    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
