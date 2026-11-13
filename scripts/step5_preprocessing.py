
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy import sparse

from banking_ml.preprocessing.preprocessing import (
    build_preprocessor
)


def save_processed_data(data, output_path):

    """
    Save transformed data.

    If the result is sparse, save it as NPZ.
    Otherwise, save it as NPY.
    """

    if sparse.issparse(data):

        sparse.save_npz(
            output_path.with_suffix(".npz"),
            data
        )

        return output_path.with_suffix(".npz")

    else:

        np.save(
            output_path.with_suffix(".npy"),
            data
        )

        return output_path.with_suffix(".npy")


def main():

    print("\nStarting Step 5: Data Preprocessing")

    # 1. Define paths
    project_root = Path(__file__).resolve().parents[1]

    input_dir = project_root / "data/processed"

    output_dir = project_root / "data/preprocessed"

    model_dir = project_root / "models/preprocessing"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load training and testing features
    print("\nLoading train and test datasets...")

    X_train = pd.read_csv(input_dir / "X_train.csv")
    X_test = pd.read_csv(input_dir / "X_test.csv")

    y_train = pd.read_csv(input_dir / "y_train.csv")
    y_test = pd.read_csv(input_dir / "y_test.csv")

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    # 3. Build preprocessing object
    preprocessor = build_preprocessor(X_train)

    # 4. Fit ONLY on training data
    print("\nFitting preprocessor on training data...")

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # 5. Transform test data using fitted preprocessor
    print("\nTransforming test data...")

    X_test_processed = preprocessor.transform(
        X_test
    )

    # 6. Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    feature_names_df = pd.DataFrame({
        "feature_name": feature_names
    })

    feature_names_df.to_csv(
        output_dir / "feature_names.csv",
        index=False
    )

    # 7. Save processed features
    train_path = save_processed_data(
        X_train_processed,
        output_dir / "X_train_processed"
    )

    test_path = save_processed_data(
        X_test_processed,
        output_dir / "X_test_processed"
    )

    # 8. Save target variables
    y_train.to_csv(
        output_dir / "y_train.csv",
        index=False
    )

    y_test.to_csv(
        output_dir / "y_test.csv",
        index=False
    )

    # 9. Save fitted preprocessor
    joblib.dump(
        preprocessor,
        model_dir / "preprocessor.joblib"
    )

    # 10. Print results
    print("\n========== PREPROCESSING RESULTS ==========")

    print(
        "Processed training shape:",
        X_train_processed.shape
    )

    print(
        "Processed testing shape:",
        X_test_processed.shape
    )

    print(
        "Number of transformed features:",
        len(feature_names)
    )

    print(
        "Training output file:",
        train_path
    )

    print(
        "Testing output file:",
        test_path
    )

    print(
        "Saved preprocessor:",
        model_dir / "preprocessor.joblib"
    )

    print("\nStep 5 completed successfully!")


if __name__ == "__main__":

    main()
