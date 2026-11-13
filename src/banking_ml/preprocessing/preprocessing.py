
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X_train):
    """
    Build a preprocessing pipeline.

    Numerical columns:
        Median imputation + StandardScaler

    Categorical columns:
        Most frequent imputation + OneHotEncoder
    """

    print("\nIdentifying numerical and categorical columns...")

    # 1. Identify numerical columns
    numerical_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # 2. Identify categorical columns
    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    print("\nNumerical columns:")
    print(numerical_columns)

    print("\nCategorical columns:")
    print(categorical_columns)

    # 3. Numerical preprocessing pipeline
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # 4. Categorical preprocessing pipeline
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    # 5. Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ],
        remainder="drop"
    )

    return preprocessor
