"""Dataset summary utilities."""

import pandas as pd


def get_dataset_overview(df):
    """
    Display basic information about the dataset.
    """

    print("\n========== DATASET OVERVIEW ==========")

    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])

    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    print(f"Memory usage: {memory_mb:.2f} MB")


def get_column_summary(df):
    """
    Return a summary of every column.
    """

    summary = pd.DataFrame({
        "data_type": df.dtypes.astype(str),
        "non_null_count": df.count(),
        "missing_count": df.isnull().sum(),
        "missing_percentage": (
            df.isnull().mean() * 100
        ).round(2),
        "unique_count": df.nunique(dropna=True)
    })

    return summary


def get_numeric_summary(df, exclude_columns=None):

    if exclude_columns is None:
        exclude_columns = []

    numeric_df = df.select_dtypes(include="number")

    numeric_df = numeric_df.drop(
        columns=exclude_columns,
        errors="ignore"
    )

    if numeric_df.empty:
        return pd.DataFrame()

    return numeric_df.describe().T


def get_target_distribution(df, target_column):
    """
    Analyze the distribution of the target variable.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    counts = df[target_column].value_counts(dropna=False)

    percentages = (
        df[target_column]
        .value_counts(normalize=True, dropna=False) * 100
    )

    result = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(2)
    })

    return result
