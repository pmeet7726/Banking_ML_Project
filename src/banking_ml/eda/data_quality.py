"""Data quality assessment utilities."""

import pandas as pd


def get_missing_values_report(df):

    """
    Calculate missing values for every column.
    """

    report = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percentage": (
            df.isnull().mean() * 100
        ).round(2)
    })

    report = report.sort_values(
        by="missing_percentage",
        ascending=False
    )

    return report


def get_duplicate_count(df):

    """
    Count completely duplicated rows.
    """

    return df.duplicated().sum()


def get_unique_values_report(df):

    """
    Count distinct values in every column.

    Does not expose the actual values.
    """

    report = pd.DataFrame({
        "unique_count": df.nunique(dropna=True),
        "total_records": len(df)
    })

    report["unique_percentage"] = (
        report["unique_count"] /
        report["total_records"] * 100
    ).round(2)

    return report


def get_outlier_report(df, exclude_columns=None):

    """
    Identify potential outliers using the IQR method.

    This function only identifies outliers.
    It does not remove or modify any records.
    """

    if exclude_columns is None:
        exclude_columns = []

    numeric_df = df.select_dtypes(include="number")

    numeric_df = numeric_df.drop(
        columns=exclude_columns,
        errors="ignore"
    )

    results = []

    for column in numeric_df.columns:

        series = numeric_df[column].dropna()

        # Skip binary or constant columns
        if series.nunique() <= 2:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[
            (series < lower_bound) |
            (series > upper_bound)
        ]

        results.append({
            "column": column,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": len(outliers),
            "outlier_percentage": (
                len(outliers) / len(series) * 100
            )
        })

    return pd.DataFrame(results)
