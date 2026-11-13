"""Exploratory data visualization utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def plot_missing_values(df, output_dir):

    """
    Visualize missing values by column.
    """

    missing = df.isnull().sum()

    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    if missing.empty:
        print("No missing values found.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    sns.barplot(
        x=missing.index,
        y=missing.values
    )

    plt.title("Missing Values by Column")
    plt.xlabel("Columns")
    plt.ylabel("Missing Records")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        output_dir / "missing_values.png",
        dpi=150
    )

    plt.close()


def plot_numeric_distributions(
    df,
    output_dir,
    exclude_columns=None
):

    """
    Generate histograms for numerical columns.
    """

    if exclude_columns is None:
        exclude_columns = []

    numeric_df = df.select_dtypes(include="number")

    numeric_df = numeric_df.drop(
        columns=exclude_columns,
        errors="ignore"
    )

    if numeric_df.empty:
        print("No numerical columns available.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_df.hist(
        bins=30,
        figsize=(16, 12),
        edgecolor="black"
    )

    plt.suptitle("Numerical Feature Distributions")

    plt.tight_layout()

    plt.savefig(
        output_dir / "numeric_distributions.png",
        dpi=150
    )

    plt.close("all")


def plot_categorical_distributions(
    df,
    output_dir,
    exclude_columns=None,
    max_categories=20
):

    """
    Generate count plots for low-cardinality
    categorical columns.
    """

    if exclude_columns is None:
        exclude_columns = []

    categorical_df = df.select_dtypes(
        include=["object", "category", "bool"]
    )

    categorical_columns = [
        column
        for column in categorical_df.columns
        if column not in exclude_columns
        and df[column].nunique(dropna=True) <= max_categories
    ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for column in categorical_columns:

        plt.figure(figsize=(10, 5))

        counts = (
            df[column]
            .fillna("Missing")
            .value_counts()
            .head(max_categories)
        )

        sns.barplot(
            x=counts.index.astype(str),
            y=counts.values
        )

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")

        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        safe_name = f"{column}_distribution.png"

        plt.savefig(
            output_dir / safe_name,
            dpi=150
        )

        plt.close()
