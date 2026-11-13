"""Correlation analysis utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def get_correlation_matrix(df, exclude_columns=None):

    """
    Calculate Pearson correlation between numerical features.
    """

    if exclude_columns is None:
        exclude_columns = []

    numeric_df = df.select_dtypes(include="number")

    numeric_df = numeric_df.drop(
        columns=exclude_columns,
        errors="ignore"
    )

    return numeric_df.corr(method="pearson")


def plot_correlation_heatmap(
    df,
    output_dir,
    exclude_columns=None
):

    """
    Generate a correlation heatmap.
    """

    correlation = get_correlation_matrix(
        df,
        exclude_columns
    )

    if correlation.empty:
        print("No numerical columns available for correlation.")
        return correlation

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        output_dir / "correlation_heatmap.png",
        dpi=150
    )

    plt.close()

    return correlation
