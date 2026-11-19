
from pathlib import Path

import numpy as np
import pandas as pd


def main():

    print("\nStarting Step 8: Feature Signal Analysis")

    # --------------------------------------------------
    # 1. Define paths
    # --------------------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    input_dir = project_root / "data/processed"

    report_dir = project_root / "reports/feature_analysis"

    report_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 2. Load original training features and target
    # --------------------------------------------------

    X_train = pd.read_csv(
        input_dir / "X_train.csv"
    )

    y_train = pd.read_csv(
        input_dir / "y_train.csv"
    )["fraud_flag"]

    data = X_train.copy()

    data["fraud_flag"] = y_train.values

    print("\nTraining dataset shape:", data.shape)

    print("\nTarget distribution:")
    print(data["fraud_flag"].value_counts())

    print(
        "\nFraud percentage:",
        round(data["fraud_flag"].mean() * 100, 2),
        "%"
    )

    # --------------------------------------------------
    # 3. Separate numerical and categorical features
    # --------------------------------------------------

    numerical_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    print("\nNumerical features:", len(numerical_columns))
    print("Categorical features:", len(categorical_columns))

    # --------------------------------------------------
    # 4. Analyze numerical features
    # --------------------------------------------------

    print("\nAnalyzing numerical features...")

    numeric_results = []

    for column in numerical_columns:

        legitimate = data.loc[
            data["fraud_flag"] == 0, column
        ].dropna()

        fraud = data.loc[
            data["fraud_flag"] == 1, column
        ].dropna()

        mean_0 = legitimate.mean()
        mean_1 = fraud.mean()

        std_0 = legitimate.std()
        std_1 = fraud.std()

        pooled_std = np.sqrt(
            (std_0 ** 2 + std_1 ** 2) / 2
        )

        if pd.isna(pooled_std) or pooled_std == 0:
            effect_size = np.nan
        else:
            effect_size = (mean_1 - mean_0) / pooled_std

        numeric_results.append({

            "feature": column,

            "legitimate_mean": mean_0,

            "fraud_mean": mean_1,

            "legitimate_median": legitimate.median(),

            "fraud_median": fraud.median(),

            "legitimate_count": len(legitimate),

            "fraud_count": len(fraud),

            "effect_size": effect_size,

            "missing_values": data[column].isna().sum()

        })

    numeric_df = pd.DataFrame(numeric_results)

    numeric_df["absolute_effect_size"] = (
        numeric_df["effect_size"].abs()
    )

    numeric_df = numeric_df.sort_values(
        by="absolute_effect_size",
        ascending=False
    )

    numeric_df.to_csv(
        report_dir / "numeric_feature_analysis.csv",
        index=False
    )

    print("\n========== NUMERICAL FEATURE ANALYSIS ==========")

    print(
        numeric_df[
            [
                "feature",
                "legitimate_mean",
                "fraud_mean",
                "effect_size"
            ]
        ].round(3).to_string(index=False)
    )

    # --------------------------------------------------
    # 5. Analyze categorical features
    # --------------------------------------------------

    print("\nAnalyzing categorical features...")

    categorical_results = []

    for column in categorical_columns:

        category_analysis = (
            data.groupby(
                column,
                dropna=False
            )
            .agg(
                total_customers=("fraud_flag", "size"),
                fraud_cases=("fraud_flag", "sum"),
                fraud_rate=("fraud_flag", "mean")
            )
            .reset_index()
        )

        category_analysis.insert(
            0,
            "feature",
            column
        )

        category_analysis = category_analysis.rename(
            columns={column: "category"}
        )

        categorical_results.append(category_analysis)

    if categorical_results:

        categorical_df = pd.concat(
            categorical_results,
            ignore_index=True
        )

        categorical_df["fraud_rate"] *= 100

        categorical_df.to_csv(
            report_dir / "categorical_feature_analysis.csv",
            index=False
        )

        print("\n========== CATEGORICAL FEATURE ANALYSIS ==========")

        print(
            categorical_df.to_string(index=False)
        )

    # --------------------------------------------------
    # 6. Save missing value report
    # --------------------------------------------------

    missing_report = (
        X_train.isnull()
        .sum()
        .reset_index()
    )

    missing_report.columns = [
        "feature",
        "missing_values"
    ]

    missing_report.to_csv(
        report_dir / "missing_values.csv",
        index=False
    )

    # --------------------------------------------------
    # 7. Final summary
    # --------------------------------------------------

    print("\n========== STEP 8 SUMMARY ==========")

    print("Numerical features analyzed:", len(numerical_columns))

    print("Categorical features analyzed:", len(categorical_columns))

    print("Fraud cases available for analysis:", int(y_train.sum()))

    print("\nReports saved to:", report_dir)

    print("\nStep 8 completed successfully!")


if __name__ == "__main__":

    main()
