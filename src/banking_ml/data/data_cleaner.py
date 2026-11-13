"""Utilities for cleaning tabular banking data."""


import pandas as pd


def clean_data(df):
    """
    Perform basic data cleaning.

    Important:
    - Does not perform missing-value imputation.
    - Does not remove outliers.
    - Does not remove customer records based on assumptions.
    """

    df = df.copy()

    print("\nStarting data cleaning...")

    # 1. Remove exact duplicate records
    duplicates_before = df.duplicated().sum()

    df = df.drop_duplicates()

    print(f"Duplicate records removed: {duplicates_before}")

    # 2. Standardize categorical columns
    categorical_columns = [
        "gender",
        "city",
        "employment_status",
        "loan_type",
        "customer_segment"
    ]

    for column in categorical_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
            )

    # 3. Convert numeric columns
    numeric_columns = [
        "customer_id",
        "age",
        "annual_income",
        "monthly_expenses",
        "account_balance",
        "credit_score",
        "credit_card_count",
        "loan_amount",
        "interest_rate",
        "years_with_bank",
        "transaction_count",
        "avg_transaction_amount",
        "savings_balance",
        "fraud_flag"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # 4. Convert last_login_date into datetime
    if "last_login_date" in df.columns:

        df["last_login_date"] = pd.to_datetime(
            df["last_login_date"],
            errors="coerce"
        )

    print("Basic data cleaning completed.")

    return df

