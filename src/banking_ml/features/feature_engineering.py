
import pandas as pd


def create_features(df, reference_date):

    """
    Create additional banking features.

    Parameters:
        df: Validated customer dataset.
        reference_date: Date representing the dataset snapshot.

    Returns:
        DataFrame containing original and engineered features.
    """

    df = df.copy()

    print("\nStarting feature engineering...")

    # 1. Convert last login date
    df["last_login_date"] = pd.to_datetime(
        df["last_login_date"],
        errors="coerce"
    )

    # 2. Calculate days since last login
    reference_date = pd.Timestamp(reference_date)

    df["days_since_last_login"] = (
        reference_date - df["last_login_date"]
    ).dt.days

    # 3. Calculate monthly income
    monthly_income = df["annual_income"] / 12

    # 4. Monthly expense-to-income ratio
    df["monthly_expense_to_income_ratio"] = (
        df["monthly_expenses"] /
        monthly_income.where(monthly_income > 0)
    )

    # 5. Loan-to-annual-income ratio
    df["loan_to_annual_income_ratio"] = (
        df["loan_amount"] /
        df["annual_income"].where(df["annual_income"] > 0)
    )

    # 6. Savings-to-annual-income ratio
    df["savings_to_annual_income_ratio"] = (
        df["savings_balance"] /
        df["annual_income"].where(df["annual_income"] > 0)
    )

    # 7. Remove original date column
    # We have already extracted useful information from it.
    df = df.drop(columns=["last_login_date"])

    print("Feature engineering completed.")

    return df
