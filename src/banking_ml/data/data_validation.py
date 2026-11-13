
import pandas as pd


def validate_data(df):

    validation_results = []

    def add_result(check_name, column, invalid_count, description):

        validation_results.append({
            "check": check_name,
            "column": column,
            "invalid_count": int(invalid_count),
            "description": description,
            "status": "PASS" if invalid_count == 0 else "REVIEW"
        })

    # 1. Check duplicate customer IDs
    if "customer_id" in df.columns:

        duplicate_ids = df["customer_id"].duplicated().sum()

        add_result(
            "Duplicate customer IDs",
            "customer_id",
            duplicate_ids,
            "Customer IDs should be unique for this dataset."
        )

    # 2. Check fraud target values
    if "fraud_flag" in df.columns:

        invalid_target = (
            ~df["fraud_flag"].isin([0, 1])
        ).sum()

        add_result(
            "Invalid target values",
            "fraud_flag",
            invalid_target,
            "Target must contain only 0 or 1."
        )

    # 3. Check negative values in columns
    non_negative_columns = [
        "age",
        "annual_income",
        "monthly_expenses",
        "credit_card_count",
        "loan_amount",
        "interest_rate",
        "years_with_bank",
        "transaction_count",
        "avg_transaction_amount",
        "savings_balance"
    ]

    for column in non_negative_columns:

        if column in df.columns:

            invalid_count = df[column].lt(0).sum()

            add_result(
                "Negative values",
                column,
                invalid_count,
                "Expected non-negative values. Review before changing."
            )

    # 4. Check credit score range
    if "credit_score" in df.columns:

        invalid_credit_scores = (
            (df["credit_score"] < 300) |
            (df["credit_score"] > 900)
        ).sum()

        add_result(
            "Credit score range",
            "credit_score",
            invalid_credit_scores,
            "Expected credit score between 300 and 900."
        )

    # 5. Check invalid or missing dates
    if "last_login_date" in df.columns:

        invalid_dates = df["last_login_date"].isna().sum()

        add_result(
            "Missing or invalid dates",
            "last_login_date",
            invalid_dates,
            "Check dates that could not be parsed or were missing."
        )

    # 6. Generate validation report
    validation_report = pd.DataFrame(validation_results)

    return validation_report
