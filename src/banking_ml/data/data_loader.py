"""Utilities for loading project data."""

import re

import pandas as pd
from sqlalchemy import text


def load_data(engine, table_name):
    table_name = table_name.strip() if isinstance(table_name, str) else table_name

    # Validate table name before using it in SQL
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError(
            "Invalid MySQL table name. Use letters, digits, and underscores, "
            "starting with a letter or underscore; keep comments on separate lines."
        )

    query = text(f"SELECT * FROM `{table_name}`")

    # Execute query and load the result into Pandas
    with engine.connect() as connection:

        df = pd.read_sql_query(
            query,
            connection
        )

    return df
