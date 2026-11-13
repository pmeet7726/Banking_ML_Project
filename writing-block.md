# End-to-End Banking Machine Learning Project
## Complete Project Notes — Steps 1 to 4
### Interview Preparation and Technical Reference

---

# 1. Project Overview

## 1.1 Project objective

The objective of this project is to build an end-to-end banking machine learning pipeline that predicts whether a customer or transaction is potentially fraudulent.

We are using customer banking information stored in a MySQL database.

The project follows a modular, production-oriented approach rather than putting all the code inside one Jupyter notebook.

The project covers:

1. Data ingestion from MySQL.
2. Exploratory Data Analysis (EDA).
3. Data cleaning and validation.
4. Feature engineering.
5. Train/test splitting.
6. Data preprocessing.
7. Model training.
8. Model evaluation.
9. Model deployment.
10. Monitoring and logging.

Our current progress covers the first four stages, with Step 4 implementation prepared and awaiting execution verification.

## 1.2 Business problem

Banks need to identify potentially fraudulent customer activity.

Manually reviewing every customer or transaction is inefficient.

Machine learning can help identify patterns associated with fraudulent behaviour and provide a prediction that can support fraud investigation.

Our target variable is:

`fraud_flag`

- 0 = Legitimate customer/record
- 1 = Fraudulent customer/record

**Important:** Our current dataset contains customer-level records and a fraud label. It is a demonstration classification project, not necessarily a production transaction-level fraud detection system.

## 1.3 Technical stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| MySQL | Data storage |
| SQLAlchemy | Database connection and SQL execution |
| PyMySQL | MySQL database driver |
| Python-dotenv | Environment configuration |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualizations |
| Scikit-learn | ML preprocessing, splitting and modelling |
| Git | Version control |
| VS Code / IntelliJ | Development environment |

---

# 2. Project Architecture

We are following a modular Python project structure.

```text
Banking_ml_project/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│
├── reports/
│   ├── eda/
│   └── cleaning/
│
├── src/
│   └── banking_ml/
│       │
│       ├── database/
│       │   ├── __init__.py
│       │   └── connection.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── data_loader.py
│       │   ├── data_cleaner.py
│       │   └── data_validation.py
│       │
│       ├── eda/
│       │   ├── __init__.py
│       │   ├── data_summary.py
│       │   ├── data_quality.py
│       │   ├── visualization.py
│       │   └── correlation.py
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   └── feature_engineering.py
│       │
│       └── config.py
│
├── scripts/
│   ├── step1_load_data.py
│   ├── step2_eda.py
│   ├── step3_data_cleaning.py
│   └── step4_feature_engineering.py
│
├── tests/
│
├── .env
├── .env.example
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## 2.1 Why did we create this structure?

Instead of writing everything in one Python file, we separate responsibilities.

For example:

- `database/connection.py` handles database connections.
- `data/data_loader.py` loads data.
- `data/data_cleaner.py` cleans data.
- `eda/` contains exploratory analysis functions.
- `features/` contains feature engineering functions.
- `scripts/` contains executable workflows.

This is called modular programming.

### Interview question: Why do we use modular architecture?

**Answer:**

Modular architecture separates different responsibilities into independent components. It improves code reusability, maintainability, readability and testability.

For example, if the data-cleaning logic changes, we can modify `data_cleaner.py` without rewriting the database connection logic.

---

# 3. Step 1 — Data Ingestion

## 3.1 What is data ingestion?

Data ingestion means retrieving data from a source and loading it into our application so that we can process it.

In our project:

- Source: MySQL database.
- Destination: Pandas DataFrame.
- Connection: SQLAlchemy and PyMySQL.

The data is loaded from MySQL rather than manually downloading a CSV file.

## 3.2 Why did we use SQLAlchemy?

SQLAlchemy is a Python SQL toolkit and database abstraction library.

It allows Python applications to connect to databases using a consistent interface.

We use:

```python
from sqlalchemy import create_engine
```

We create an engine that manages the database connection infrastructure.

Example:

```python
engine = create_engine(
    database_url,
    pool_pre_ping=True,
    echo=False
)
```

Important parameters:

**pool_pre_ping=True**

Checks whether a pooled connection is still alive before using it.

This helps avoid errors caused by stale database connections.

**echo=False**

Prevents SQLAlchemy from printing SQL statements to the console.

## 3.3 Why use PyMySQL?

PyMySQL is a Python MySQL database driver.

SQLAlchemy provides the abstraction layer, while PyMySQL allows Python to communicate with MySQL.

The connection URL uses:

```text
mysql+pymysql://
```

## 3.4 Why use `.env`?

Database credentials should not be hardcoded in Python files.

Instead, we store configuration in environment variables.

Example:

```text
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=your_database
DB_TABLE=your_table
```

Python reads these values using:

```python
from dotenv import load_dotenv
import os

load_dotenv()

db_host = os.getenv("DB_HOST")
```

Advantages:

- Credentials are separated from source code.
- Different environments can use different configurations.
- We avoid hardcoding passwords.
- Configuration can be changed without changing application logic.

**Security rule:** Never commit `.env` to GitHub. Add it to `.gitignore`.

## 3.5 What does `get_database_config()` do?

It reads the environment variables and returns a Python dictionary containing database configuration.

Conceptually:

```python
config = {
    "host": "...",
    "port": "...",
    "user": "...",
    "password": "...",
    "database": "...",
    "DB_TABLE": "..."
}
```

One issue we encountered:

```text
KeyError: 'DB_TABLE'
```

This happened because the configuration dictionary did not contain the key that the script attempted to access.

We learned an important programming lesson:

The dictionary key used in a script must exactly match the key returned by the configuration function.

## 3.6 What does `data_loader.py` do?

It contains the reusable function that loads the database table.

Conceptually:

```python
def load_data(engine, table_name):

    query = f"SELECT * FROM `{table_name}`"

    df = pd.read_sql_query(
        query,
        engine
    )

    return df
```

Our implementation also validates the table identifier before using it in SQL.

This is important because SQL table names generally cannot be passed as ordinary bound parameters in the same way as query values.

Validating the identifier helps prevent unsafe table-name injection.

## 3.7 Why use `pd.read_sql_query()`?

It executes a SQL query and returns the results as a Pandas DataFrame.

```python
df = pd.read_sql_query(query, engine)
```

Now we can use Pandas for data analysis and machine learning.

## 3.8 Step 1 actual output

Our MySQL connection succeeded.

Dataset shape:

```text
(1000, 20)
```

This means:

- 1,000 rows.
- 20 columns.

The data was successfully loaded.

## 3.9 Our dataset columns

| Column | Description |
|---|---|
| customer_id | Unique customer identifier |
| age | Customer age |
| gender | Customer gender category |
| city | Customer location |
| employment_status | Employment category |
| annual_income | Annual customer income |
| monthly_expenses | Monthly expenses |
| account_balance | Bank account balance |
| credit_score | Customer credit score |
| credit_card_count | Number of credit cards |
| loan_amount | Customer loan amount |
| loan_type | Type of loan |
| interest_rate | Loan interest rate |
| years_with_bank | Customer relationship duration |
| transaction_count | Number of transactions |
| avg_transaction_amount | Average transaction amount |
| savings_balance | Savings account balance |
| last_login_date | Most recent login date |
| customer_segment | Customer segment |
| fraud_flag | Fraud target variable |

## 3.10 Initial missing values

| Column | Missing values |
|---|---:|
| monthly_expenses | 23 |
| credit_score | 27 |
| savings_balance | 44 |
| All other columns | 0 |

Total missing entries:

94.

There are 20,000 cells in the dataset, so 94 missing entries represent 0.47% of all cells.

## 3.11 Target distribution

```text
fraud_flag

0    962
1     38
```

Fraud percentage:

```text
3.8%
```

Legitimate percentage:

```text
96.2%
```

This tells us that our classification dataset is imbalanced.

### Important interview question: What is class imbalance?

Class imbalance occurs when one target class has significantly more observations than another.

In our dataset:

- Legitimate class: 96.2%.
- Fraud class: 3.8%.

A model could predict every record as legitimate and still achieve 96.2% accuracy.

However, it would fail to identify fraudulent records.

Therefore, accuracy alone is not sufficient.

We will later use:

- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision / PR-AUC

---

# 4. Step 2 — Exploratory Data Analysis (EDA)

## 4.1 What is EDA?

Exploratory Data Analysis is the process of understanding the dataset before building a machine learning model.

We investigate:

- Data distributions.
- Missing values.
- Duplicates.
- Outliers.
- Numerical relationships.
- Categorical distributions.
- Target distribution.

The goal is to understand the quality and characteristics of our data.

## 4.2 Why is EDA important?

Imagine building a fraud detection model without investigating the data.

We might discover later that:

- Some columns contain missing values.
- Some values are incorrectly formatted.
- One feature contains extreme outliers.
- The target is heavily imbalanced.
- Some columns are identifiers rather than useful predictive features.

EDA helps us discover these issues early.

## 4.3 What did we implement?

We created a dedicated EDA package.

### A. `data_summary.py`

Contains functions to understand the structure of the dataset.

**get_dataset_overview(df)**

Provides:

- Number of rows.
- Number of columns.
- Memory usage.

**get_column_summary(df)**

Provides:

- Column names.
- Data types.
- Non-null counts.
- Missing counts.
- Missing percentages.
- Unique-value counts.

**get_numeric_summary(df)**

Uses:

```python
df.describe().T
```

This produces descriptive statistics for numerical columns.

Important statistics:

| Statistic | Meaning |
|---|---|
| count | Number of non-missing values |
| mean | Average |
| std | Standard deviation |
| min | Minimum |
| 25% | First quartile |
| 50% | Median |
| 75% | Third quartile |
| max | Maximum |

Why `.T`?

It transposes the result so that each numerical feature appears as a row, making the report easier to read.

We excluded identifier columns such as `customer_id` from numerical summaries.

### B. `data_quality.py`

This module investigates data quality.

**get_missing_values_report(df)**

Calculates missing values:

```python
df.isnull().sum()
```

Calculates missing percentages:

```python
df.isnull().mean() * 100
```

Why does `.mean()` work here?

Pandas treats:

- True as 1.
- False as 0.

Therefore, calculating the mean of the missing-value mask gives the missing proportion.

**get_duplicate_count(df)**

Uses:

```python
df.duplicated().sum()
```

This identifies exact duplicate rows.

**get_unique_values_report(df)**

Calculates the number of distinct values in each column.

This helps identify:

- Identifier-like columns.
- Low-cardinality categorical features.
- Potentially unexpected categories.

**get_outlier_report(df)**

Uses the Interquartile Range method.

Formula:

```text
IQR = Q3 - Q1

Lower boundary = Q1 - 1.5 × IQR

Upper boundary = Q3 + 1.5 × IQR
```

Values outside these boundaries are flagged as potential outliers.

Important:

An outlier is not automatically an incorrect value.

For example, a customer earning $500,000 annually may be unusual but perfectly legitimate.

We should investigate outliers before removing or changing them.

### C. `visualization.py`

We generated charts using Matplotlib and Seaborn.

We implemented:

1. Missing-value chart.
2. Numerical distribution histograms.
3. Categorical distribution plots.

Why visualize data?

Numerical reports tell us the statistics, but charts help us identify patterns.

For example:

- A histogram can show skewness.
- A bar chart can show category imbalance.
- A missing-value chart can show which columns require attention.

### D. `correlation.py`

We implemented correlation analysis.

Pearson correlation measures linear relationships between numerical variables.

The correlation coefficient ranges from -1 to +1.

| Correlation | Interpretation |
|---|---|
| +1 | Perfect positive linear relationship |
| 0 | No linear relationship |
| -1 | Perfect negative linear relationship |

Example:

If annual income and savings balance have a positive correlation, customers with higher income may also tend to have higher savings.

But correlation does not prove causation.

Also, a low Pearson correlation does not mean a feature has no predictive value. It may have a nonlinear relationship.

## 4.4 Reports generated

Our Step 2 script generated reports under:

```text
reports/eda/
```

Expected files include:

```text
column_summary.csv
numeric_summary.csv
missing_values.csv
unique_values.csv
outlier_report.csv
target_distribution.csv
correlation_matrix.csv
missing_values.png
numeric_distributions.png
correlation_heatmap.png
```

Some outputs are conditional. For example, a missing-value chart is only generated when missing values exist.

## 4.5 Important EDA concepts for interviews

**What is the difference between mean and median?**

Mean is the arithmetic average.

Median is the middle value after sorting.

Median is less sensitive to extreme values.

For skewed financial data, median can be a more representative measure of central tendency.

**What is skewness?**

Skewness describes the asymmetry of a distribution.

- Positive skew: longer right tail.
- Negative skew: longer left tail.
- Approximately zero: relatively symmetric distribution.

**What is an outlier?**

An observation that is unusually far from the majority of observations.

It may be a genuine observation or a data-quality problem.

**What is correlation?**

A numerical measure describing the strength and direction of a relationship between two variables.

---

# 5. Step 3 — Data Cleaning and Validation

## 5.1 What is data cleaning?

Data cleaning is the process of correcting or standardizing data so it can be reliably used for downstream processing.

Examples:

- Removing exact duplicates.
- Standardizing text.
- Correcting data types.
- Identifying invalid values.
- Handling missing values where appropriate.

## 5.2 What is data validation?

Data validation checks whether the data follows expected rules.

For example:

- Customer IDs should be unique.
- Fraud target should contain only 0 and 1.
- Credit scores should fall within the expected range.
- Dates should be valid.

Cleaning modifies data where appropriate.

Validation checks whether the data meets defined rules.

They are related but different activities.

## 5.3 Why did we create separate files?

We created:

```text
data_cleaner.py
data_validation.py
```

The reason is separation of responsibilities.

The cleaning module performs data transformations.

The validation module checks data quality.

This makes the code easier to maintain and test.

## 5.4 Cleaning operations implemented

### A. Remove exact duplicates

```python
df = df.drop_duplicates()
```

This removes identical rows.

Our output:

```text
Duplicate records removed: 0
```

No exact duplicate records were found.

Important distinction:

Two records with the same customer ID but different data are not necessarily exact duplicates.

That is why we separately validate customer IDs.

### B. Standardize categorical columns

We used:

```python
df[column] = (
    df[column]
    .astype("string")
    .str.strip()
    .replace("", pd.NA)
)
```

This performs three operations.

`astype("string")`

Converts the column to Pandas string dtype.

`.str.strip()`

Removes leading and trailing whitespace.

`.replace("", pd.NA)`

Converts empty strings into missing values.

Example:

```text
" Toronto"
"Toronto "
"Toronto"
```

After standardization:

```text
"Toronto"
"Toronto"
"Toronto"
```

Why is this important?

Without standardization, the model may interpret these as separate categories.

### C. Convert numeric columns

We used:

```python
pd.to_numeric(
    df[column],
    errors="coerce"
)
```

This converts the column into a numerical type.

If an invalid value is encountered, `errors="coerce"` converts it to NaN.

Example:

```text
"120000" → 120000
"invalid" → NaN
```

This prevents invalid strings from remaining inside numerical columns.

However, coercion can create new missing values, so we must validate after conversion.

### D. Convert dates

We used:

```python
pd.to_datetime(
    df["last_login_date"],
    errors="coerce"
)
```

This converts date strings into proper datetime values.

Invalid dates become `NaT`.

This allows us to perform date calculations later.

## 5.5 Data validation operations

We implemented the following checks.

**1. Duplicate customer IDs**

```python
df["customer_id"].duplicated().sum()
```

A customer ID should be unique because our dataset represents one record per customer.

**2. Target validation**

```python
~df["fraud_flag"].isin([0, 1])
```

The fraud target should contain only 0 and 1.

**3. Negative-value validation**

We checked columns such as:

- age
- annual_income
- monthly_expenses
- credit_card_count
- loan_amount
- interest_rate
- years_with_bank
- transaction_count
- avg_transaction_amount
- savings_balance

Why didn't we include `account_balance`?

Because bank account balances can legitimately be negative due to overdrafts.

This is an important example of applying business knowledge to data validation.

**4. Credit score validation**

```python
(df["credit_score"] < 300) | (
    df["credit_score"] > 900
)
```

We assumed the expected credit score range was 300–900.

This range should always be confirmed against the credit-score definition and source used by the actual business.

**5. Date validation**

```python
df["last_login_date"].isna().sum()
```

This identifies missing or invalid dates after conversion.

## 5.6 Why didn't we impute missing values in Step 3?

This is one of the most important ML concepts in our project.

Suppose we have 1,000 records.

We calculate the median credit score using all 1,000 records.

Then we split the data into:

- Training: 800 records.
- Testing: 200 records.

The calculated median has already used information from the test set.

This creates data leakage.

Instead, we will:

1. Split the data.
2. Fit the imputer using training data only.
3. Transform training data.
4. Transform testing data using the same fitted imputer.

This ensures the test set does not influence the training preprocessing statistics.

## 5.7 Step 3 actual execution output

Original shape:

```text
(1000, 20)
```

Cleaned shape:

```text
(1000, 20)
```

Duplicates removed:

```text
0
```

All 14 implemented validation checks passed.

Missing values remaining:

```text
monthly_expenses    23
credit_score        27
savings_balance     44
```

Date conversion succeeded.

The output was saved to:

```text
data/interim/validated_data.csv
```

Validation report:

```text
reports/cleaning/validation_report.csv
```

## 5.8 What does PASS actually mean?

A PASS means that the particular validation rule we implemented did not identify an issue.

It does not prove that every record is correct.

For example, our current validation does not check every possible business rule.

We will continue improving validation as the project develops.

---

# 6. Step 4 — Feature Engineering and Train/Test Split

Status: Implementation prepared; execution output still needs verification.

## 6.1 What is feature engineering?

Feature engineering is the process of creating or transforming input variables to help a machine learning model learn useful patterns.

A model's performance depends not only on the algorithm but also on the quality and relevance of its input features.

For example:

Instead of only giving a model monthly expenses and annual income, we can calculate how much of the customer's monthly income is spent.

This ratio may provide more meaningful information.

## 6.2 Features we designed

### Feature 1: Days since last login

Formula:

```text
days_since_last_login =
reference_date - last_login_date
```

Example:

```text
Reference date: 2026-11-13
Last login:     2026-11-03

Days since login = 10
```

Why useful?

It may help identify customers who have not logged into their banking account recently.

Important: The reference date must represent the actual dataset snapshot date. It must not be chosen arbitrarily.

### Feature 2: Monthly expense-to-income ratio

Formula:

```text
monthly_income = annual_income / 12

monthly_expense_to_income_ratio =
monthly_expenses / monthly_income
```

Example:

```text
Annual income = 120,000
Monthly income = 10,000
Monthly expenses = 5,000

Ratio = 5,000 / 10,000

Ratio = 0.5
```

Interpretation:

The customer spends 50% of their monthly income.

### Feature 3: Loan-to-annual-income ratio

Formula:

```text
loan_to_annual_income_ratio =
loan_amount / annual_income
```

Example:

```text
Loan amount = 60,000
Annual income = 120,000

Ratio = 0.5
```

Interpretation:

The loan amount equals 50% of annual income.

### Feature 4: Savings-to-annual-income ratio

Formula:

```text
savings_to_annual_income_ratio =
savings_balance / annual_income
```

Example:

```text
Savings balance = 30,000
Annual income = 120,000

Ratio = 0.25
```

Interpretation:

The customer's savings equal 25% of annual income.

These are candidate features. Their actual usefulness must be tested.

## 6.3 Why use `.where()`?

We use:

```python
df["annual_income"].where(
    df["annual_income"] > 0
)
```

This prevents division by zero.

If annual income is zero, the denominator becomes missing.

We avoid creating infinite values.

The resulting ratio may be NaN, which we will handle later using our preprocessing pipeline.

## 6.4 Why use `df.copy()`?

```python
df = df.copy()
```

It creates a separate DataFrame for transformations.

This avoids unintentionally modifying the original input DataFrame.

This is useful for reusable functions and predictable processing.

## 6.5 Why remove `last_login_date`?

We have extracted the relevant date information into:

```text
days_since_last_login
```

The original raw date is not intended to be passed directly into the model.

The new numerical feature is more suitable for our initial modelling pipeline.

## 6.6 What is X and y?

This is a fundamental machine learning concept.

**X = independent variables / features**

These are the input columns used by the model.

**y = dependent variable / target**

This is the value that the model attempts to predict.

In our project:

```python
X = df.drop(
    columns=["fraud_flag", "customer_id"]
)

y = df["fraud_flag"]
```

Why remove `fraud_flag`?

It is the answer we are trying to predict.

If we include it in X, the model gets access to the correct answer during training.

This is direct target leakage.

Why remove `customer_id`?

It is an identifier, not a meaningful customer behaviour feature.

We retain customer IDs separately for traceability.

## 6.7 What is train/test splitting?

We divide our dataset into separate groups.

Training data is used to teach the model.

Testing data is used to evaluate its performance on unseen records.

We chose an 80/20 split.

```text
Total records: 1,000

Training: 800 records (80%)

Testing:  200 records (20%)
```

## 6.8 Why use `train_test_split()`?

We use Scikit-learn:

```python
from sklearn.model_selection import train_test_split
```

Example:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Understanding the parameters

**test_size=0.20**

20% of the data is reserved for testing.

**random_state=42**

Makes the random split reproducible.

If we run the same split again with unchanged data and parameters, we obtain the same allocation.

42 is simply a conventional example seed, not a special ML value.

**stratify=y**

Maintains approximately the same target class proportions in the training and testing datasets.

This is especially important for our imbalanced fraud dataset.

## 6.9 Expected class distribution

Original distribution:

```text
Legitimate: 962
Fraud:       38
```

Expected split:

| Dataset | Legitimate | Fraud | Total |
|---|---:|---:|---:|
| Training | 770 | 30 | 800 |
| Testing | 192 | 8 | 200 |
| Total | 962 | 38 | 1,000 |

These are the expected counts for the stated stratified split.

## 6.10 Why stratification matters

Imagine we randomly split the dataset and accidentally put most fraudulent records into the training set.

The model would have very few fraud examples in testing.

Its evaluation would be unreliable.

Stratification helps maintain the original class distribution in each partition.

## 6.11 Why save customer IDs separately?

We designed these files:

```text
train_customer_ids.csv
test_customer_ids.csv
```

This allows us to trace predictions back to their original customer records without giving customer IDs to the model.

In real banking applications, access to identifiers and customer information must follow strict privacy and access-control requirements.

## 6.12 Expected Step 4 outputs

```text
data/processed/
│
├── X_train.csv
├── X_test.csv
├── y_train.csv
├── y_test.csv
├── train_customer_ids.csv
└── test_customer_ids.csv
```

These files are created by the Step 4 script once it runs successfully.

---

# 7. Most Important ML Concepts Learned So Far

## 7.1 Data leakage

Data leakage happens when information that would not legitimately be available during model training or prediction influences the model.

Example:

Calculating the median imputation value using both training and testing data.

Correct approach:

Fit the imputer on training data only.

Other examples of leakage:

- Including the target variable in X.
- Using information recorded after the prediction time.
- Performing feature selection using the full dataset before splitting.
- Scaling the entire dataset before splitting.
- Tuning a model using the test set.

## 7.2 Overfitting

Overfitting happens when a model learns the training data too closely, including noise and accidental patterns.

It performs well on training data but poorly on unseen data.

We will investigate overfitting during model training by comparing training and validation performance.

## 7.3 Underfitting

Underfitting happens when a model is too simple to learn the patterns in the data.

It performs poorly on both training and testing data.

## 7.4 Reproducibility

Reproducibility means that the same code, data and configuration should produce consistent results.

We use:

```python
random_state=42
```

We should also:

- Fix package versions.
- Keep configuration separate from code.
- Version-control source code.
- Track dataset versions.
- Record model parameters.
- Record the reference date for time-dependent features.

## 7.5 Why do we keep raw, interim and processed folders?

| Folder | Purpose |
|---|---|
| raw | Original source data |
| interim | Intermediate cleaned data |
| processed | Model-ready partitions or transformed data |

This makes the project easier to understand and reproduce.

We should not overwrite original source data during cleaning.

---

# 8. Critical Rules to Follow While Building This Project

These are the engineering principles I want you to remember throughout the remaining stages.

## Rule 1: Never hardcode database credentials

Use environment variables and protect `.env`.

## Rule 2: Never modify raw data directly

Always create a copy or write to a separate output location.

## Rule 3: Investigate missing values before imputing

Understand why values are missing and whether the missingness itself has meaning.

## Rule 4: Never remove outliers automatically

Investigate them first.

A large financial value may be legitimate.

## Rule 5: Avoid data leakage

Fit learned transformations only on training data.

## Rule 6: Use pipelines

Use Scikit-learn `Pipeline` and `ColumnTransformer` to combine preprocessing and model training.

## Rule 7: Use the correct evaluation metrics

Fraud detection is imbalanced, so accuracy alone is misleading.

## Rule 8: Preserve reproducibility

Use fixed seeds, recorded configurations and consistent dataset versions.

## Rule 9: Validate every stage

Check data shapes, data types, missing values, invalid values and target distribution.

## Rule 10: Write reusable code

Separate loading, cleaning, feature engineering, preprocessing and modelling into independent modules.

## Rule 11: Protect customer information

Do not expose real customer records in logs, GitHub repositories, screenshots or reports.

## Rule 12: Use domain knowledge

A negative account balance may be legitimate. A zero annual income may require investigation. Validation rules should reflect the actual business context.

---

# 9. Current Project Status

| Step | Description | Status |
|---|---|---|
| 1 | Data ingestion from MySQL | Completed |
| 2 | Exploratory Data Analysis | Completed |
| 3 | Data cleaning and validation | Completed |
| 4 | Feature engineering and train/test split | Implementation prepared; execution pending |
| 5 | Preprocessing pipeline | Not started |
| 6 | Model training | Not started |
| 7 | Model evaluation | Not started |
| 8 | Model selection and tuning | Not started |
| 9 | Model serialization | Not started |
| 10 | API development | Not started |
| 11 | Docker and deployment | Not started |
| 12 | Monitoring and logging | Not started |

---

# 10. Interview Questions and Answers

## Q1. Explain your banking ML project.

**Answer:**

I am building an end-to-end banking fraud classification pipeline using Python, MySQL and Scikit-learn.

The data is stored in MySQL, and I use SQLAlchemy and Pandas to ingest it.

I perform EDA to understand the dataset, investigate missing values, identify potential outliers and analyze the target distribution.

I then implement modular data cleaning and validation functions.

After that, I engineer financial and behavioural features such as expense-to-income ratio, loan-to-income ratio, savings-to-income ratio and days since last login.

I split the dataset into training and testing partitions using stratified sampling because the fraud class is underrepresented.

The next stages involve preprocessing, model training, evaluation, deployment and monitoring.

## Q2. Why did you choose a fraud detection problem?

Fraud detection is an important banking use case. It is also a useful classification problem for learning about imbalanced datasets, precision, recall, threshold selection and model evaluation.

## Q3. Why did you use MySQL?

MySQL is a relational database commonly used to store structured data.

Using it allows us to build a realistic data ingestion workflow instead of depending entirely on manually downloaded CSV files.

## Q4. Why SQLAlchemy instead of writing database connection code everywhere?

SQLAlchemy provides a reusable database abstraction and connection management layer.

We centralize connection creation and reuse it across the project.

## Q5. What is the difference between EDA and data cleaning?

EDA helps us understand the dataset and identify issues.

Data cleaning applies appropriate transformations to address data-quality problems.

EDA is primarily investigative, while cleaning is primarily corrective.

## Q6. How did you identify outliers?

We used the Interquartile Range method.

IQR is calculated as Q3 minus Q1.

Potential outliers are observations below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR.

However, I do not automatically remove them because financial data can naturally contain extreme values.

## Q7. Why use median imputation?

Median is less sensitive to extreme observations than mean.

For skewed financial variables, median can be a reasonable initial imputation strategy.

The imputer will be fitted using training data only.

## Q8. What is the difference between `drop_duplicates()` and checking duplicate IDs?

`drop_duplicates()` identifies identical complete rows.

Checking duplicate IDs identifies records that share the same customer identifier, even if other column values differ.

Both checks are necessary for different reasons.

## Q9. Why did you convert date columns to datetime?

Datetime conversion allows date calculations, sorting, comparisons and feature extraction.

For example, we can calculate days since a customer's last login.

## Q10. Why create financial ratios?

Ratios provide relative information.

A loan of $50,000 can mean different things for customers with annual incomes of $60,000 and $300,000.

A loan-to-income ratio provides a more comparable measure.

## Q11. Why remove customer_id?

Customer IDs are identifiers and generally do not represent meaningful customer behaviour.

Including them can encourage a model to learn accidental patterns.

We retain them separately for traceability.

## Q12. Why use stratified splitting?

Our target is highly imbalanced.

Stratification helps preserve the fraud-to-legitimate ratio in training and testing sets.

## Q13. Why not evaluate the model using training data?

Training data has already been used to fit the model.

Evaluating on the same data can produce overly optimistic results.

Testing on unseen records provides a better estimate of generalization performance.

## Q14. What is the difference between precision and recall?

Precision answers:

Of all records predicted as fraud, how many were actually fraud?

Recall answers:

Of all actual fraud records, how many did the model successfully identify?

For fraud detection, both matter.

## Q15. What is the difference between ROC-AUC and PR-AUC?

ROC-AUC evaluates ranking performance using the true-positive rate and false-positive rate across thresholds.

Precision-recall analysis focuses on precision and recall.

For highly imbalanced classification, PR-AUC or Average Precision is particularly informative because it focuses on the minority positive class.

## Q16. Why use a pipeline?

A pipeline combines preprocessing and model training into one reproducible workflow.

It also helps prevent data leakage when transformations are fitted separately within each training fold during cross-validation.

## Q17. What is the difference between fit and transform?

`fit()` learns parameters from the data.

`transform()` applies the learned parameters.

For example, a StandardScaler learns the mean and standard deviation during fitting and uses them during transformation.

## Q18. What is `ColumnTransformer`?

ColumnTransformer applies different transformations to different groups of columns.

For example:

- Numerical columns: imputation and scaling.
- Categorical columns: imputation and one-hot encoding.

## Q19. What is the purpose of random_state?

It controls the randomness of operations such as train/test splitting.

This helps make experiments reproducible.

## Q20. What is the purpose of model monitoring?

A deployed model can lose performance over time because customer behaviour and fraud patterns change.

Monitoring helps detect data drift, prediction changes, failures and performance degradation.

---

# 11. Concepts You Should Be Able to Explain Without Looking at Notes

Before moving forward, make sure you can explain these in your own words:

1. What is data ingestion?
2. What is SQLAlchemy?
3. What is a Pandas DataFrame?
4. What is EDA?
5. What is the difference between mean and median?
6. What are missing values?
7. What are outliers?
8. What is IQR?
9. What is correlation?
10. What is data cleaning?
11. What is data validation?
12. What is feature engineering?
13. What are independent and dependent variables?
14. What is class imbalance?
15. What is stratified sampling?
16. What is data leakage?
17. What is overfitting?
18. What is underfitting?
19. What is train/test splitting?
20. What is reproducibility?

---

# 12. What Comes Next?

Our next stage is:

## Step 5 — Data Preprocessing using Scikit-learn

We will implement:

1. Numerical feature identification.
2. Categorical feature identification.
3. Missing-value imputation.
4. One-hot encoding.
5. Feature scaling.
6. `SimpleImputer`.
7. `StandardScaler`.
8. `OneHotEncoder`.
9. `ColumnTransformer`.
10. `Pipeline`.

We will fit all preprocessing operations using training data only.

After preprocessing, we will move into model training with Logistic Regression, Decision Tree, Random Forest and other suitable algorithms.

The final objective is to build a reproducible, maintainable and deployable fraud classification system.

---

## Final interview takeaway

The most important thing to communicate in an ML Engineer interview is not simply that you know how to call `RandomForestClassifier()`.

You should be able to explain the complete lifecycle:

**Where the data comes from → How you validate it → How you understand it → How you engineer features → How you prevent leakage → How you preprocess it → How you train and evaluate models → How you deploy and monitor them.**

That is the engineering thinking behind an end-to-end machine learning project.
