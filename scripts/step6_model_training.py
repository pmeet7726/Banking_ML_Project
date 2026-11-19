
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from banking_ml.models.train_models import get_models


def main():

    print("\nStarting Step 6: Model Training")

    # --------------------------------------------------
    # 1. Define project paths
    # --------------------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    input_dir = project_root / "data/preprocessed"

    model_dir = project_root / "models/training"

    report_dir = project_root / "reports/model_training"

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 2. Load preprocessed datasets
    # --------------------------------------------------

    print("\nLoading preprocessed datasets...")

    X_train = np.load(
        input_dir / "X_train_processed.npy"
    )

    X_test = np.load(
        input_dir / "X_test_processed.npy"
    )

    y_train = pd.read_csv(
        input_dir / "y_train.csv"
    ).values.ravel()

    y_test = pd.read_csv(
        input_dir / "y_test.csv"
    ).values.ravel()

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("Training fraud cases:", (y_train == 1).sum())
    print("Testing fraud cases:", (y_test == 1).sum())

    # --------------------------------------------------
    # 3. Define cross-validation strategy
    # --------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {

        "accuracy": "accuracy",

        "precision": make_scorer(
            precision_score,
            zero_division=0
        ),

        "recall": make_scorer(
            recall_score,
            zero_division=0
        ),

        "f1": make_scorer(
            f1_score,
            zero_division=0
        ),

        "average_precision": "average_precision",

        "roc_auc": "roc_auc"
    }

    # --------------------------------------------------
    # 4. Get models
    # --------------------------------------------------

    models = get_models()

    results = []

    print("\nStarting 5-Fold Cross-Validation...")

    # --------------------------------------------------
    # 5. Train and evaluate each model using CV
    # --------------------------------------------------

    for model_name, model in models.items():

        print(f"\nTraining: {model_name}")

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        result = {
            "model": model_name
        }

        for metric in scoring:

            values = scores[f"test_{metric}"]

            result[f"{metric}_mean"] = values.mean()

            result[f"{metric}_std"] = values.std()

        results.append(result)

        print(
            "Average Precision:",
            round(result["average_precision_mean"], 4)
        )

        print(
            "Recall:",
            round(result["recall_mean"], 4)
        )

        print(
            "F1 Score:",
            round(result["f1_mean"], 4)
        )

    # --------------------------------------------------
    # 6. Save cross-validation results
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="average_precision_mean",
        ascending=False
    )

    results_df.to_csv(
        report_dir / "cross_validation_results.csv",
        index=False
    )

    print("\n========== CROSS-VALIDATION RESULTS ==========")

    print(
        results_df[
            [
                "model",
                "accuracy_mean",
                "precision_mean",
                "recall_mean",
                "f1_mean",
                "average_precision_mean",
                "roc_auc_mean"
            ]
        ].round(4).to_string(index=False)
    )

    # --------------------------------------------------
    # 7. Select the best model using CV Average Precision
    # --------------------------------------------------

    best_model_name = results_df.iloc[0]["model"]

    best_model = models[best_model_name]

    print("\nBest model:", best_model_name)

    # --------------------------------------------------
    # 8. Fit the selected model on all training data
    # --------------------------------------------------

    print("\nFitting the selected model on full training data...")

    best_model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # 9. Evaluate the selected model on test data
    # --------------------------------------------------

    print("\nEvaluating selected model on test data...")

    y_test_pred = best_model.predict(X_test)

    y_test_proba = best_model.predict_proba(X_test)[:, 1]

    test_metrics = {

        "model": best_model_name,

        "accuracy": accuracy_score(
            y_test,
            y_test_pred
        ),

        "precision": precision_score(
            y_test,
            y_test_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            y_test_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            y_test_pred,
            zero_division=0
        ),

        "average_precision": average_precision_score(
            y_test,
            y_test_proba
        ),

        "roc_auc": roc_auc_score(
            y_test,
            y_test_proba
        )
    }

    metrics_df = pd.DataFrame([test_metrics])

    metrics_df.to_csv(
        report_dir / "test_metrics.csv",
        index=False
    )

    print("\n========== FINAL TEST RESULTS ==========")

    for metric, value in test_metrics.items():

        if metric != "model":

            print(f"{metric}: {value:.4f}")

    # --------------------------------------------------
    # 10. Confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_test_pred
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"]
    )

    cm_df.to_csv(
        report_dir / "confusion_matrix.csv"
    )

    print("\n========== CONFUSION MATRIX ==========")

    print(cm_df)

    # --------------------------------------------------
    # 11. Classification report
    # --------------------------------------------------

    report = classification_report(
        y_test,
        y_test_pred,
        zero_division=0
    )

    print("\n========== CLASSIFICATION REPORT ==========")

    print(report)

    with open(
        report_dir / "classification_report.txt",
        "w"
    ) as file:

        file.write(report)

    # --------------------------------------------------
    # 12. Save test predictions
    # --------------------------------------------------

    predictions_df = pd.DataFrame({

        "actual_fraud_flag": y_test,

        "predicted_fraud_flag": y_test_pred,

        "fraud_probability": y_test_proba

    })

    predictions_df.to_csv(
        report_dir / "test_predictions.csv",
        index=False
    )

    # --------------------------------------------------
    # 13. Save trained model
    # --------------------------------------------------

    joblib.dump(
        best_model,
        model_dir / "best_model.joblib"
    )

    print("\nSaved best model:", model_dir / "best_model.joblib")

    print("\nStep 6 completed successfully!")


if __name__ == "__main__":

    main()
