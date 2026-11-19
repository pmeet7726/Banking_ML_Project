
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)


def calculate_metrics(y_true, y_pred, y_probability):

    return {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),

        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),

        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),

        "f2": fbeta_score(
            y_true, y_pred, beta=2, zero_division=0
        ),

        "average_precision": average_precision_score(
            y_true, y_probability
        ),

        "roc_auc": roc_auc_score(
            y_true, y_probability
        )
    }


def main():

    print("\nStarting Step 7: Threshold Optimization")

    # --------------------------------------------------
    # 1. Define paths
    # --------------------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    input_dir = project_root / "data/preprocessed"

    model_dir = project_root / "models/training"

    report_dir = project_root / "reports/threshold_optimization"

    report_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 2. Load datasets and trained model
    # --------------------------------------------------

    print("\nLoading datasets and trained model...")

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

    model = joblib.load(
        model_dir / "best_model.joblib"
    )

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    # --------------------------------------------------
    # 3. Generate out-of-fold training probabilities
    # --------------------------------------------------

    print("\nGenerating out-of-fold probabilities...")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    oof_probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    print("OOF probability generation completed.")

    # --------------------------------------------------
    # 4. Find the best threshold using F2
    # --------------------------------------------------

    print("\nFinding the best threshold using F2...")

    precision, recall, thresholds = precision_recall_curve(
        y_train,
        oof_probabilities
    )

    # Precision and recall have one extra entry
    # compared with thresholds.

    precision = precision[:-1]
    recall = recall[:-1]

    f2_scores = (
        5 * precision * recall
    ) / (
        4 * precision + recall + 1e-10
    )

    best_index = np.argmax(f2_scores)

    best_threshold = float(thresholds[best_index])

    print("\nBest threshold:", round(best_threshold, 4))

    print(
        "OOF Precision:",
        round(precision[best_index], 4)
    )

    print(
        "OOF Recall:",
        round(recall[best_index], 4)
    )

    print(
        "OOF F2 Score:",
        round(f2_scores[best_index], 4)
    )

    # --------------------------------------------------
    # 5. Fit model on full training data
    # --------------------------------------------------

    print("\nFitting model on full training data...")

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # 6. Get test probabilities
    # --------------------------------------------------

    print("\nGenerating test probabilities...")

    y_test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # Default threshold
    y_test_default = (
        y_test_probabilities >= 0.50
    ).astype(int)

    # Optimized threshold
    y_test_optimized = (
        y_test_probabilities >= best_threshold
    ).astype(int)

    # --------------------------------------------------
    # 7. Compare both thresholds
    # --------------------------------------------------

    default_metrics = calculate_metrics(
        y_test,
        y_test_default,
        y_test_probabilities
    )

    optimized_metrics = calculate_metrics(
        y_test,
        y_test_optimized,
        y_test_probabilities
    )

    comparison_df = pd.DataFrame([
        {
            "threshold": 0.50,
            **default_metrics
        },
        {
            "threshold": best_threshold,
            **optimized_metrics
        }
    ])

    comparison_df.to_csv(
        report_dir / "threshold_comparison.csv",
        index=False
    )

    print("\n========== TEST THRESHOLD COMPARISON ==========")

    print(comparison_df.round(4).to_string(index=False))

    # --------------------------------------------------
    # 8. Save optimized confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_test_optimized
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"]
    )

    cm_df.to_csv(
        report_dir / "optimized_confusion_matrix.csv"
    )

    print("\n========== OPTIMIZED CONFUSION MATRIX ==========")

    print(cm_df)

    # --------------------------------------------------
    # 9. Save test predictions
    # --------------------------------------------------

    predictions_df = pd.DataFrame({

        "actual_fraud_flag": y_test,

        "fraud_probability": y_test_probabilities,

        "default_prediction": y_test_default,

        "optimized_prediction": y_test_optimized

    })

    predictions_df.to_csv(
        report_dir / "threshold_predictions.csv",
        index=False
    )

    # --------------------------------------------------
    # 10. Save selected threshold
    # --------------------------------------------------

    threshold_config = {
        "model": "Gradient Boosting",
        "threshold": best_threshold,
        "selection_metric": "F2",
        "selection_method": "5-fold out-of-fold training probabilities"
    }

    with open(
        report_dir / "threshold_config.json",
        "w"
    ) as file:

        json.dump(
            threshold_config,
            file,
            indent=4
        )

    # Save the refitted model
    joblib.dump(
        model,
        model_dir / "best_model.joblib"
    )

    print("\nSaved threshold configuration.")

    print("\nStep 7 completed successfully!")


if __name__ == "__main__":

    main()
