import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.imbalance import load_processed_train_test_data
from src.utils.config import FIGURES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, TRAINED_MODEL_DIR


def load_preprocessor():
    """
    Load saved preprocessing pipeline.
    """
    preprocessor_path = PROCESSED_DATA_DIR / "preprocessor.pkl"

    if not preprocessor_path.exists():
        raise FileNotFoundError(
            "preprocessor.pkl not found. Run preprocessing first:\n"
            "python -m src.data.preprocess"
        )

    return joblib.load(preprocessor_path)


def load_best_model():
    """
    Load best model saved during Step 7.
    """
    best_model_path = TRAINED_MODEL_DIR / "best_model.pkl"

    if not best_model_path.exists():
        raise FileNotFoundError(
            "best_model.pkl not found. Run model training first:\n"
            "python -m src.models.train_model"
        )

    return joblib.load(best_model_path)


def load_training_summary():
    """
    Load training summary to get best model name.
    """
    summary_path = REPORTS_DIR / "training_summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(
            "training_summary.json not found. Run model training first:\n"
            "python -m src.models.train_model"
        )

    with open(summary_path, "r", encoding="utf-8") as file:
        summary = json.load(file)

    return summary


def get_prediction_probabilities(model, X_test_transformed):
    """
    Get fraud probability from the trained model.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("The selected model does not support predict_proba().")

    return model.predict_proba(X_test_transformed)[:, 1]


def evaluate_single_threshold(y_true, y_proba, threshold):
    """
    Evaluate model predictions at a single threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
    }


def evaluate_thresholds(y_true, y_proba, thresholds=None):
    """
    Evaluate multiple thresholds.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 1.00, 0.05)

    results = []

    for threshold in thresholds:
        metrics = evaluate_single_threshold(y_true, y_proba, threshold)
        results.append(metrics)

    results_df = pd.DataFrame(results)

    return results_df


def select_best_threshold(results_df):
    """
    Select best threshold using F1-score.

    F1-score is used here as a balanced starting point.
    """
    best_row = results_df.sort_values(
        by=["f1_score", "recall", "precision"],
        ascending=False,
    ).iloc[0]

    return best_row


def plot_threshold_metrics(results_df):
    """
    Save threshold tuning plots.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(results_df["threshold"], results_df["precision"], marker="o", label="Precision")
    plt.plot(results_df["threshold"], results_df["recall"], marker="o", label="Recall")
    plt.plot(results_df["threshold"], results_df["f1_score"], marker="o", label="F1-score")
    plt.title("Precision, Recall, and F1-score by Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "threshold_precision_recall_f1.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved threshold metrics plot: {output_path}")


def plot_confusion_values_by_threshold(results_df):
    """
    Save confusion matrix values across thresholds.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(results_df["threshold"], results_df["false_positives"], marker="o", label="False Positives")
    plt.plot(results_df["threshold"], results_df["false_negatives"], marker="o", label="False Negatives")
    plt.plot(results_df["threshold"], results_df["true_positives"], marker="o", label="True Positives")
    plt.title("Confusion Matrix Values by Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "threshold_confusion_values.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved threshold confusion values plot: {output_path}")


def save_threshold_outputs(results_df, best_threshold_row, best_model_name):
    """
    Save threshold tuning results and summary.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results_path = REPORTS_DIR / "threshold_tuning_results.csv"
    results_df.to_csv(results_path, index=False)

    summary = {
        "best_model": best_model_name,
        "selection_metric": "f1_score",
        "selected_threshold": float(best_threshold_row["threshold"]),
        "precision": float(best_threshold_row["precision"]),
        "recall": float(best_threshold_row["recall"]),
        "f1_score": float(best_threshold_row["f1_score"]),
        "true_negatives": int(best_threshold_row["true_negatives"]),
        "false_positives": int(best_threshold_row["false_positives"]),
        "false_negatives": int(best_threshold_row["false_negatives"]),
        "true_positives": int(best_threshold_row["true_positives"]),
        "notes": [
            "The default threshold 0.5 is not always optimal for fraud detection.",
            "This selected threshold maximizes F1-score as a balanced starting point.",
            "In a real bank, the final threshold should be selected based on fraud cost, false positive cost, and analyst review capacity.",
        ],
    }

    summary_path = REPORTS_DIR / "threshold_tuning_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    # Save small metadata for future API use.
    model_metadata = {
        "model_name": best_model_name,
        "selected_threshold": float(best_threshold_row["threshold"]),
        "threshold_selection_metric": "f1_score",
    }

    metadata_path = TRAINED_MODEL_DIR / "model_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(model_metadata, file, indent=4)

    print(f"Threshold tuning results saved to: {results_path}")
    print(f"Threshold tuning summary saved to: {summary_path}")
    print(f"Model metadata saved to: {metadata_path}")


def create_threshold_report(results_df, best_threshold_row, best_model_name):
    """
    Create Markdown threshold tuning report.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    top_thresholds = results_df.sort_values(
        by="f1_score",
        ascending=False,
    ).head(10)

    top_thresholds_markdown = top_thresholds.to_markdown(index=False)

    report_lines = [
        "# Threshold Tuning Report",
        "",
        "## Project",
        "",
        "End-to-End Fraud Detection System",
        "",
        "## Version",
        "",
        "Version 1 — Classical Machine Learning Fraud Detection",
        "",
        "---",
        "",
        "## Goal",
        "",
        "The goal of threshold tuning is to find a better fraud decision threshold instead of using the default threshold of `0.5`.",
        "",
        "Fraud detection is highly imbalanced, so the default threshold may not give the best balance between fraud detection and false alarms.",
        "",
        "---",
        "",
        "## Best Model",
        "",
        f"`{best_model_name}`",
        "",
        "---",
        "",
        "## Selected Threshold",
        "",
        f"`{best_threshold_row['threshold']:.2f}`",
        "",
        "This threshold was selected based on the highest F1-score.",
        "",
        "---",
        "",
        "## Metrics at Selected Threshold",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Precision | {best_threshold_row['precision']:.6f} |",
        f"| Recall | {best_threshold_row['recall']:.6f} |",
        f"| F1-score | {best_threshold_row['f1_score']:.6f} |",
        f"| True Negatives | {int(best_threshold_row['true_negatives'])} |",
        f"| False Positives | {int(best_threshold_row['false_positives'])} |",
        f"| False Negatives | {int(best_threshold_row['false_negatives'])} |",
        f"| True Positives | {int(best_threshold_row['true_positives'])} |",
        "",
        "---",
        "",
        "## Top Thresholds by F1-score",
        "",
        top_thresholds_markdown,
        "",
        "---",
        "",
        "## Generated Figures",
        "",
        "- `reports/figures/threshold_precision_recall_f1.png`",
        "- `reports/figures/threshold_confusion_values.png`",
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "Lower thresholds usually increase recall, meaning more fraud cases are detected.",
        "",
        "However, lower thresholds can also increase false positives, meaning more legitimate transactions are flagged as fraud.",
        "",
        "Higher thresholds usually reduce false positives but may miss more fraud cases.",
        "",
        "The selected threshold is a balanced starting point. In a real banking system, the final threshold should be selected based on business cost, analyst review capacity, and customer impact.",
        "",
        "---",
        "",
        "## Next Step",
        "",
        "Step 10 — SHAP Explainability",
        "",
    ]

    output_path = REPORTS_DIR / "threshold_tuning_report.md"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines))

    print(f"Threshold tuning report saved to: {output_path}")


def run_threshold_tuning():
    """
    Main threshold tuning pipeline.
    """
    print("Loading processed test data...")
    _, X_test, _, y_test = load_processed_train_test_data()

    print("Loading preprocessor...")
    preprocessor = load_preprocessor()

    print("Loading best model...")
    best_model = load_best_model()

    print("Loading training summary...")
    training_summary = load_training_summary()
    best_model_name = training_summary["best_model"]

    print("Transforming test data...")
    X_test_transformed = preprocessor.transform(X_test)

    print("Getting fraud probabilities...")
    y_proba = get_prediction_probabilities(best_model, X_test_transformed)

    print("Evaluating thresholds...")
    thresholds = np.arange(0.05, 1.00, 0.05)
    results_df = evaluate_thresholds(y_test, y_proba, thresholds)

    print("Selecting best threshold...")
    best_threshold_row = select_best_threshold(results_df)

    print("\nBest threshold:")
    print(best_threshold_row)

    print("\nCreating plots...")
    plot_threshold_metrics(results_df)
    plot_confusion_values_by_threshold(results_df)

    print("\nSaving outputs...")
    save_threshold_outputs(results_df, best_threshold_row, best_model_name)

    print("\nCreating threshold report...")
    create_threshold_report(results_df, best_threshold_row, best_model_name)

    print("\nThreshold tuning completed successfully.")

    return results_df, best_threshold_row


if __name__ == "__main__":
    run_threshold_tuning()