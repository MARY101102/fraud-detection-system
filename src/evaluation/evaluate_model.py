import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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
    Load the best model saved during model training.
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
    Load training summary created in Step 7.
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


def load_model_comparison():
    """
    Load model comparison table created during training.
    """
    comparison_path = REPORTS_DIR / "model_comparison.csv"

    if not comparison_path.exists():
        raise FileNotFoundError(
            "model_comparison.csv not found. Run model training first:\n"
            "python -m src.models.train_model"
        )

    return pd.read_csv(comparison_path)


def evaluate_predictions(y_test, y_pred, y_proba):
    """
    Calculate evaluation metrics for model predictions.
    """
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    return metrics


def plot_confusion_matrix(y_test, y_pred, best_model_name):
    """
    Save confusion matrix plot for the best model.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Non-Fraud", "Fraud"],
    )

    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()

    output_path = FIGURES_DIR / "confusion_matrix_best_model.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved confusion matrix: {output_path}")


def plot_roc_curve(y_test, y_proba, best_model_name):
    """
    Save ROC curve plot for the best model.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    RocCurveDisplay.from_predictions(y_test, y_proba)

    plt.title(f"ROC Curve - {best_model_name}")
    plt.tight_layout()

    output_path = FIGURES_DIR / "roc_curve_best_model.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved ROC curve: {output_path}")


def plot_precision_recall_curve(y_test, y_proba, best_model_name):
    """
    Save precision-recall curve plot for the best model.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    PrecisionRecallDisplay.from_predictions(y_test, y_proba)

    plt.title(f"Precision-Recall Curve - {best_model_name}")
    plt.tight_layout()

    output_path = FIGURES_DIR / "precision_recall_curve_best_model.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved precision-recall curve: {output_path}")


def plot_model_comparison(model_comparison):
    """
    Save model comparison plots.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sorted_by_pr_auc = model_comparison.sort_values(by="pr_auc", ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(sorted_by_pr_auc["model_name"], sorted_by_pr_auc["pr_auc"])
    plt.title("Model Comparison by PR-AUC")
    plt.xlabel("Model")
    plt.ylabel("PR-AUC")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = FIGURES_DIR / "model_comparison_pr_auc.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved PR-AUC comparison plot: {output_path}")

    sorted_by_recall = model_comparison.sort_values(by="recall", ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(sorted_by_recall["model_name"], sorted_by_recall["recall"])
    plt.title("Model Comparison by Recall")
    plt.xlabel("Model")
    plt.ylabel("Recall")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = FIGURES_DIR / "model_comparison_recall.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved recall comparison plot: {output_path}")


def create_evaluation_report(best_model_name, metrics, model_comparison):
    """
    Create Markdown evaluation report.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    best_model_row = model_comparison[
        model_comparison["model_name"] == best_model_name
    ]

    if not best_model_row.empty:
        best_model_row_markdown = best_model_row.to_markdown(index=False)
    else:
        best_model_row_markdown = "Best model row not found in model comparison table."

    model_comparison_markdown = model_comparison.sort_values(
        by="pr_auc",
        ascending=False,
    ).to_markdown(index=False)

    report_lines = [
        "# Model Evaluation Report",
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
        "## Evaluation Goal",
        "",
        "The goal of this evaluation is to compare classical machine learning models for fraud detection and select the best model using fraud-focused metrics.",
        "",
        "Because the dataset is highly imbalanced, accuracy is not used as the main metric.",
        "",
        "---",
        "",
        "## Best Model",
        "",
        "Selected best model:",
        "",
        f"`{best_model_name}`",
        "",
        "The model was selected based mainly on **PR-AUC**, because PR-AUC is more meaningful for imbalanced fraud detection problems.",
        "",
        "---",
        "",
        "## Best Model Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Precision | {metrics['precision']:.6f} |",
        f"| Recall | {metrics['recall']:.6f} |",
        f"| F1-score | {metrics['f1_score']:.6f} |",
        f"| ROC-AUC | {metrics['roc_auc']:.6f} |",
        f"| PR-AUC | {metrics['pr_auc']:.6f} |",
        "",
        "---",
        "",
        "## Confusion Matrix Values",
        "",
        "| Value | Count |",
        "|---|---:|",
        f"| True Negatives | {metrics['true_negatives']} |",
        f"| False Positives | {metrics['false_positives']} |",
        f"| False Negatives | {metrics['false_negatives']} |",
        f"| True Positives | {metrics['true_positives']} |",
        "",
        "---",
        "",
        "## Best Model Row From Comparison Table",
        "",
        best_model_row_markdown,
        "",
        "---",
        "",
        "## Full Model Comparison",
        "",
        model_comparison_markdown,
        "",
        "---",
        "",
        "## Generated Figures",
        "",
        "The following plots were generated:",
        "",
        "- `reports/figures/confusion_matrix_best_model.png`",
        "- `reports/figures/roc_curve_best_model.png`",
        "- `reports/figures/precision_recall_curve_best_model.png`",
        "- `reports/figures/model_comparison_pr_auc.png`",
        "- `reports/figures/model_comparison_recall.png`",
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "The model evaluation focuses on fraud-specific metrics:",
        "",
        "- **Recall** shows how many actual fraud transactions were detected.",
        "- **Precision** shows how many predicted fraud cases were actually fraud.",
        "- **F1-score** balances precision and recall.",
        "- **ROC-AUC** measures general class separation.",
        "- **PR-AUC** is especially important because fraud data is highly imbalanced.",
        "",
        "In fraud detection, false negatives are risky because fraud transactions are missed.",
        "False positives are also important because they create unnecessary manual review and may disturb real customers.",
        "",
        "---",
        "",
        "## Important Note",
        "",
        "This evaluation uses the default classification threshold of `0.5`.",
        "",
        "The next step is threshold tuning, where different probability thresholds will be tested to find a better fraud decision boundary.",
        "",
        "---",
        "",
        "## Next Step",
        "",
        "Step 9 — Threshold Tuning",
        "",
    ]

    report = "\n".join(report_lines)

    output_path = REPORTS_DIR / "evaluation_report.md"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report)

    print(f"Evaluation report saved to: {output_path}")


def run_evaluation_report():
    """
    Main evaluation report pipeline.
    """
    print("Loading train/test data...")
    _, X_test, _, y_test = load_processed_train_test_data()

    print("Loading preprocessor...")
    preprocessor = load_preprocessor()

    print("Loading best model...")
    best_model = load_best_model()

    print("Loading training summary...")
    training_summary = load_training_summary()
    best_model_name = training_summary["best_model"]

    print("Loading model comparison...")
    model_comparison = load_model_comparison()

    print("Transforming test data...")
    X_test_transformed = preprocessor.transform(X_test)

    print("Generating predictions...")
    y_pred = best_model.predict(X_test_transformed)

    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test_transformed)[:, 1]
    else:
        y_proba = y_pred

    print("Calculating metrics...")
    metrics = evaluate_predictions(y_test, y_pred, y_proba)

    print("Creating plots...")
    plot_confusion_matrix(y_test, y_pred, best_model_name)
    plot_roc_curve(y_test, y_proba, best_model_name)
    plot_precision_recall_curve(y_test, y_proba, best_model_name)
    plot_model_comparison(model_comparison)

    print("Creating evaluation report...")
    create_evaluation_report(best_model_name, metrics, model_comparison)

    print("\nEvaluation report completed successfully.")

    return metrics, model_comparison


if __name__ == "__main__":
    run_evaluation_report()