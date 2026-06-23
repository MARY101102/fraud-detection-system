import json
from pathlib import Path

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from src.utils.config import PROCESSED_DATA_DIR, REPORTS_DIR


RANDOM_STATE = 42


def load_processed_train_test_data():
    """
    Load processed train/test data created during preprocessing.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    X_test_path = PROCESSED_DATA_DIR / "X_test.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
    y_test_path = PROCESSED_DATA_DIR / "y_test.csv"

    required_files = [
        X_train_path,
        X_test_path,
        y_train_path,
        y_test_path,
    ]

    missing_files = [str(path) for path in required_files if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "Processed files not found. Run preprocessing first:\n"
            "python -m src.data.preprocess\n\n"
            f"Missing files: {missing_files}"
        )

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).squeeze("columns")
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    return X_train, X_test, y_train, y_test


def calculate_target_distribution(y: pd.Series) -> dict:
    """
    Calculate class counts and percentages.
    """
    counts = y.value_counts().sort_index()
    percentages = y.value_counts(normalize=True).sort_index() * 100

    return {
        "class_counts": {
            str(class_label): int(count)
            for class_label, count in counts.items()
        },
        "class_percentages": {
            str(class_label): float(percentage)
            for class_label, percentage in percentages.items()
        },
    }


def calculate_class_weights(y_train: pd.Series) -> dict:
    """
    Calculate class weights for models like Logistic Regression and Random Forest.
    """
    classes = np.array(sorted(y_train.unique()))

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    return {
        int(class_label): float(weight)
        for class_label, weight in zip(classes, weights)
    }


def calculate_xgboost_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Calculate scale_pos_weight for XGBoost.

    Formula:
        number of negative samples / number of positive samples
    """
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    if positive_count == 0:
        raise ValueError("No positive fraud samples found in y_train.")

    return float(negative_count / positive_count)


def create_random_undersampled_training_set(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy=0.2,
    random_state=RANDOM_STATE,
):
    """
    Create an undersampled training set.

    sampling_strategy=0.2 means:
        minority / majority = 0.2 after resampling

    Example:
        If fraud count = 8,000,
        majority normal count becomes around 40,000.
    """
    undersampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )

    X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

    return X_train_under, y_train_under


def save_undersampled_data(X_train_under, y_train_under):
    """
    Save undersampled training data into data/processed.
    """
    X_train_under.to_csv(PROCESSED_DATA_DIR / "X_train_under.csv", index=False)
    y_train_under.to_csv(PROCESSED_DATA_DIR / "y_train_under.csv", index=False)

    print("Undersampled training data saved:")
    print(PROCESSED_DATA_DIR / "X_train_under.csv")
    print(PROCESSED_DATA_DIR / "y_train_under.csv")


def save_imbalance_report(report: dict):
    """
    Save imbalance report as JSON.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / "imbalance_report.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"Imbalance report saved to: {output_path}")


def run_imbalance_preparation(create_undersampled_data=True):
    """
    Main imbalance handling preparation function.
    """
    print("Loading processed train/test data...")
    X_train, X_test, y_train, y_test = load_processed_train_test_data()

    print("\nTrain target distribution:")
    train_distribution = calculate_target_distribution(y_train)
    print(train_distribution)

    print("\nTest target distribution:")
    test_distribution = calculate_target_distribution(y_test)
    print(test_distribution)

    print("\nCalculating class weights...")
    class_weights = calculate_class_weights(y_train)
    print("Class weights:", class_weights)

    print("\nCalculating XGBoost scale_pos_weight...")
    scale_pos_weight = calculate_xgboost_scale_pos_weight(y_train)
    print("scale_pos_weight:", scale_pos_weight)

    report = {
        "train_distribution": train_distribution,
        "test_distribution": test_distribution,
        "class_weights_for_sklearn": class_weights,
        "xgboost_scale_pos_weight": scale_pos_weight,
        "catboost_class_weights": [
            class_weights.get(0),
            class_weights.get(1),
        ],
        "notes": [
            "The dataset is highly imbalanced.",
            "Accuracy should not be used as the main metric.",
            "Use recall, precision, F1-score, ROC-AUC, PR-AUC, and confusion matrix.",
            "Resampling must be applied only to training data, never test data.",
        ],
    }

    if create_undersampled_data:
        print("\nCreating random undersampled training set...")
        X_train_under, y_train_under = create_random_undersampled_training_set(
            X_train,
            y_train,
            sampling_strategy=0.2,
        )

        print("Original training shape:", X_train.shape)
        print("Undersampled training shape:", X_train_under.shape)

        print("\nUndersampled target distribution:")
        undersampled_distribution = calculate_target_distribution(y_train_under)
        print(undersampled_distribution)

        save_undersampled_data(X_train_under, y_train_under)

        report["undersampled_train_distribution"] = undersampled_distribution
        report["undersampling_strategy"] = "RandomUnderSampler with sampling_strategy=0.2"

    save_imbalance_report(report)

    print("\nImbalance preparation completed successfully.")

    return report


if __name__ == "__main__":
    run_imbalance_preparation()