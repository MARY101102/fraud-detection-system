import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.models.imbalance import (
    calculate_class_weights,
    calculate_xgboost_scale_pos_weight,
    load_processed_train_test_data,
)
from src.utils.config import PROCESSED_DATA_DIR, REPORTS_DIR, TRAINED_MODEL_DIR


RANDOM_STATE = 42
TRAINING_SAMPLE_SIZE = 250_000


def load_preprocessor():
    """
    Load the preprocessing pipeline saved in Step 4/5.
    """
    preprocessor_path = PROCESSED_DATA_DIR / "preprocessor.pkl"

    if not preprocessor_path.exists():
        raise FileNotFoundError(
            "preprocessor.pkl not found. Run preprocessing first:\n"
            "python -m src.data.preprocess"
        )

    return joblib.load(preprocessor_path)


def load_undersampled_training_data():
    """
    Load undersampled training data created in Step 6.
    """
    X_under_path = PROCESSED_DATA_DIR / "X_train_under.csv"
    y_under_path = PROCESSED_DATA_DIR / "y_train_under.csv"

    if not X_under_path.exists() or not y_under_path.exists():
        raise FileNotFoundError(
            "Undersampled training files not found. Run imbalance preparation first:\n"
            "python -m src.models.imbalance"
        )

    X_train_under = pd.read_csv(X_under_path)
    y_train_under = pd.read_csv(y_under_path).squeeze("columns")

    return X_train_under, y_train_under


def create_training_sample(X_train, y_train, sample_size=TRAINING_SAMPLE_SIZE):
    """
    Create a smaller stratified sample from the full training data.

    This keeps training faster while preserving the fraud/non-fraud ratio.
    """
    if len(X_train) <= sample_size:
        return X_train, y_train

    _, X_sample, _, y_sample = train_test_split(
        X_train,
        y_train,
        test_size=sample_size,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    return X_sample, y_sample


def transform_data(preprocessor, X_train, X_test):
    """
    Apply the saved preprocessing pipeline.

    The preprocessor was already fitted on original X_train during preprocessing.
    """
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    return X_train_transformed, X_test_transformed


def build_models(y_train):
    """
    Build model configurations.

    Some models use class weights because fraud data is imbalanced.
    """
    class_weights = calculate_class_weights(y_train)
    scale_pos_weight = calculate_xgboost_scale_pos_weight(y_train)

    catboost_class_weights = [
        class_weights[0],
        class_weights[1],
    ]

    models = {
        "logistic_regression_balanced": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost_weighted": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "catboost_weighted": CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            loss_function="Logloss",
            eval_metric="AUC",
            class_weights=catboost_class_weights,
            random_seed=RANDOM_STATE,
            verbose=False,
        ),
        "xgboost_undersampled": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    return models


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model using fraud-focused metrics.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

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


def save_model(model, model_name):
    """
    Save a trained model locally.
    """
    TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = TRAINED_MODEL_DIR / f"{model_name}.pkl"
    joblib.dump(model, model_path)

    print(f"Saved model: {model_path}")


def save_training_summary(results_df, best_model_name):
    """
    Save training summary files.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_path = REPORTS_DIR / "model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)

    summary = {
        "best_model": best_model_name,
        "selection_metric": "pr_auc",
        "notes": [
            "Models were evaluated on the untouched original test set.",
            "PR-AUC is used as the main ranking metric because the dataset is highly imbalanced.",
            "Accuracy is not used as the primary metric for fraud detection.",
        ],
    }

    summary_path = REPORTS_DIR / "training_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Model comparison saved to: {comparison_path}")
    print(f"Training summary saved to: {summary_path}")


def train_and_evaluate_models():
    """
    Main model training pipeline.
    """
    print("Loading processed train/test data...")
    X_train, X_test, y_train, y_test = load_processed_train_test_data()

    print("Loading preprocessor...")
    preprocessor = load_preprocessor()

    print("\nCreating stratified training sample for weighted models...")
    X_train_sample, y_train_sample = create_training_sample(X_train, y_train)

    print("Training sample shape:", X_train_sample.shape)
    print("Training sample target distribution:")
    print(y_train_sample.value_counts(normalize=True) * 100)

    print("\nLoading undersampled training data...")
    X_train_under, y_train_under = load_undersampled_training_data()

    print("Undersampled training shape:", X_train_under.shape)
    print("Undersampled target distribution:")
    print(y_train_under.value_counts(normalize=True) * 100)

    print("\nTransforming test data...")
    X_test_transformed = preprocessor.transform(X_test)

    print("\nBuilding models...")
    models = build_models(y_train)

    results = []
    trained_models = {}

    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")

        if model_name == "xgboost_undersampled":
            X_train_current = X_train_under
            y_train_current = y_train_under
        else:
            X_train_current = X_train_sample
            y_train_current = y_train_sample

        X_train_transformed = preprocessor.transform(X_train_current)

        model.fit(X_train_transformed, y_train_current)

        print(f"Evaluating model: {model_name}")
        metrics = evaluate_model(model, X_test_transformed, y_test)

        result = {
            "model_name": model_name,
            "training_rows": len(X_train_current),
            **metrics,
        }

        results.append(result)
        trained_models[model_name] = model

        save_model(model, model_name)

        print("Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="pr_auc", ascending=False)

    best_model_name = results_df.iloc[0]["model_name"]
    best_model = trained_models[best_model_name]

    print(f"\nBest model based on PR-AUC: {best_model_name}")

    save_model(best_model, "best_model")
    save_training_summary(results_df, best_model_name)

    print("\nModel training completed successfully.")

    return results_df


if __name__ == "__main__":
    train_and_evaluate_models()