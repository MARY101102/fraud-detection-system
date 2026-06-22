from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import load_raw_data
from src.utils.config import PROCESSED_DATA_DIR


TARGET_COLUMN = "is_fraud"

COLUMNS_TO_RENAME = {
    "nameOrig": "name_orig",
    "oldbalanceOrg": "old_balance_orig",
    "newbalanceOrig": "new_balance_orig",
    "nameDest": "name_dest",
    "oldbalanceDest": "old_balance_dest",
    "newbalanceDest": "new_balance_dest",
    "isFraud": "is_fraud",
    "isFlaggedFraud": "is_flagged_fraud",
}

COLUMNS_TO_DROP = [
    "name_orig",
    "name_dest",
    "is_flagged_fraud",
]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename dataset columns into clean snake_case names.
    """
    df = df.copy()
    df = df.rename(columns=COLUMNS_TO_RENAME)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows if any exist.
    """
    df = df.copy()
    before_count = len(df)
    df = df.drop_duplicates()
    after_count = len(df)

    removed_count = before_count - after_count
    print(f"Duplicate rows removed: {removed_count}")

    return df


def validate_target_column(df: pd.DataFrame) -> None:
    """
    Validate that the target column exists.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")


def prepare_features_and_target(df: pd.DataFrame):
    """
    Separate features X and target y.
    Drop unnecessary or risky columns for Version 1.
    """
    df = df.copy()

    validate_target_column(df)

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    existing_drop_columns = [col for col in COLUMNS_TO_DROP if col in X.columns]
    X = X.drop(columns=existing_drop_columns)

    print("Dropped columns:", existing_drop_columns)
    print("Feature columns:", X.columns.tolist())
    print("Target column:", TARGET_COLUMN)

    return X, y


def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Create train/test split.

    Stratify is used to keep fraud/non-fraud ratio similar in train and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Create preprocessing pipeline for numerical and categorical columns.

    Numerical columns:
        StandardScaler

    Categorical columns:
        OneHotEncoder
    """
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print("Categorical columns:", categorical_columns)
    print("Numerical columns:", numerical_columns)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def save_processed_data(X_train, X_test, y_train, y_test) -> None:
    """
    Save train/test split files into data/processed.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    print("Processed train/test files saved successfully.")


def save_preprocessor(preprocessor: ColumnTransformer) -> None:
    """
    Save preprocessing pipeline.
    """
    output_path = PROCESSED_DATA_DIR / "preprocessor.pkl"
    joblib.dump(preprocessor, output_path)

    print(f"Preprocessor saved to: {output_path}")


def run_preprocessing_pipeline():
    """
    Full preprocessing pipeline.
    """
    print("Loading raw data...")
    df = load_raw_data()

    print("\nCleaning column names...")
    df = clean_column_names(df)

    print("\nRemoving duplicates...")
    df = remove_duplicates(df)

    print("\nPreparing features and target...")
    X, y = prepare_features_and_target(df)

    print("\nCreating train/test split...")
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    print("\nBuilding preprocessor...")
    preprocessor = build_preprocessor(X_train)

    print("\nFitting preprocessor on training data only...")
    preprocessor.fit(X_train)

    print("\nSaving processed data...")
    save_processed_data(X_train, X_test, y_train, y_test)

    print("\nSaving preprocessor...")
    save_preprocessor(preprocessor)

    print("\nPreprocessing completed successfully.")

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    run_preprocessing_pipeline()