import numpy as np
import pandas as pd


def safe_divide(numerator, denominator):
    """
    Safely divide two values/Series.

    If denominator is 0, return 0 instead of infinity.
    """
    return np.where(denominator == 0, 0, numerator / denominator)


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create transaction amount-based features.
    """
    df = df.copy()

    df["amount_log"] = np.log1p(df["amount"])
    df["amount_is_zero"] = (df["amount"] == 0).astype(int)

    # 95th percentile as a simple high amount threshold
    high_amount_threshold = df["amount"].quantile(0.95)
    df["amount_is_high"] = (df["amount"] > high_amount_threshold).astype(int)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from PaySim step column.

    In PaySim, step usually represents an hour in the simulation.
    """
    df = df.copy()

    df["hour"] = df["step"] % 24
    df["day"] = df["step"] // 24

    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    # Simulated weekend-style feature based on day number
    df["day_of_week"] = df["day"] % 7
    df["is_weekend_simulated"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def add_transaction_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary flags from transaction type.
    """
    df = df.copy()

    df["is_cash_out"] = (df["type"] == "CASH_OUT").astype(int)
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    df["is_payment"] = (df["type"] == "PAYMENT").astype(int)
    df["is_debit"] = (df["type"] == "DEBIT").astype(int)
    df["is_cash_in"] = (df["type"] == "CASH_IN").astype(int)

    return df


def add_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create balance-based features.

    Note:
        Some balance features may use post-transaction values.
        This is useful for analysis/model comparison, but should be discussed
        as a possible real-time leakage concern in the report.
    """
    df = df.copy()

    df["origin_balance_diff"] = (
        df["old_balance_orig"] - df["new_balance_orig"]
    )

    df["destination_balance_diff"] = (
        df["new_balance_dest"] - df["old_balance_dest"]
    )

    df["origin_balance_error"] = (
        df["old_balance_orig"] - df["amount"] - df["new_balance_orig"]
    )

    df["destination_balance_error"] = (
        df["old_balance_dest"] + df["amount"] - df["new_balance_dest"]
    )

    df["origin_balance_ratio"] = safe_divide(
        df["amount"],
        df["old_balance_orig"]
    )

    df["destination_balance_ratio"] = safe_divide(
        df["amount"],
        df["old_balance_dest"]
    )

    df["origin_old_balance_is_zero"] = (
        df["old_balance_orig"] == 0
    ).astype(int)

    df["destination_old_balance_is_zero"] = (
        df["old_balance_dest"] == 0
    ).astype(int)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering function.

    This function receives a cleaned dataframe and returns
    a dataframe with additional engineered features.
    """
    df = df.copy()

    required_columns = [
        "step",
        "type",
        "amount",
        "old_balance_orig",
        "new_balance_orig",
        "old_balance_dest",
        "new_balance_dest",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns for feature engineering: {missing_columns}")

    df = add_amount_features(df)
    df = add_time_features(df)
    df = add_transaction_type_features(df)
    df = add_balance_features(df)

    print("Feature engineering completed.")
    print(f"Total columns after feature engineering: {df.shape[1]}")

    return df


if __name__ == "__main__":
    from src.data.load_data import load_raw_data
    from src.data.preprocess import clean_column_names, remove_duplicates

    raw_df = load_raw_data()
    clean_df = clean_column_names(raw_df)
    clean_df = remove_duplicates(clean_df)

    feature_df = build_features(clean_df)

    print(feature_df.head())
    print(feature_df.columns.tolist())