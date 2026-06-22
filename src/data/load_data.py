import pandas as pd
from src.utils.config import RAW_DATA_FILE


def load_raw_data(file_path=RAW_DATA_FILE):
   
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")
        return df

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}\n"
            "Please download the PaySim dataset, rename it to fraud.csv, "
            "and place it inside data/raw/"
        )


def basic_data_check(df):
   
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nDataset information:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    if "isFraud" in df.columns:
        print("\nTarget distribution:")
        print(df["isFraud"].value_counts())

        print("\nTarget distribution percentage:")
        print(df["isFraud"].value_counts(normalize=True) * 100)
    else:
        print("\nWarning: Target column 'isFraud' not found.")


if __name__ == "__main__":
    data = load_raw_data()
    basic_data_check(data)