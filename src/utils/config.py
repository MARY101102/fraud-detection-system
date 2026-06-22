from pathlib import Path

# Project root folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data paths
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Model paths
TRAINED_MODEL_DIR = PROJECT_ROOT / "models" / "trained"

# Report paths
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# File names
RAW_DATA_FILE = RAW_DATA_DIR / "fraud.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_fraud.csv"
FEATURE_DATA_FILE = PROCESSED_DATA_DIR / "features_fraud.csv"