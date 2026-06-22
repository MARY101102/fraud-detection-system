# Fraud Detection System — Version 1 Progress Notes

Project title:

**End-to-End Fraud Detection System with Classical ML, Graph Neural Networks, and MLOps Deployment**

Current version:

**Version 1 — Classical ML Fraud Detection**

---

## 1. Project Purpose

This project is being built as a CV and interview-ready machine learning project.

The full project will eventually include:

- Classical ML fraud detection
- Graph Neural Network fraud detection
- FastAPI prediction API
- Mock bank transaction sender
- PostgreSQL prediction logs
- Fraud analyst dashboard
- Docker setup
- CI/CD
- MLflow tracking
- Monitoring and drift detection

Version 1 focuses only on the classical ML foundation.

---

## 2. Version 1 Scope

Version 1 includes:

- Exploratory Data Analysis
- Data preprocessing
- Feature engineering
- Imbalanced dataset handling
- Classical ML model training
- XGBoost / CatBoost modeling
- Proper model evaluation
- SHAP explainability
- Evaluation report
- Saved best model

---

## 3. Project Workflow

The Version 1 workflow is:

```text
Raw fraud dataset
        ↓
Load data
        ↓
EDA
        ↓
Data cleaning
        ↓
Preprocessing
        ↓
Feature engineering
        ↓
Handle class imbalance
        ↓
Train baseline models
        ↓
Train XGBoost / CatBoost
        ↓
Evaluate models
        ↓
Tune threshold
        ↓
Explain with SHAP
        ↓
Save best model
        ↓
Write evaluation report
```

---

## 4. Step 1 — Project Setup

Completed setup:

- Created main project folder: `fraud-detection-system`
- Created clean ML project structure
- Created virtual environment
- Added dependencies in `requirements.txt`
- Added `.gitignore`
- Added `README.md`
- Added basic source-code folders

Important folders:

```text
data/raw/              original dataset, not pushed to GitHub
data/processed/        cleaned/processed data, not pushed to GitHub
notebooks/             EDA and experiments
src/data/              data loading and preprocessing
src/features/          feature engineering
src/models/            model training and prediction
src/evaluation/        evaluation logic
src/utils/             shared config/helper code
models/trained/        saved models, not pushed to GitHub
reports/               reports and generated outputs
tests/                 test files
```

Why this matters:

A clean folder structure makes the project look professional, reproducible, and ready for later API/MLOps work.

---

## 5. Step 2 — Dataset Selection and Loading

Recommended dataset:

**PaySim Synthetic Financial Dataset**

Reason:

- It has transaction-style fraud data.
- It contains origin and destination account IDs.
- It has a fraud label: `isFraud`.
- It is useful later for GNN because relationships can be created using account IDs.
- It is synthetic, so it is suitable for a public GitHub/CV project.

Dataset file should be placed here:

```text
data/raw/fraud.csv
```

Important:

Do not push the dataset to GitHub.

The dataset should stay local because it may be large and should be downloaded separately by anyone who wants to run the project.

Expected columns:

```text
step
type
amount
nameOrig
oldbalanceOrg
newbalanceOrig
nameDest
oldbalanceDest
newbalanceDest
isFraud
isFlaggedFraud
```

Target column:

```text
isFraud
```

Target meaning:

```text
0 = normal transaction
1 = fraudulent transaction
```

---

## 6. Step 3 — Exploratory Data Analysis

EDA means exploring the dataset before model training.

EDA checks:

- Dataset shape
- Column names
- Data types
- Missing values
- Duplicate rows
- Target distribution
- Class imbalance ratio
- Transaction type distribution
- Fraud by transaction type
- Amount distribution
- Fraud vs non-fraud amount behavior
- Fraud over time steps
- Balance column behavior
- `isFlaggedFraud` behavior
- Origin/destination account patterns
- Correlation with target

Important insight:

Fraud detection datasets are usually highly imbalanced. Therefore, accuracy alone is not a good evaluation metric.

Better metrics:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix

---

## 7. Important Concepts to Revise

### Fraud Detection

Fraud detection is a binary classification problem.

The model predicts whether a transaction is normal or fraudulent.

---

### Class Imbalance

Fraud cases are usually much fewer than normal transactions.

Example:

```text
Normal transactions = 99%
Fraud transactions = 1%
```

A model can get high accuracy by predicting everything as normal, but that model is useless.

---

### Precision

Precision answers:

```text
Out of all transactions predicted as fraud, how many were actually fraud?
```

High precision means fewer false alarms.

---

### Recall

Recall answers:

```text
Out of all actual fraud transactions, how many did the model catch?
```

High recall is very important in fraud detection because missed fraud is costly.

---

### F1-score

F1-score balances precision and recall.

---

### ROC-AUC

ROC-AUC measures how well the model separates fraud and non-fraud classes.

---

### PR-AUC

PR-AUC is especially useful for imbalanced datasets.

It is often more meaningful than ROC-AUC in fraud detection.

---

### Data Leakage

Data leakage happens when the model uses information that would not be available during real prediction time.

Examples:

- Using future data during training
- Applying preprocessing before train/test split
- Using test data during feature engineering
- Using post-transaction values that may not be available before fraud decision

For PaySim, balance-related columns should be handled carefully.

---

## 8. GitHub Push Checklist

Before pushing to GitHub:

```text
[ ] Dataset is not committed
[ ] Virtual environment is not committed
[ ] Trained models are not committed
[ ] Large processed data files are not committed
[ ] .gitignore is added
[ ] README.md is updated
[ ] requirements.txt is added
[ ] Project notes file is added
[ ] Code runs without import errors
```

---

## 9. Git Commands Used

Initialize Git:

```bash
git init
```

Check files:

```bash
git status
```

Stage files:

```bash
git add .
```

Commit files:

```bash
git commit -m "Initial project setup for fraud detection system"
```

Connect remote repository:

```bash
git remote add origin https://github.com/YOUR_USERNAME/fraud-detection-system.git
```

Rename branch to main:

```bash
git branch -M main
```

Push to GitHub:

```bash
git push -u origin main
```

---

## 10. Interview Explanation

A good explanation:

```text
I started the project by creating a production-style ML repository structure with separate folders for raw data, processed data, notebooks, source code, trained models, reports, and tests. For Version 1, I focused on classical machine learning fraud detection using a synthetic transaction dataset. I began with data loading and exploratory data analysis to understand class imbalance, fraud distribution, transaction types, amount patterns, and potential data leakage risks before moving to preprocessing and model training.
```

---

## 11. Current Status

Completed/planned so far:

```text
[ ] Step 1 — Project setup
[ ] Step 2 — Dataset selection and loading
[ ] Step 3 — EDA
[ ] Step 4 — Data cleaning and preprocessing
[ ] Step 5 — Feature engineering
[ ] Step 6 — Imbalance handling
[ ] Step 7 — Model training
[ ] Step 8 — Model evaluation
[ ] Step 9 — Threshold tuning
[ ] Step 10 — SHAP explainability
[ ] Step 11 — Save best model
[ ] Step 12 — Evaluation report
[ ] Step 13 — README final update
```

Update this checklist as the project progresses.
