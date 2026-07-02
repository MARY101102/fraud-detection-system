# End-to-End Fraud Detection System

## Project Overview

This project is an end-to-end fraud detection system built to demonstrate machine learning, data processing, model evaluation, explainability, and later MLOps deployment.

## Version 1: Classical Machine Learning Fraud Detection

Version 1 focuses on:

- Exploratory Data Analysis
- Data preprocessing
- Feature engineering
- Imbalanced dataset handling
- Classical ML model training
- XGBoost / CatBoost modeling
- Proper model evaluation
- SHAP explainability

## Problem Statement

Fraud detection is a binary classification problem where the goal is to predict whether a transaction is fraudulent or legitimate.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- SHAP
- Jupyter Notebook

## Current Status

Version 1 setup is in progress.

## Project Notes
Version-wise learning and implementation notes are stored in:
```text
docs/VERSION_1_PROGRESS_NOTES.md

## Threshold Tuning

Fraud detection should not rely blindly on the default classification threshold of `0.5`.

This project tests multiple fraud probability thresholds and selects a better threshold based on F1-score.

Generated threshold tuning files:

```text
reports/threshold_tuning_results.csv
reports/threshold_tuning_summary.json
reports/threshold_tuning_report.md