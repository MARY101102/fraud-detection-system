# Model Evaluation Report

## Project

End-to-End Fraud Detection System

## Version

Version 1 — Classical Machine Learning Fraud Detection

---

## Evaluation Goal

The goal of this evaluation is to compare classical machine learning models for fraud detection and select the best model using fraud-focused metrics.

Because the dataset is highly imbalanced, accuracy is not used as the main metric.

---

## Best Model

Selected best model:

`xgboost_undersampled`

The model was selected based mainly on **PR-AUC**, because PR-AUC is more meaningful for imbalanced fraud detection problems.

---

## Best Model Metrics

| Metric | Value |
|---|---:|
| Precision | 0.831728 |
| Recall | 0.998783 |
| F1-score | 0.907633 |
| ROC-AUC | 0.999870 |
| PR-AUC | 0.998591 |

---

## Confusion Matrix Values

| Value | Count |
|---|---:|
| True Negatives | 1270549 |
| False Positives | 332 |
| False Negatives | 2 |
| True Positives | 1641 |

---

## Best Model Row From Comparison Table

| model_name           |   training_rows |   precision |   recall |   f1_score |   roc_auc |   pr_auc |   true_negatives |   false_positives |   false_negatives |   true_positives |
|:---------------------|----------------:|------------:|---------:|-----------:|----------:|---------:|-----------------:|------------------:|------------------:|-----------------:|
| xgboost_undersampled |           39420 |    0.831728 | 0.998783 |   0.907633 |   0.99987 | 0.998591 |          1270549 |               332 |                 2 |             1641 |

---

## Full Model Comparison

| model_name                   |   training_rows |   precision |   recall |   f1_score |   roc_auc |   pr_auc |   true_negatives |   false_positives |   false_negatives |   true_positives |
|:-----------------------------|----------------:|------------:|---------:|-----------:|----------:|---------:|-----------------:|------------------:|------------------:|-----------------:|
| xgboost_undersampled         |           39420 |   0.831728  | 0.998783 |  0.907633  |  0.99987  | 0.998591 |          1270549 |               332 |                 2 |             1641 |
| random_forest_balanced       |          250000 |   0.997559  | 0.995131 |  0.996344  |  0.998649 | 0.997542 |          1270877 |                 4 |                 8 |             1635 |
| catboost_weighted            |          250000 |   0.993321  | 0.99574  |  0.994529  |  0.999487 | 0.997196 |          1270870 |                11 |                 7 |             1636 |
| xgboost_weighted             |          250000 |   0.996346  | 0.99574  |  0.996043  |  0.998332 | 0.996432 |          1270875 |                 6 |                 7 |             1636 |
| logistic_regression_balanced |          250000 |   0.0306799 | 0.980523 |  0.0594981 |  0.994408 | 0.62263  |          1219982 |             50899 |                32 |             1611 |

---

## Generated Figures

The following plots were generated:

- `reports/figures/confusion_matrix_best_model.png`
- `reports/figures/roc_curve_best_model.png`
- `reports/figures/precision_recall_curve_best_model.png`
- `reports/figures/model_comparison_pr_auc.png`
- `reports/figures/model_comparison_recall.png`

---

## Interpretation

The model evaluation focuses on fraud-specific metrics:

- **Recall** shows how many actual fraud transactions were detected.
- **Precision** shows how many predicted fraud cases were actually fraud.
- **F1-score** balances precision and recall.
- **ROC-AUC** measures general class separation.
- **PR-AUC** is especially important because fraud data is highly imbalanced.

In fraud detection, false negatives are risky because fraud transactions are missed.
False positives are also important because they create unnecessary manual review and may disturb real customers.

---

## Important Note

This evaluation uses the default classification threshold of `0.5`.

The next step is threshold tuning, where different probability thresholds will be tested to find a better fraud decision boundary.

---

## Next Step

Step 9 — Threshold Tuning
