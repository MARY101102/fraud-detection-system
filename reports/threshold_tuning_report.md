# Threshold Tuning Report

## Project

End-to-End Fraud Detection System

## Version

Version 1 — Classical Machine Learning Fraud Detection

---

## Goal

The goal of threshold tuning is to find a better fraud decision threshold instead of using the default threshold of `0.5`.

Fraud detection is highly imbalanced, so the default threshold may not give the best balance between fraud detection and false alarms.

---

## Best Model

`xgboost_undersampled`

---

## Selected Threshold

`0.95`

This threshold was selected based on the highest F1-score.

---

## Metrics at Selected Threshold

| Metric | Value |
|---|---:|
| Precision | 0.941413 |
| Recall | 0.997565 |
| F1-score | 0.968676 |
| True Negatives | 1270779 |
| False Positives | 102 |
| False Negatives | 4 |
| True Positives | 1639 |

---

## Top Thresholds by F1-score

|   threshold |   precision |   recall |   f1_score |   true_negatives |   false_positives |   false_negatives |   true_positives |   false_positive_rate |   false_negative_rate |
|------------:|------------:|---------:|-----------:|-----------------:|------------------:|------------------:|-----------------:|----------------------:|----------------------:|
|        0.95 |    0.941413 | 0.997565 |   0.968676 |      1.27078e+06 |               102 |                 4 |             1639 |           8.02593e-05 |            0.00243457 |
|        0.9  |    0.907028 | 0.997565 |   0.950145 |      1.27071e+06 |               168 |                 4 |             1639 |           0.000132192 |            0.00243457 |
|        0.85 |    0.889853 | 0.998174 |   0.940906 |      1.27068e+06 |               203 |                 3 |             1640 |           0.000159732 |            0.00182593 |
|        0.8  |    0.874667 | 0.998174 |   0.932348 |      1.27065e+06 |               235 |                 3 |             1640 |           0.000184911 |            0.00182593 |
|        0.75 |    0.861345 | 0.998174 |   0.924725 |      1.27062e+06 |               264 |                 3 |             1640 |           0.00020773  |            0.00182593 |
|        0.7  |    0.853722 | 0.998174 |   0.920314 |      1.2706e+06  |               281 |                 3 |             1640 |           0.000221106 |            0.00182593 |
|        0.65 |    0.848062 | 0.998783 |   0.917272 |      1.27059e+06 |               294 |                 2 |             1641 |           0.000231336 |            0.00121729 |
|        0.6  |    0.843268 | 0.998783 |   0.914461 |      1.27058e+06 |               305 |                 2 |             1641 |           0.000239991 |            0.00121729 |
|        0.55 |    0.836391 | 0.998783 |   0.910402 |      1.27056e+06 |               321 |                 2 |             1641 |           0.000252581 |            0.00121729 |
|        0.5  |    0.831728 | 0.998783 |   0.907633 |      1.27055e+06 |               332 |                 2 |             1641 |           0.000261236 |            0.00121729 |

---

## Generated Figures

- `reports/figures/threshold_precision_recall_f1.png`
- `reports/figures/threshold_confusion_values.png`

---

## Interpretation

Lower thresholds usually increase recall, meaning more fraud cases are detected.

However, lower thresholds can also increase false positives, meaning more legitimate transactions are flagged as fraud.

Higher thresholds usually reduce false positives but may miss more fraud cases.

The selected threshold is a balanced starting point. In a real banking system, the final threshold should be selected based on business cost, analyst review capacity, and customer impact.

---

## Next Step

Step 10 — SHAP Explainability
