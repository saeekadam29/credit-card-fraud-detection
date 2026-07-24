# Fraud Detection Risk Analytics 

Machine learning pipeline that flags fraudulent credit card transactions in a highly imbalanced, real-world-style dataset — built with a strict train/test leakage boundary, so every engineered feature would hold up if deployed on transactions the model has never seen.

## Problem

Credit card fraud detection is a **needle-in-a-haystack** classification problem: fraud makes up only **0.172%** of transactions (492 out of 284,807). A model that predicts "not fraud" every time would be 99.8% accurate and completely useless — so this project is built around metrics and techniques that actually matter for rare-event detection: ROC-AUC, PR-AUC, recall on the fraud class, and SMOTE-based resampling, instead of raw accuracy.

## Dataset

- **284,807** anonymized European credit card transactions
- 28 PCA-transformed features (`V1`–`V28`) + `Time` and `Amount`
- Target: `Class` (1 = fraud, 0 = legitimate)
- No missing values

## Approach

**1. Leakage-safe feature engineering**
Every engineered feature is split into two categories on purpose:
- *Row-level, safe pre-split*: `hour` (from `Time`), `log_amount`
- *Distribution-dependent, computed from the training set only and applied to test*: `amount_rank` (percentile rank), `is_high_value` (95th-percentile flag), `txn_velocity_per_hour` (transaction counts per hour)

This mirrors how a model would actually behave in production — it never sees statistics derived from data it shouldn't have access to yet.

**2. Train/test split**
Stratified 80/20 split (227,845 train / 56,962 test) to preserve the fraud ratio in both sets.

**3. Class imbalance — SMOTE**
SMOTE oversampling applied **only to the training set** (balanced to 227,451 / 227,451) — the test set is left untouched so evaluation reflects real-world class distribution.

**4. Models**
| Model | ROC-AUC | PR-AUC | F1 (fraud class) | Recall (fraud) | Precision (fraud) |
|---|---|---|---|---|---|
| Logistic Regression | 0.9662 | 0.7290 | 0.125 | 0.91 | 0.07 |
| **Random Forest** | **0.9854** | **0.8074** | **0.601** | 0.88 | 0.46 |

Random Forest (100 trees, max depth 10) was the stronger model, catching 88% of fraud cases while cutting false positives dramatically compared to logistic regression.

**5. Feature importance**
Top drivers of fraud predictions: `V14`, `V4`, `V10`, `V12`, `V17` — anonymized PCA components, consistent with prior published work on this dataset.
<img width="1200" height="900" alt="feature_importance" src="https://github.com/user-attachments/assets/6113ac37-f122-47e1-a38e-6704b823eddf" />

<img width="900" height="600" alt="01_class_distribution1" src="https://github.com/user-attachments/assets/56ed01ff-154d-42ee-84c8-96e1e18ba694" />

<img width="750" height="600" alt="cm_logistic_regression" src="https://github.com/user-attachments/assets/e7d75d6d-cf0a-4501-a262-d5a25e78cfef" />

<img width="750" height="600" alt="cm_random_forest" src="https://github.com/user-attachments/assets/bd488b49-c02f-4686-8ac4-d6f550776399" />

<img width="1800" height="750" alt="model_comparison_curves" src="https://github.com/user-attachments/assets/779c55cb-a237-4d0c-af77-4430047180f9" />

## Repo structure
├── final_fraud_detection.ipynb   # full analysis notebook
├── 01_class_distribution.png     # class imbalance visualization
├── cm_logistic_regression.png    # confusion matrix — logistic regression
├── cm_random_forest.png          # confusion matrix — random forest
├── model_comparison_curves.png   # ROC + precision-recall curves
├── feature_importance.png        # top 15 feature importances (RF)
└── README.md
```
## Tech stack
Python · pandas · NumPy · scikit-learn · imbalanced-learn (SMOTE) · matplotlib · seaborn
## How to run
bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
jupyter notebook final_fraud_detection.ipynb 
Chose evaluation metrics appropriate for a 0.17%-positive-class problem instead of misleading accuracy
Compared an interpretable baseline (Logistic Regression) against a stronger non-linear model (Random Forest) and explained the trade-off in precision vs. recall
