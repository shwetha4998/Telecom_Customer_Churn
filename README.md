# Telecom_Customer_Churn
# Telecom Customer Churn Prediction
A full-cycle machine learning pipeline to identify customers at risk of churning — from raw data to a production-ready inference function.

---

## Overview

Customer churn is one of the costliest problems in the telecom industry. This project builds an end-to-end binary classification pipeline to predict whether a customer will churn, using a real-world telecom dataset with **7,043 customers** and **38 features**.

The pipeline covers everything: exploratory analysis, preprocessing, model training, evaluation, cross-validation, explainability, serialisation, and inference.

---

## Dataset Summary

| Property | Detail |
|---|---|
| Rows | 7,043 |
| Features | 38 |
| Target Column | `Customer Status` |
| Churned | 1,869 (26.5%) |
| Stayed | 4,720 |
| Joined | 454 |
| Task | Binary Classification: Churned (1) vs Not-Churned (0) |

---

## Project Structure

```
telecom-churn-prediction/
│
├── data/
│   └── telecom_churn.csv           # Raw dataset
│
├── notebooks/
│   └── churn_analysis.ipynb        # Full EDA + modelling notebook
│
├── models/
│   ├── best_model.pkl              # Serialised best model
│   ├── scaler.pkl                  # Fitted StandardScaler
│   └── label_encoders.pkl          # LabelEncoder dictionary
│
├── outputs/
│   └── model_comparison.csv        # Metrics for all trained models
│
├── src/
│   ├── preprocess.py               # Preprocessing pipeline
│   ├── train.py                    # Model training script
│   ├── evaluate.py                 # Evaluation utilities
│   └── inference.py                # predict_churn() inference function
│
├── requirements.txt
└── README.md
```

---

## Pipeline Walkthrough

### Step 1 — Data Inspection

- Loaded dataset with `pd.read_csv()`; inspected shape, column names, and dtypes
- Used `.info()` for memory usage and `.describe()` for summary statistics
- Checked value counts for `Customer Status`, `Churn Category`, and `Churn Reason`

---

### Step 2 — Exploratory Data Analysis (EDA)

- Plotted churn distribution (bar chart + pie chart)
- Compared monthly charges, tenure, and total revenue across churn groups
- Analysed contract type vs churn rate — **Month-to-Month** customers churn significantly more
- Generated a correlation heatmap for all numeric features
- Visualised missing value patterns
- Examined churn breakdown by **Internet Type**, **Offer**, and **Payment Method**

---

### Step 3 — Data Preprocessing

| Sub-step | Action |
|---|---|
| **3a — Drop Leakage Columns** | Removed `Customer ID`, `City`, `Zip Code`, `Latitude`, `Longitude`, `Churn Category`, `Churn Reason` |
| **3b — Binary Target** | `Churned → 1`, `Stayed → 0`, `Joined → 0` |
| **3c — Missing Values** | Categorical: fill with `'Unknown'`; Numerical: fill with column median |
| **3d — Encoding** | `LabelEncoder` on all object columns; stored in a dict for inference reuse |
| **3e — Feature Engineering** | `Avg_Revenue_per_Month = Total Revenue / (Tenure + 1)` <br> `Charge_to_Tenure_Ratio = Monthly Charge / (Tenure + 1)` |
| **3f — Train/Val/Test Split** | 60% Train \| 20% Validation \| 20% Test (stratified) |
| **3g — Feature Scaling** | `StandardScaler` fit on train only, applied to all splits |
| **3h — Class Imbalance** | SMOTE applied on training data only (fallback: `class_weight='balanced'`) |

---

### Step 4 — Model Training

Multiple classifiers trained on SMOTE-augmented training data and evaluated on the validation set:

- Logistic Regression
- K-Nearest Neighbours
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Machine
- Neural Network (MLP)

---

### Step 5 — Model Evaluation

| Metric | Why It Matters for Churn |
|---|---|
| **Accuracy** | Overall correctness — can mislead with imbalance |
| **Precision** | Cost of false alarms (wasted retention spend) |
| **Recall** | Cost of missing actual churners (lost revenue) |
| **F1-Score** | Balanced precision-recall trade-off |
| **AUC-ROC** | Discriminative power: churner vs non-churner |
| **Confusion Matrix** | Full TP / FP / TN / FN breakdown |

---

### Step 6 — Cross-Validation

- **5-fold Stratified Cross-Validation** on the top 3 models by F1
- Reports mean F1 and standard deviation across folds
- Used to detect overfitting and produce reliable performance estimates

---

### Step 7 — Final Test Set Evaluation

- Best model evaluated on the **held-out test set** (unseen during training)
- Full classification report (precision, recall, F1 per class)
- ROC curve plotted with AUC-ROC score
- Confusion matrix visualised

---

### Step 8 — Feature Importance & Explainability

- Feature importances extracted from tree-based models (Random Forest, Gradient Boosting, XGBoost)
- Permutation importance used for non-tree models
- Top 15 features plotted by importance score
- **SHAP values** used for deeper model explainability (advanced)

---

### Step 9 — Model Serialisation

```python
import joblib

joblib.dump(best_model,      'models/best_model.pkl')
joblib.dump(scaler,          'models/scaler.pkl')
joblib.dump(label_encoders,  'models/label_encoders.pkl')

# Model comparison saved
results_df.to_csv('outputs/model_comparison.csv', index=False)
```

---

### Step 10 — Inference Pipeline

A `predict_churn()` function accepts raw customer data, applies the full preprocessing pipeline (encoding → feature engineering → scaling), and returns:

- **Churn probability** (float)
- **Binary prediction** (0 = Stay, 1 = Churn)

```python
from src.inference import predict_churn

result = predict_churn(new_customer_data)
# {'churn_probability': 0.83, 'prediction': 1}
```

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/your-username/telecom-churn-prediction.git
cd telecom-churn-prediction

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt** includes:
```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
matplotlib
seaborn
shap
joblib
```

---

## Key Insights

- **Month-to-Month contract** customers have the highest churn rate
- **High monthly charges** with low tenure strongly correlate with churn
- **Offer type** and **Internet type** are significant churn predictors
- SMOTE improved recall for the minority (churned) class without data leakage

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)
![Pandas](https://img.shields.io/badge/Pandas-DataFrames-150458?logo=pandas)

---

## Author

**Shwetha Elavarasan**
MSc Big Data Analytics & AI | MLOps & Cloud-Native AI
[LinkedIn](https://www.linkedin.com/in/shwetha-elavarasan-7a923a189)) · [GitHub]((https://github.com/shwetha4998))

---
