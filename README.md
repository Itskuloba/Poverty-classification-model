---

# Income Classification Model — README

---

## Project Overview

**Goal:** Train a machine learning classifier to predict whether an individual earns more than `$50K` per year (`1`) or less/equal to `$50K` (`0`) using demographic and employment-related features.

**Intended users:** data scientists, economists, policymakers, and organizations interested in income distribution, labor market analysis, and demographic-economic interactions.

---

## Why this matters (Problem Definition)

* Income inequality and labor economics are central themes in economics and public policy.
* Traditional statistical studies provide descriptive insights, but **machine learning enables predictive modeling** to better understand patterns of income distribution.
* A robust classification model can highlight the demographic and socio-economic factors most associated with higher income, supporting **evidence-based policies** around education, training, and labor markets.

---

## Data Sources

* **Adult Dataset (UCI Machine Learning Repository)** — U.S. Census income data, 1994.
  🔗 [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)

**Features include:** age, workclass, education, marital status, occupation, relationship, race, sex, capital gain/loss, hours worked per week and native country.

**Target variable:** `income` (<=50K, >50K).

---

## Full ML Pipeline

1. **Problem Definition** — binary classification (income >50K or <=50K).
2. **Data Collection** — load dataset from UCI repository.
3. **Data Cleaning & EDA** — handle missing values, check outliers, analyze class distribution .
4. **Feature Engineering** — encode categorical variables (education, workclass, occupation), normalize continuous variables (age, hours, capital-gain/loss).
5. **Modeling** — baseline logistic regression; advanced models like Random Forest, Gradient Boosting (XGBoost, LightGBM).
6. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC; analyze errors by gender, education, and occupation groups.
7. **Deployment** — Streamlit app for predictions.

---

## Modeling Choices & Justification

* **Baseline:** Logistic Regression ; interpretable, establishes benchmark.
* **Primary models:** Random Forest, Gradient Boosting ; capture non-linear relationships and interactions among features.
* **Class imbalance:** Apply SMOTE or use class weighting.
* **Hyperparameter tuning:** `GridSearchCV` or `RandomizedSearchCV` for optimizing tree depths, learning rates and regularization.

---

## Data Cleaning & Transformation Notes

* Handle missing values features.
* Encode categorical variables.
* Scale continuous variables for models requiring normalization.
* Combine related categories where sparse.

---

## Evaluation

Evaluation will consider both accuracy and fairness:

* **Primary metric:** F1-score for the >50K class (since class imbalance exists).
* **Secondary metrics:** Precision, Recall, ROC-AUC.
* **Bias/fairness analysis:** Evaluate performance across subgroups  to detect systemic bias.
* **Target accuracy:** Aim for **> 80–85%** test accuracy.

---

## Credits

* **Author:** Nicholus Kuloba
* **Data:** UCI Machine Learning Repository — *Adult Census Income Dataset*

---






