# Poverty Classification Model — README

---

## Project overview

**Goal:** Train an ML classifier to predict whether a household is `poor` (1) or `non-poor` (0) using household survey variables (income/expenditure, demographics, assets, region, etc.).
**Intended users:** policymakers, NGOs, researchers who need scalable, faster poverty targeting tools.

---

## Why this matters (problem definition)

* Poverty measurement is central to development economics and effective policy targeting.
* Surveys are costly and infrequent — ML gives a scalable, lower-cost way to generate timely poverty estimates and reduce exclusion/inclusion errors in transfer programs.
* The model aims to maximize recall (minimize missed poor households) while maintaining acceptable precision (avoid misallocating scarce resources).

---

## Data sources

Primary :

* **KIHBS 2015/16 (Kenya)** — KNBS microdata. ([https://statistics.knbs.or.ke/nada/index.php/catalog/KIHBS](https://statistics.knbs.or.ke/nada/index.php/catalog/KIHBS))
* **LSMS / World Bank Microdata** — multiple countries. ([https://microdata.worldbank.org](https://microdata.worldbank.org))
* **DHS Program** — demographic & household variables. ([https://dhsprogram.com/data/](https://dhsprogram.com/data/))

---


## Full ML pipeline 

1. **Problem Definition** — binary classification (poor vs non-poor); explain economic significance.
2. **Data Collection** — datasets gathered from KIHBS/LSMS/DHS.
3. **Data Cleaning & EDA** — missing values, outliers, distribution checks, class balance analysis.
4. **Feature Engineering** — per-capita consumption, household size-adjusted variables, region dummies, education encoding.
5. **Modeling** — logistic regression baseline and tree-based models.
6. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, PR curve; confusion matrix and error analysis.
7. **Deployment** — Streamlit app.

---

## Modeling choices & justification

* **Baseline:** Logistic Regression — simple and interpretable (policy audiences value interpretability).
* **Primary models:** Random Forest, XGBoost — handle non-linearities and interactions common in household data.
* **Class imbalance:**  SMOTE for oversampling, or set `class_weight='balanced'` / `scale_pos_weight` for tree models.
* **Hyperparameter tuning:** `RandomizedSearchCV` / `GridSearchCV` with cross-validation.

---

## Data cleaning & transformation notes


* Standardize variable names and units (KES, per capita amounts).
* Impute missing values: median for continuous; mode/`Unknown` for categorical. Document percent missing per column.
* Create derived features: if needed.
* Encode categoricals: one-hot for region/education, ordinal encoding for education levels.
* Scale continuous features for models that need it.

---

## Evaluation 

Evaluation focuses on policy-relevant metrics:

* **Primary:** Recall (sensitivity).
* **Secondary:** Precision, F1-score, ROC-AUC, PR-AUC.
* **Additional:** Confusion matrix stratified by region and urban/rural to detect bias.
* **Target accuracy:** Aim for **> 75–99%** overall accuracy.

Include robust error analysis:

* Which groups are misclassified most? (e.g., female-headed households, remote regions)

---

## Credits

* **Author:** Nicholus Kuloba 
* **Data:** KNBS (KIHBS), World Bank (LSMS), DHS .

---




