## Problem Statement

Predict whether a passenger survived the Titanic shipwreck using attributes like age, gender, ticket class, and boarding port. This is a binary classification task based on historical data.

---

## Libraries Used

All standard libraries were used at the start:

* `pandas`, `numpy` for data handling
* `seaborn`, `matplotlib` for EDA and visualizations
* `scikit-learn`, `xgboost` for modeling
* Custom preprocessing in `titanic_preprocess.py`

---

## Exploratory Data Analysis (EDA)

Key insights from visual exploration:

* **Survival Distribution:** 1.6x more people died than survived.
* **Sex & Survival:** Males died far more; females survived more — visualized via bar plot.
* **Embarkation Port:** Majority boarded at **Southampton (S)**; it also had the highest death count due to sheer volume. Visualized using labeled bar plots and a donut chart (72.4% from S).
* **Correlation Matrix:** No strong linear correlation, but categorical features held deeper patterns.

---

## Feature Engineering

Created **4 impactful new features**:

1. **Deck (from Cabin):** Mapped cabin letters. Found mid-decks (B to F) had over 50% survival rate.
2. **Title (from Name using Regex):** Titles revealed gender, marital status, social class.

   * *E.g.*: Married women had nearly **80% survival**.
   * Visualized via combo of bar plot (count) and line plot (survival rate).
3. **FamilySize (SibSp + Parch + 1)**
4. **FamilyCategory (based on FamilySize):**

   * Solo: 1
   * Small: 2–4 → highest survival (>57%)
   * Medium: 5–8
   * Large: >8

These transformations uncovered hidden patterns that significantly improved model performance.

---

## Data Cleaning & Preprocessing

* Filled missing **Embarked** with 'S' (most common).
* Imputed missing **Age** with median.
* Filled 687/891 missing **Cabin** values with `"U"` (unknown).
* Created a reusable script `titanic_preprocess.py` to:

  * Extract features (Title, Deck, FamilySize)
  * Fill missing values
  * Drop irrelevant columns (e.g., Name, Ticket)
  * One-hot encode categoricals
  * Standardize numeric fields for models that need scaling

---

## Modeling

### 1. **Logistic Regression**

* Initially failed to converge with default solver (`lbfgs`).
* Improved by scaling data and tweaking regularization (`C`).
* **CV Accuracy:** 82.94%
* **Kaggle Score:** 77.51%
* **Why strong?** Titanic has linearly separable features (Sex, Title, Pclass), ideal for Logistic Regression.

### 2. **Decision Tree**

* **CV Accuracy:** 79.9%

### 3. **Random Forest**

* **CV Accuracy:** 80%

### 4. **XGBoost**

* **CV Accuracy:** 81%

### 5. **Stacked Voting Classifier (Ensemble)**

* Combined Logistic Regression, Random Forest, and XGBoost using soft voting.
* **Final CV Accuracy:** 83%

### 6. **Random Forest + GridSearchCV**

* Tuned hyperparameters:

  * 300 estimators
  * max\_depth=10
  * min\_samples\_split=10
* **Improved CV Accuracy:** \~82.7%
* **Best Kaggle Score:** **78.46%**

---


## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

Dataset can be found on this [link](https://www.kaggle.com/competitions/titanic). 
