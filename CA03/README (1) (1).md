# Census Income Classification with Decision Trees

## Overview
This project develops and evaluates a Decision Tree classification model to predict whether an individual's annual income exceeds $50,000 using demographic and employment-related census features. The workflow includes data inspection, preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, and prediction on new observations.

The goal is to build an interpretable model that balances predictive performance with generalization while avoiding overfitting.

---

## Dataset
The dataset contains census information such as:

- Age
- Education level
- Occupation
- Work class
- Hours worked per week
- Capital gains/losses
- Race and gender groupings
- Predefined train/test split column
- Target variable: income category (<=50K or >50K)

The target variable is encoded as:

- 0 → <=50K
- 1 → >50K

---

## Motivation for Discretization

### Why discretize?
Binning numeric features improves:

- Interpretability
- Robustness to outliers
- Stability of decision tree splits
- Generalization performance
- Simpler decision boundaries

### Risks without discretization
Without binning:

- Trees may overfit to small numeric changes
- Thresholds become unstable
- Models become harder to interpret
- Decision paths grow unnecessarily complex

---

## Methodology

### 1. Data Quality Assessment
- Missing value checks
- Duplicate detection
- Data type inspection
- Descriptive statistics
- Category frequency analysis

### 2. Preprocessing
A preprocessing pipeline is built using a ColumnTransformer:

Numeric features:
- Median imputation

Categorical features:
- Mode imputation
- One‑Hot Encoding

### 3. Model
Decision Tree Classifier

---

## Baseline Model
A default Decision Tree is first trained to establish a performance benchmark.

Metrics reported:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Hyperparameter Tuning

Sequential tuning is performed to improve generalization:

| Step | Parameter | Values Tested |
|------|-----------|---------------|
| 1 | criterion | gini, entropy |
| 2 | min_samples_leaf | 5–40 |
| 3 | max_features | None, 0.3–0.8, auto |
| 4 | max_depth | 2–16 |

Primary selection metric: Accuracy

---

## Best Model

Final hyperparameters:

{
  'criterion': 'gini',
  'min_samples_leaf': 20,
  'max_features': 0.8,
  'max_depth': 10
}

These settings provide:
- Controlled tree growth
- Reduced noise sensitivity
- Improved generalization
- Lower risk of overfitting

---

## Evaluation

Performance is measured on the held‑out test set using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

Visualizations:
- Confusion matrix heatmap
- Decision tree diagram

---

## Tree Interpretation

Observations from the learned tree:

- Most informative features appear near the root
- Education, occupation, and work characteristics strongly influence predictions
- Regularization parameters limit depth and prevent memorization of noise

---

## Runtime
Training the best model takes approximately 8–10 seconds on a standard laptop.

---

## Overfitting Assessment

Regularization checks:

- max_depth = 10
- min_samples_leaf = 20

These values indicate controlled complexity and reduced overfitting risk.

---

## Predicting New Data

Example prediction workflow:

1. Apply identical preprocessing
2. Transform features
3. Predict class and probability

The predicted probability represents model confidence, not guaranteed correctness.

---

## How to Run

### Install dependencies

pip install pandas numpy matplotlib seaborn scikit-learn graphviz

### Execute

python decision_tree_income.py

or run the Jupyter notebook.

---

## Project Structure

├── CA03_Decision_Tree_Nicholas_Thomas
├── README.md
└── data.csv

---

## Key Takeaways

- Decision Trees provide strong interpretability
- Feature binning improves stability
- Sequential tuning improves performance
- Regularization reduces overfitting
- Visualization aids explanation of model logic

---

## Author

MSBA Decision Trees Project
