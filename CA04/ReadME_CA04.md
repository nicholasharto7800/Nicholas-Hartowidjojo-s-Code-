
# CA04 — Ensemble Models on Census Income (with CA03-style Preprocessing)

This repository contains reproducible code that loads a Census-style income dataset, performs light cleaning, constructs train/test splits, and evaluates four ensemble classifiers—Random Forest, AdaBoost, Gradient Boosting, and XGBoost—by sweeping the key hyperparameter `n_estimators`. It generates Accuracy and ROC–AUC curves vs. `n_estimators` and produces a numbers-only comparison matrix across models.

## Dataset

- Source (exact path used in the code):
  `https://github.com/ArinB/MSBA-CA-03-Decision-Trees/blob/master/census_data.csv?raw=true`
- Task: Binary classification of income (e.g., `<=50K` vs. `>50K`).

## Environment

- Python >= 3.9
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
- Optional: xgboost (for XGBoost sweeps)

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scriptsctivate
# macOS/Linux:
source .venv/bin/activate
pip install -U pandas numpy matplotlib seaborn scikit-learn xgboost
```

## How to Run

1. Run the notebook or script in order:
   - Part 1: Load data, minimal cleaning, detect target and split columns, create train/test sets (`x_train`, `x_test`, `y_train`, `y_test`).
   - Part 2: Random Forest sweep over `n_estimators`; plot Accuracy and AUC vs. `n_estimators`.
   - Part 4: Repeat the sweep for AdaBoost, Gradient Boosting, and XGBoost.
   - Part 5: Build a numbers-only comparison matrix (rows: Accuracy, AUC; columns: Random Forest, AdaBoost, Gradient Boost, XGB).

2. Interpret the plots and matrix, and record observations for your report.

## Code Highlights

### Part 1 — Data Loading & Minimal Cleaning
- Loads CSV from the exact URL above.
- Prints top categories for object columns and strips whitespace.
- Heuristically detects target and split columns using common names/patterns.
- Maps labels to 0/1 (`>50K` -> 1, `<=50K` -> 0).
- Builds train/test frames and selects `feature_cols`.

### Part 2 — Random Forest (`n_estimators` sweep)
- `n_estimators_options = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]`.
- `ColumnTransformer` with `OneHotEncoder(handle_unknown='ignore')` over `feature_cols`.
- For each `n_estimators` value:
  - Fit `Pipeline(preprocessor -> RandomForestClassifier)` on `x_train, y_train`.
  - Compute Accuracy and ROC–AUC on `x_test` and store.
- Plot Accuracy vs. `n_estimators` and AUC vs. `n_estimators`.

### Part 4 — AdaBoost, Gradient Boosting, XGBoost
- Same sweep process using `AdaBoostClassifier`, `GradientBoostingClassifier`, and `XGBClassifier`.
- For XGBoost, sets `eval_metric='logloss'` for compatibility with modern versions.

### Part 5 — Numbers-only Comparison Matrix
- Produces a compact matrix with rows `Accuracy`, `AUC` and columns `Random Forest`, `AdaBoost`, `Gradient Boost`, `XGB` containing the best numeric scores for each model.

## Expected Outputs

1. Two figures per model:
   - Accuracy vs. `n_estimators`
   - AUC vs. `n_estimators`
2. A final comparison matrix summarizing the best Accuracy and AUC across models.

## Tips and Troubleshooting

- Keep the same preprocessing (`ColumnTransformer`) across models for fair comparison.
- Ensure `feature_cols` excludes label and split columns (for example, `flag`, `y`).
- If XGBoost is not installed or available, skip that section or install `xgboost`.
- If you see encoder shape or category errors, confirm you are passing raw feature DataFrames to the Pipeline and that the encoder includes all predictor columns.

## License

MIT (or your preferred license).

## Acknowledgments

- Dataset path and assignment framing are based on the referenced course materials.
- Thanks to the maintainers of scikit-learn and XGBoost.
